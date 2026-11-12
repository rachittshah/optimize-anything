#!/usr/bin/env python3
"""optimize-anything doctor: check the MCP server install for each coding agent.

Per agent (only those installed): (1) entry registered + config parses, (2) the configured
command starts with the env that agent would pass and lists the 5 tools over MCP stdio,
(3) a free 2-iteration toy optimization (mock backend, python evaluator) completes through
the server, (4) the skill file is present. Default checks spend no model tokens.

  scripts/doctor.py [--agents claude,codex,opencode,gemini] [--live claude,codex,...] [--json]

--live runs the agent's own headless mode once to call list_optimization_runs (spends a few
model tokens). Stdlib only; Python >= 3.11 (tomllib).
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from pathlib import Path

NAME = "optimize-anything"
HOME = Path.home()
TOOLS = {"optimize_anything", "check_optimization", "get_best_candidate", "stop_optimization",
         "list_optimization_runs"}
AGENTS = ("claude", "codex", "opencode", "gemini")
CODEX_DEFAULT_ENV = ("HOME", "LANG", "LOGNAME", "PATH", "SHELL", "TERM", "TMPDIR", "USER",
                     "__CF_USER_TEXT_ENCODING")
GEMINI_REDACT = re.compile(r"TOKEN|SECRET|PASSWORD|KEY|AUTH|CREDENTIAL", re.I)
SKILL_PATHS = {
    "claude": HOME / ".claude/skills" / NAME / "SKILL.md",
    "codex": HOME / ".agents/skills" / NAME / "SKILL.md",
    "opencode": HOME / ".agents/skills" / NAME / "SKILL.md",
    "gemini": HOME / ".agents/skills" / NAME / "SKILL.md",
}
TOY_EVALUATOR = (
    "def evaluate(candidate, example):\n"
    "    t = candidate if isinstance(candidate, str) else ' '.join(candidate.values())\n"
    "    n = len([ln for ln in t.splitlines() if ln.strip()])\n"
    "    return min(1.0, n / 4), {'lines': n, 'hint': 'add more non-empty lines'}\n"
)
LIVE_PROMPT = ("Call the list_optimization_runs tool from the optimize-anything MCP server with "
               "limit 3, then reply with only the value of the `total` field it returned.")


# ---------------------------------------------------------------- config readers

def opencode_files() -> list[Path]:
    d = HOME / ".config/opencode"
    return [p for p in (d / "opencode.json", d / "opencode.jsonc") if p.exists()]


def strip_jsonc(text: str) -> str:
    """Remove // and /* */ comments and trailing commas outside strings."""
    out, i, n, in_str = [], 0, len(text), False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1]); i += 2; continue
            if c == '"':
                in_str = False
        elif c == '"':
            in_str = True; out.append(c)
        elif text.startswith("//", i):
            while i < n and text[i] != "\n":
                i += 1
            continue
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        else:
            out.append(c)
        i += 1
    return re.sub(r",(\s*[}\]])", r"\1", "".join(out))


def read_entry(agent: str) -> tuple[dict | None, str]:
    """(normalized entry {command, args, env, env_vars}, where it was found or error)."""
    try:
        if agent == "claude":
            p = HOME / ".claude.json"
            e = json.loads(p.read_text()).get("mcpServers", {}).get(NAME)
            return (None, f"no user-scope entry in {p}") if not e else (
                {"command": e.get("command"), "args": e.get("args", []), "env": e.get("env", {})}, str(p))
        if agent == "codex":
            p = HOME / ".codex/config.toml"
            e = tomllib.loads(p.read_text()).get("mcp_servers", {}).get(NAME)
            return (None, f"no [mcp_servers.{NAME}] in {p}") if not e else (
                {"command": e.get("command"), "args": e.get("args", []), "env": e.get("env", {}),
                 "env_vars": e.get("env_vars", []),
                 "approval": e.get("default_tools_approval_mode")}, str(p))
        if agent == "opencode":
            for p in opencode_files():
                d = json.loads(strip_jsonc(p.read_text()))
                e = d.get("mcp", {}).get("servers", {}).get(NAME)
                if e:
                    cmd = e.get("command") or []
                    return ({"command": cmd[0] if cmd else None, "args": cmd[1:],
                             "env": e.get("environment", {})}, str(p))
            return None, "no mcp.servers.optimize-anything in ~/.config/opencode/opencode.json(c)"
        if agent == "gemini":
            p = HOME / ".gemini/settings.json"
            e = json.loads(p.read_text()).get("mcpServers", {}).get(NAME)
            return (None, f"no mcpServers.{NAME} in {p}") if not e else (
                {"command": e.get("command"), "args": e.get("args", []), "env": e.get("env", {})}, str(p))
    except FileNotFoundError as ex:
        return None, f"config file missing: {ex.filename}"
    except (json.JSONDecodeError, tomllib.TOMLDecodeError) as ex:
        return None, f"config does not parse: {ex}"
    return None, "unknown agent"


def cli_registered(agent: str) -> tuple[bool, str]:
    """Ask the agent's own CLI whether the server is registered."""
    cmds = {
        "claude": ["claude", "mcp", "get", NAME],
        "codex": ["codex", "mcp", "get", NAME, "--json"],
        "opencode": ["opencode", "debug", "config"],
        "gemini": ["gemini", "mcp", "list"],
    }
    try:
        r = subprocess.run(cmds[agent], capture_output=True, text=True, timeout=60,
                           stdin=subprocess.DEVNULL, cwd=tempfile.gettempdir())
    except (OSError, subprocess.TimeoutExpired) as ex:
        return False, f"{' '.join(cmds[agent])}: {ex}"
    text = r.stdout + r.stderr
    found = NAME in text and r.returncode == 0
    return found, f"`{' '.join(cmds[agent])}` {'lists it' if found else 'does not list it (rc=%d)' % r.returncode}"


def agent_env(agent: str, entry: dict) -> dict[str, str]:
    """The env this agent would give the server process (see docs/research/experts/config-formats.md)."""
    parent = dict(os.environ)
    if agent == "codex":
        env = {k: parent[k] for k in CODEX_DEFAULT_ENV if k in parent}
        env.update({k: parent[k] for k in entry.get("env_vars", []) if k in parent})
    elif agent == "gemini":
        env = {k: v for k, v in parent.items() if not GEMINI_REDACT.search(k)}
    else:
        env = parent
    for k, v in (entry.get("env") or {}).items():
        v = str(v)
        if agent == "gemini":  # $VAR / ${VAR} expansion; undefined -> ""
            v = re.sub(r"\$\{(\w+)\}|\$(\w+)", lambda m: parent.get(m.group(1) or m.group(2), ""), v)
        env[k] = v
    return env


# ---------------------------------------------------------------- MCP stdio client

class McpClient:
    def __init__(self, command: str, args: list[str], env: dict[str, str]):
        self.proc = subprocess.Popen([command, *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, env=env, text=True, bufsize=1,
                                     cwd=tempfile.gettempdir())
        self.q: queue.Queue = queue.Queue()
        self.stderr: list[str] = []
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=lambda: self.stderr.extend(self.proc.stderr), daemon=True).start()
        self.next_id = 0

    def _read(self):
        for line in self.proc.stdout:
            line = line.strip()
            if line:
                try:
                    self.q.put(json.loads(line))
                except json.JSONDecodeError:
                    pass
        self.q.put(None)

    def request(self, method: str, params: dict | None = None, timeout: float = 30) -> dict:
        self.next_id += 1
        rid = self.next_id
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": rid, "method": method,
                                          "params": params or {}}) + "\n")
        self.proc.stdin.flush()
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"{method}: no response in {timeout}s")
            msg = self.q.get(timeout=left)
            if msg is None:
                raise RuntimeError(f"server exited: {''.join(self.stderr)[-500:]}")
            if msg.get("id") == rid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg["result"]

    def notify(self, method: str):
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method}) + "\n")
        self.proc.stdin.flush()

    def call(self, tool: str, arguments: dict, timeout: float = 60) -> tuple[dict | str, bool]:
        res = self.request("tools/call", {"name": tool, "arguments": arguments}, timeout=timeout)
        text = "".join(c.get("text", "") for c in res.get("content", []))
        try:
            return json.loads(text), bool(res.get("isError"))
        except json.JSONDecodeError:
            return text, bool(res.get("isError"))

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def initialize(c: McpClient) -> dict:
    res = c.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                   "clientInfo": {"name": "oa-doctor", "version": "1"}})
    c.notify("notifications/initialized")
    return res


def toy_run(c: McpClient) -> tuple[bool, str]:
    """Dry run, then a 2-iteration mock run through the server; asserts result.json + score."""
    args = {"seed_candidate": {"text": "a dog sat on the mat"},
            "evaluator": {"type": "python", "code": TOY_EVALUATOR},
            "objective": "write more non-empty lines (the free mock proposer appends one per proposal)",
            "config": {"max_iterations": 2, "backend": "mock"}}
    dry, err = c.call("optimize_anything", {**args, "dry_run": True}, timeout=60)
    if err or not isinstance(dry, dict) or not dry.get("ok"):
        return False, f"dry_run failed: {str(dry)[:300]}"
    start, err = c.call("optimize_anything", args)
    if err or not isinstance(start, dict) or "run_id" not in start:
        return False, f"start failed: {str(start)[:300]}"
    run_id, run_dir = start["run_id"], Path(start["run_dir"])
    deadline = time.monotonic() + 180
    st: dict = {}
    while time.monotonic() < deadline:
        st, err = c.call("check_optimization", {"run_id": run_id, "wait_seconds": 20}, timeout=60)
        if err or not isinstance(st, dict):
            return False, f"check_optimization error: {str(st)[:300]}"
        if st.get("status") != "running":
            break
    if st.get("status") != "completed":
        return False, f"run {run_id} ended as {st.get('status')}: {st.get('detail') or st.get('error')}"
    if not (run_dir / "result.json").exists():
        return False, f"run {run_id} completed but {run_dir}/result.json is missing"
    best, seed = st.get("best_score"), st.get("seed_score")
    if not isinstance(best, (int, float)) or not isinstance(seed, (int, float)) or best <= seed:
        return False, f"best_score {best} did not beat seed_score {seed}"
    bc, err = c.call("get_best_candidate", {"run_id": run_id})
    if err or not (run_dir / "best.json").exists():
        return False, f"get_best_candidate failed: {str(bc)[:200]}"
    shutil.rmtree(run_dir, ignore_errors=True)  # toy runs stay out of the user's run list
    return True, f"dry-run ok; run completed in {st.get('elapsed_s')}s, best {best} > seed {seed}"


def server_checks(agent: str, entry: dict) -> list[tuple[str, bool, str]]:
    out = []
    cmd = entry.get("command")
    if not cmd or not (shutil.which(cmd) or Path(cmd).exists()):
        return [("server starts", False, f"command not found: {cmd!r} (re-run scripts/install.sh)")]
    env = agent_env(agent, entry)
    env["OA_RUNS_DIR"] = str(HOME / ".optimize-anything/doctor-runs")
    c = McpClient(cmd, list(entry.get("args", [])), env)
    try:
        info = initialize(c)
        tools = {t["name"] for t in c.request("tools/list")["tools"]}
        missing = TOOLS - tools
        out.append(("server starts + tools/list", not missing,
                    f"{info.get('serverInfo', {}).get('name')} lists {len(tools)} tools"
                    + (f"; missing {sorted(missing)}" if missing else "")))
        if not missing:
            ok, detail = toy_run(c)
            out.append(("toy optimization (mock, free)", ok, detail))
    except Exception as ex:
        out.append(("server starts + tools/list", False,
                    f"{type(ex).__name__}: {ex}; stderr: {''.join(c.stderr)[-300:]}"))
    finally:
        c.close()
    return out


def skill_check(agent: str) -> tuple[str, bool, str]:
    p = SKILL_PATHS[agent]
    if not p.exists():
        return ("skill installed", False, f"missing {p} (run scripts/install.sh)")
    head = p.read_text()[:2000]
    ok = head.startswith("---") and re.search(r"^name:\s*optimize-anything\s*$", head, re.M) \
        and re.search(r"^description:", head, re.M)
    return ("skill installed", bool(ok), str(p) if ok else f"{p} lacks frontmatter name/description")


def check_agent(agent: str) -> list[tuple[str, bool, str]]:
    rows = []
    entry, where = read_entry(agent)
    reg_ok, reg_detail = cli_registered(agent)
    rows.append(("config entry", entry is not None and reg_ok,
                 f"{where}; {reg_detail}" if entry else where))
    if agent == "codex" and entry and entry.get("approval") != "approve":
        rows.append(("codex approval mode", False,
                     'set default_tools_approval_mode = "approve" or codex exec refuses the tools'))
    if entry:
        rows.extend(server_checks(agent, entry))
    rows.append(skill_check(agent))
    return rows


# ---------------------------------------------------------------- live checks

def live_check(agent: str) -> tuple[bool, str]:
    tmp = tempfile.mkdtemp(prefix="oa-live-")
    tool = "list_optimization_runs"
    if agent == "claude":
        cmd = ["claude", "-p", LIVE_PROMPT, "--model", os.environ.get("OA_LIVE_CLAUDE_MODEL", "claude-haiku-5-5"), "--output-format", "stream-json",
               "--verbose", "--no-session-persistence", "--max-budget-usd", "0.5",
               "--allowedTools", f"mcp__{NAME}__{tool}"]
    elif agent == "codex":
        cmd = ["codex", "exec", "--skip-git-repo-check", "--ephemeral", "--json", "-s", "read-only",
               "-m", os.environ.get("OA_LIVE_CODEX_MODEL", "gpt-6-luna"), LIVE_PROMPT]
    elif agent == "opencode":
        # the user's default model may be a small local one; pin a free hosted model (override via env)
        model = os.environ.get("OA_LIVE_OPENCODE_MODEL", "opencode/deepseek-v4.1-flash-free")
        cmd = ["opencode", "run", "--standalone", "--format", "json", "--auto", "-m", model, LIVE_PROMPT]
    else:
        cmd = ["gemini", "-p", LIVE_PROMPT, "-o", "stream-json", "--approval-mode", "yolo",
               "--allowed-mcp-server-names", NAME, "--skip-trust"]
    t0 = time.monotonic()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=90 if agent == "gemini" else 300,
                           cwd=tmp, stdin=subprocess.DEVNULL)
    except FileNotFoundError:
        return False, f"{cmd[0]} not installed"
    except subprocess.TimeoutExpired as ex:
        partial = "".join(x.decode() if isinstance(x, bytes) else (x or "") for x in (ex.stdout, ex.stderr))
        if re.search(r"authenticat|log ?in|sign ?in", partial, re.I):
            return False, f"{agent} is not logged in (it waits for an interactive login); run `{agent}` once to authenticate"
        return False, f"timed out after {ex.timeout}s"
    secs = round(time.monotonic() - t0)
    events = []
    for line in r.stdout.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    called, cost = False, ""
    blob = json.dumps(events)
    if agent == "claude":
        for e in events:
            for c in (e.get("message", {}) or {}).get("content", []) if isinstance(e.get("message"), dict) else []:
                if c.get("type") == "tool_use" and c.get("name") == f"mcp__{NAME}__{tool}":
                    called = True
            if e.get("type") == "result":
                cost = f"${e.get('total_cost_usd', 0):.4f}"
                if e.get("permission_denials"):
                    called = False
    elif agent == "codex":
        for e in events:
            it = e.get("item") or {}
            if e.get("type") == "item.completed" and it.get("type") == "mcp_tool_call" \
                    and it.get("server") == NAME and it.get("tool") == tool and not it.get("error"):
                called = True
            if e.get("type") == "turn.completed":
                u = e.get("usage", {})
                cost = f"{u.get('input_tokens', 0)} in / {u.get('output_tokens', 0)} out tokens"
    elif agent == "opencode":
        # v2 wraps MCP tools in a code-mode `execute` tool: look for the tool name in its input.
        for e in events:
            part = e.get("part") or {}
            if e.get("type") == "tool_use" and tool in json.dumps(part.get("state", {}).get("input", {})) \
                    and part.get("state", {}).get("status") == "completed":
                called = True
        costs = [e.get("part", {}).get("cost", 0) or 0 for e in events if e.get("type") == "step_finish"]
        cost = f"${sum(costs):.4f}"
    else:
        for e in events:
            if e.get("type") == "tool_use" and tool in str(e.get("tool_name", "")):
                called = True
        if not events and r.returncode != 0:
            return False, f"rc={r.returncode} after {secs}s: {(r.stderr or r.stdout)[-300:].strip()}"
    tail = (r.stderr or "")[-200:].strip().replace("\n", " ")
    return called, (f"tool called in {secs}s ({cost})" if called
                    else f"tool NOT called (rc={r.returncode}, {secs}s) {tail or blob[-200:]}")


# ---------------------------------------------------------------- main

FIX = {
    "config entry": "run scripts/install.sh --agents {a}",
    "server starts + tools/list": "cd mcp-server && npm run build; check node path in the entry",
    "toy optimization (mock, free)": "cd engine && uv sync; see stderr.log in the run dir",
    "skill installed": "run scripts/install.sh --agents {a}",
    "codex approval mode": "run scripts/install.sh --agents codex",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", default="", help="comma list (default: all installed)")
    ap.add_argument("--live", default="", help="comma list of agents to check with a real model call")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    agents = [x for x in a.agents.split(",") if x] or [x for x in AGENTS if shutil.which(x)]
    report: dict = {}
    failed = False
    for agent in agents:
        if agent not in AGENTS:
            print(f"unknown agent {agent}", file=sys.stderr)
            return 2
        if not shutil.which(agent):
            report[agent] = [("installed", False, f"{agent} not on PATH")]
            failed = True
            continue
        report[agent] = check_agent(agent)
    live = {}
    for agent in [x for x in a.live.split(",") if x]:
        live[agent] = live_check(agent)
    failed = failed or any(not ok for rows in report.values() for _, ok, _ in rows) \
        or any(not ok for ok, _ in live.values())
    if a.json:
        print(json.dumps({"agents": {k: [dict(check=c, ok=o, detail=d) for c, o, d in v] for k, v in report.items()},
                          "live": {k: dict(ok=o, detail=d) for k, (o, d) in live.items()}}, indent=2))
    else:
        print(f"{'agent':<9} {'check':<32} {'result':<6} detail")
        for agent, rows in report.items():
            for check, ok, detail in rows:
                fix = "" if ok else f"  -> fix: {FIX.get(check, '').format(a=agent)}"
                print(f"{agent:<9} {check:<32} {'PASS' if ok else 'FAIL':<6} {detail}{fix}")
        for agent, (ok, detail) in live.items():
            print(f"{agent:<9} {'live: agent calls the tool':<32} {'PASS' if ok else 'FAIL':<6} {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
