#!/usr/bin/env python3
"""Register the optimize-anything MCP server + skill with coding agents. Called by install.sh.

Safety rules (these are the user's real config files):
- back up every file before writing it, to ~/.optimize-anything/backups/<timestamp>/
- write atomically; re-parse after writing; everything except our entry must be unchanged
  (parsed-value comparison, not counts); on any failure restore the backup
- API keys are referenced by NAME only, never written as values
Stdlib only; Python >= 3.11.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

NAME = "optimize-anything"
HOME = Path.home()
BACKUP_ROOT = HOME / ".optimize-anything" / "backups"
AGENTS = ("claude", "codex", "opencode", "gemini")
# Env var NAMES forwarded to the server where the agent strips the environment.
KEY_VARS = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
OA_VARS = ["OA_BACKEND", "OA_PROPOSER_MODEL", "OA_JUDGE_MODEL", "OA_OPENAI_BASE_URL", "OA_RUNS_DIR"]
CODEX_BEGIN = f"# >>> {NAME} (managed by {NAME}/scripts/install.sh; edits here are overwritten) >>>"
CODEX_END = f"# <<< {NAME} <<<"


class InstallError(RuntimeError):
    pass


class Ctx:
    def __init__(self, repo: Path, node: str, uv: str, dry_run: bool, link: bool):
        self.repo, self.node, self.uv, self.dry_run, self.link = repo, node, uv, dry_run, link
        self.stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.backup_dir = BACKUP_ROOT / self.stamp
        self.log: list[str] = []

    @property
    def server_js(self) -> str:
        return str(self.repo / "mcp-server" / "dist" / "index.js")

    @property
    def base_env(self) -> dict[str, str]:
        return {"OA_ENGINE_DIR": str(self.repo / "engine"), "OA_UV": self.uv}

    def say(self, msg: str) -> None:
        self.log.append(msg)
        print(msg)

    def backup(self, path: Path, label: str) -> Path | None:
        if not path.exists():
            return None
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        dest = self.backup_dir / label
        shutil.copy2(path, dest)
        return dest


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o600
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.oa-")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def without(d: dict, *path: str) -> dict:
    """Deep copy of d with d[path[0]]...[path[-1]] removed (and containers emptied by that)."""
    d = copy.deepcopy(d)
    node, parents = d, []
    for k in path[:-1]:
        if not isinstance(node.get(k), dict):
            return d
        parents.append((node, k))
        node = node[k]
    node.pop(path[-1], None)
    for parent, k in reversed(parents):
        if parent[k] == {}:
            parent.pop(k)
        else:
            break
    return d


def run(cmd: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL,
                          cwd=tempfile.gettempdir())


# ---------------------------------------------------------------- JSON-file agents

def json_load(path: Path) -> tuple[dict, str]:
    if not path.exists():
        return {}, ""
    text = path.read_text()
    try:
        return (json.loads(text) if text.strip() else {}), text
    except json.JSONDecodeError as e:
        raise InstallError(f"{path} is not plain JSON ({e}); it may contain comments. Add the entry by hand "
                           "(see README) or remove the comments.")


def json_indent(text: str) -> int:
    m = re.search(r"\n( +)\S", text)
    return len(m.group(1)) if m else 2


def edit_json(ctx: Ctx, path: Path, label: str, keypath: tuple[str, ...], entry: dict | None) -> str:
    """Set (entry) or remove (None) data[keypath]; verify the rest is unchanged."""
    before, text = json_load(path)
    cur = before
    for k in keypath[:-1]:
        cur = cur.get(k, {}) if isinstance(cur, dict) else {}
    existing = cur.get(keypath[-1]) if isinstance(cur, dict) else None
    if entry is None and existing is None:
        return "not present"
    if entry is not None and existing == entry:
        return "unchanged"
    action = "remove" if entry is None else ("update" if existing is not None else "add")
    if ctx.dry_run:
        return f"would {action} {'.'.join(keypath)} in {path}"
    after = copy.deepcopy(before)
    node = after
    for k in keypath[:-1]:
        node = node.setdefault(k, {})
        if not isinstance(node, dict):
            raise InstallError(f"{path}: {k} is not an object")
    if entry is None:
        node.pop(keypath[-1], None)
        # drop containers that only existed for us
        for depth in range(len(keypath) - 1, 0, -1):
            parent = after
            for k in keypath[:depth - 1]:
                parent = parent[k]
            if parent.get(keypath[depth - 1]) == {}:
                parent.pop(keypath[depth - 1])
            else:
                break
    else:
        node[keypath[-1]] = entry
    bak = ctx.backup(path, label)
    eol = "\n" if (not text or text.endswith("\n")) else ""  # keep the file's trailing-newline style
    atomic_write(path, json.dumps(after, indent=json_indent(text) if text else 2, ensure_ascii=False) + eol)
    try:
        check = json.loads(path.read_text())
        if without(check, *keypath) != without(before, *keypath):
            raise InstallError("entries other than ours changed")
        got = check
        for k in keypath:
            got = got.get(k) if isinstance(got, dict) else None
        if got != entry:
            raise InstallError("our entry did not round-trip")
    except Exception as e:
        if bak:
            shutil.copy2(bak, path)
        else:
            path.unlink(missing_ok=True)
        raise InstallError(f"{path}: verification failed ({e}); restored the previous file")
    return f"{action.rstrip('e')}ed {'.'.join(keypath)} in {path} (backup: {bak or 'none, new file'})"


# ---------------------------------------------------------------- agents

def claude_entry(ctx: Ctx) -> dict:
    return {"type": "stdio", "command": ctx.node, "args": [ctx.server_js], "env": ctx.base_env}


def do_claude(ctx: Ctx, uninstall: bool) -> str:
    path = HOME / ".claude.json"
    before = json.loads(path.read_text()).get("mcpServers", {}) if path.exists() else {}
    existing = before.get(NAME)
    desired = None if uninstall else claude_entry(ctx)
    if desired is None and existing is None:
        return "not present"
    if desired is not None and existing == desired:
        return "unchanged"
    if ctx.dry_run:
        return f"would {'remove' if uninstall else 'add'} user-scope server via `claude mcp {'remove' if uninstall else 'add-json'} -s user {NAME}`"
    # ~/.claude.json is a live state file other sessions rewrite: back it up for reference,
    # but only ever change it through the CLI and compare the mcpServers map.
    bak = ctx.backup(path, "claude.json")
    if existing is not None:
        r = run(["claude", "mcp", "remove", "-s", "user", NAME])
        if r.returncode != 0:
            raise InstallError(f"claude mcp remove failed: {r.stderr.strip()[:300]}")
    if desired is not None:
        r = run(["claude", "mcp", "add-json", "-s", "user", NAME, json.dumps(desired)])
        if r.returncode != 0:
            raise InstallError(f"claude mcp add-json failed: {r.stderr.strip()[:300]}")
    after = json.loads(path.read_text()).get("mcpServers", {})
    others_ok = {k: v for k, v in after.items() if k != NAME} == {k: v for k, v in before.items() if k != NAME}
    if not others_ok or after.get(NAME) != desired:
        # undo through the CLI (never restore the whole live file)
        run(["claude", "mcp", "remove", "-s", "user", NAME])
        if existing is not None:
            run(["claude", "mcp", "add-json", "-s", "user", NAME, json.dumps(existing)])
        raise InstallError("~/.claude.json mcpServers verification failed; reverted our entry via the CLI "
                           f"(full-file backup at {bak})")
    return f"{'removed' if uninstall else 'registered'} user-scope server via claude CLI (backup: {bak})"


def toml_str(s: str) -> str:
    return json.dumps(s)  # JSON string escapes are valid TOML basic-string escapes


def codex_block(ctx: Ctx) -> str:
    env = ", ".join(f"{k} = {toml_str(v)}" for k, v in ctx.base_env.items())
    env_vars = ", ".join(toml_str(v) for v in KEY_VARS + OA_VARS)
    return "\n".join([
        CODEX_BEGIN,
        f"[mcp_servers.{NAME}]",
        f"command = {toml_str(ctx.node)}",
        f"args = [{toml_str(ctx.server_js)}]",
        f"env = {{ {env} }}",
        f"env_vars = [{env_vars}]  # forwarded by name; values stay in your environment",
        "startup_timeout_sec = 30",
        "tool_timeout_sec = 120",
        'default_tools_approval_mode = "approve"  # codex exec refuses MCP tool calls otherwise',
        CODEX_END,
    ]) + "\n"


def codex_strip(text: str) -> str:
    """Remove our managed block, or an unmanaged [mcp_servers.optimize-anything*] table."""
    if CODEX_BEGIN in text:
        pat = re.compile(re.escape(CODEX_BEGIN) + r".*?" + re.escape(CODEX_END) + r"\n?", re.S)
        return pat.sub("", text)
    lines, out, skipping = text.splitlines(keepends=True), [], False
    hdr = re.compile(r'^\s*\[\s*mcp_servers\s*\.\s*("optimize-anything"|optimize-anything)\s*(\.[^\]]*)?\]\s*(#.*)?$')
    for line in lines:
        if hdr.match(line):
            skipping = True
            continue
        if skipping and re.match(r"^\s*\[", line):
            skipping = False
        if not skipping:
            out.append(line)
    return "".join(out)


def do_codex(ctx: Ctx, uninstall: bool) -> str:
    path = HOME / ".codex" / "config.toml"
    text = path.read_text() if path.exists() else ""
    try:
        before = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise InstallError(f"{path} does not parse as TOML ({e}); not touching it")
    stripped = codex_strip(text)
    if uninstall:
        new = stripped
    else:
        base = stripped if (not stripped or stripped.endswith("\n")) else stripped + "\n"
        new = base + codex_block(ctx)  # no separator line, so uninstall restores the bytes exactly
    if new == text:
        return "unchanged" if not uninstall else "not present"
    if ctx.dry_run:
        return f"would {'remove' if uninstall else 'write'} [mcp_servers.{NAME}] in {path}"
    after = tomllib.loads(new)  # never write something that doesn't parse
    if without(after, "mcp_servers", NAME) != without(before, "mcp_servers", NAME):
        raise InstallError(f"{path}: rewrite would change other entries; not writing")
    if not uninstall:
        ours = after["mcp_servers"][NAME]
        if ours.get("command") != ctx.node or ours.get("default_tools_approval_mode") != "approve":
            raise InstallError(f"{path}: generated table did not parse as expected; not writing")
    bak = ctx.backup(path, "codex-config.toml")
    atomic_write(path, new)
    try:
        if tomllib.loads(path.read_text()) != after:
            raise InstallError("file differs after write")
        r = run(["codex", "mcp", "get", NAME, "--json"])
        if not uninstall and r.returncode != 0:
            raise InstallError(f"`codex mcp get {NAME}` failed: {(r.stderr or r.stdout).strip()[:200]}")
    except Exception as e:
        if bak:
            shutil.copy2(bak, path)
        raise InstallError(f"{path}: verification failed ({e}); restored the previous file")
    return f"{'removed' if uninstall else 'wrote'} [mcp_servers.{NAME}] in {path} (backup: {bak})"


def opencode_path() -> Path:
    d = HOME / ".config" / "opencode"
    for name in ("opencode.json", "opencode.jsonc"):
        p = d / name
        if p.exists():
            try:
                data = json.loads(p.read_text() or "{}")
            except json.JSONDecodeError:
                continue
            if data.get("mcp", {}).get("servers", {}).get(NAME) is not None:
                return p
    return d / "opencode.json"  # the main global file (created if missing)


def do_opencode(ctx: Ctx, uninstall: bool) -> str:
    path = opencode_path()
    if not uninstall:
        data, _ = json_load(path)
        mcp = data.get("mcp")
        if isinstance(mcp, dict) and any(k != "servers" for k in mcp):
            raise InstallError(f"{path}: `mcp` has v1-style keys {sorted(k for k in mcp if k != 'servers')}; "
                               "migrate them to mcp.servers (OpenCode v2) first")
    entry = None if uninstall else {"type": "local", "command": [ctx.node, ctx.server_js],
                                    "environment": ctx.base_env}
    res = edit_json(ctx, path, f"opencode-{path.name}", ("mcp", "servers", NAME), entry)
    return res


def do_gemini(ctx: Ctx, uninstall: bool) -> str:
    path = HOME / ".gemini" / "settings.json"
    env = dict(ctx.base_env)
    # Gemini redacts *KEY*/*TOKEN* vars unless re-listed. "$VAR" is expanded by Gemini at launch;
    # only names set now are listed, since an unset one would expand to "" and could shadow
    # subscription auth for the claude CLI backend.
    for k in KEY_VARS:
        if os.environ.get(k):
            env[k] = f"${k}"
    entry = None if uninstall else {"command": ctx.node, "args": [ctx.server_js], "env": env, "timeout": 120000}
    return edit_json(ctx, path, "gemini-settings.json", ("mcpServers", NAME), entry)


SKILL_MARKER = f"<!-- installed by {NAME}/scripts/install.sh; re-running it overwrites this file -->"


def stamped_skill(src: Path) -> str:
    """Repo SKILL.md with our marker after the frontmatter (frontmatter must stay first for Codex)."""
    text = src.read_text()
    m = re.match(r"(---\n.*?\n---\n)", text, re.S)
    if not m or not re.search(r"^name:\s*optimize-anything\s*$", m.group(1), re.M):
        raise InstallError(f"{src} lacks YAML frontmatter with name: optimize-anything (Codex rejects it)")
    return m.group(1) + SKILL_MARKER + "\n" + text[m.end():]


def do_skill(ctx: Ctx, dest: Path, uninstall: bool) -> str:
    src = ctx.repo / "skill" / "SKILL.md"
    want = stamped_skill(src)
    is_link_ours = dest.is_symlink() and Path(os.readlink(dest)) == src
    is_file_ours = dest.is_file() and not dest.is_symlink() and SKILL_MARKER in dest.read_text()[:4000]
    if uninstall:
        if not dest.exists() and not dest.is_symlink():
            return "not present"
        if is_link_ours or is_file_ours:
            if ctx.dry_run:
                return f"would remove {dest}"
            dest.unlink()
            try:
                dest.parent.rmdir()
            except OSError:
                pass
            return f"removed {dest}"
        return f"left {dest} in place (not written by this installer)"
    if ctx.link and is_link_ours:
        return "unchanged (symlink)"
    if not ctx.link and is_file_ours and dest.read_text() == want:
        return "unchanged"
    if ctx.dry_run:
        return f"would {'link' if ctx.link else 'write'} {dest}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    bak = None
    # Our own previous copy is overwritten silently; anything else is kept as a backup.
    if (dest.exists() or dest.is_symlink()) and not (is_link_ours or is_file_ours):
        bak = dest.with_name(f"{dest.name}.bak.{ctx.stamp}")
        os.replace(dest, bak)
    elif dest.is_symlink() or (ctx.link and dest.exists()):
        dest.unlink()
    if ctx.link:
        dest.symlink_to(src)
    else:
        atomic_write(dest, want)
        os.chmod(dest, 0o644)
    return f"{'linked' if ctx.link else 'wrote'} {dest}" + (f" (previous file kept: {bak})" if bak else "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--node", required=True)
    ap.add_argument("--uv", required=True)
    ap.add_argument("--agents", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--uninstall", action="store_true")
    ap.add_argument("--link", action="store_true")
    a = ap.parse_args()
    ctx = Ctx(Path(a.repo).resolve(), a.node, a.uv, a.dry_run, a.link)
    requested = [x for x in a.agents.split(",") if x]
    for x in requested:
        if x not in AGENTS:
            print(f"unknown agent {x!r}; choose from {','.join(AGENTS)}", file=sys.stderr)
            return 2
    agents = requested or [x for x in AGENTS if shutil.which(x)]
    if not agents:
        print("no supported agent found on PATH (claude, codex, opencode, gemini)", file=sys.stderr)
        return 1
    fns = {"claude": do_claude, "codex": do_codex, "opencode": do_opencode, "gemini": do_gemini}
    failures = 0
    for agent in agents:
        if not shutil.which(agent):
            print(f"[{agent}] skipped: not on PATH")
            continue
        try:
            print(f"[{agent}] mcp: {fns[agent](ctx, a.uninstall)}")
        except (InstallError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as e:
            failures += 1
            print(f"[{agent}] mcp: FAILED: {e}")
    skill_dests = set()
    if "claude" in agents:
        skill_dests.add(HOME / ".claude" / "skills" / NAME / "SKILL.md")
    if set(agents) & {"codex", "opencode", "gemini"}:
        skill_dests.add(HOME / ".agents" / "skills" / NAME / "SKILL.md")
    for dest in sorted(skill_dests):
        try:
            print(f"[skill] {do_skill(ctx, dest, a.uninstall)}")
        except (InstallError, OSError) as e:
            failures += 1
            print(f"[skill] FAILED: {e}")
    if ctx.backup_dir.exists():
        print(f"backups: {ctx.backup_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
