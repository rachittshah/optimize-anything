# Agent config formats: MCP stdio server + skills (verified 2026-11-11)

Versions checked locally: `claude` 2.1.322, `codex` 0.165.0 (`codex-cli 0.165.0`), `opencode` 2.0.29 (`opencode v2.0.29`). Gemini CLI is not installed, so its section comes from the docs only.

**Evidence types:** [DOC] = official docs (URL given). [HELP] = literal `--help` text, saved under `research/raw/`. [PROBE] = a live run using a small stdio MCP server (`scratchpad/probe/index.js`). The probe records the names of the env vars it receives and exposes one tool, `probe_ping`, which returns `PONG-7731`. Every probe was passed per run (`--mcp-config`, `-c`, or a scratch project dir). **No user config was modified and `mcp add` was never run.**

---

## TL;DR table

| | Claude Code | Codex CLI | OpenCode v2 | Gemini CLI (docs only) |
|---|---|---|---|---|
| User MCP file | `~/.claude.json` → `mcpServers` | `~/.codex/config.toml` → `[mcp_servers.<n>]` | `~/.config/opencode/opencode.json(c)` → `mcp.servers.<n>` | `~/.gemini/settings.json` → `mcpServers` |
| Add command | `claude mcp add -s user <n> -- node /abs/dist/index.js` | `codex mcp add <n> -- node /abs/dist/index.js` (always writes user config) | `opencode mcp add --global <n> -- node /abs/dist/index.js` | `gemini mcp add -s user <n> node /abs/dist/index.js` |
| Server env | **Full parent env inherited** [PROBE] | **Whitelist only** (9 default vars) + `env` + `env_vars` [PROBE] | Full parent env inherited [PROBE+DOC] | Inherited, but `*TOKEN*/*KEY*/*SECRET*…` names redacted unless listed in `env` [DOC] |
| Skills (SKILL.md) | `~/.claude/skills/<n>/SKILL.md` | `~/.agents/skills/<n>/SKILL.md` (documented); `~/.codex/skills` also loaded [PROBE: `codex debug prompt-input`] | `~/.config/opencode/skills/` (+ reads `~/.claude/skills`, `~/.agents/skills`) | `~/.gemini/skills/` or `~/.agents/skills/` |
| Instructions file | `~/.claude/CLAUDE.md` | `~/.codex/AGENTS.md` | `AGENTS.md` (+ `instructions` key) | `~/.gemini/GEMINI.md` |
| Headless mode | `claude -p` | `codex exec` | `opencode run` | `gemini -p` |
| Auto-approve the MCP tool | `--allowedTools mcp__<srv>__<tool>` | `-c mcp_servers.<n>.default_tools_approval_mode="approve"` (**required**, otherwise the call fails) | `--auto` | `"trust": true` in the server entry, or `--approval-mode=yolo` |
| Tokens/cost in output | `--output-format json` → `total_cost_usd`, `usage`, `modelUsage` [PROBE] | `--json` → `turn.completed.usage` (tokens only, no USD) [PROBE] | `--format json` → each `step_finish` has `tokens` + `cost` [PROBE] | `-o json` → `stats` [DOC] |
| Tool-list (doctor) | No CLI tool listing. `claude mcp get/list` health-checks only | No CLI tool listing (`codex debug prompt-input` renders skills but NOT MCP tools [PROBE]). `codex mcp list --json` / `get` show config only | No CLI tool listing. `opencode mcp list` shows status; `opencode api mcp.list` returned `data:[]` before first use [PROBE] | `/mcp` (interactive only); `gemini mcp list` shows status |

**Recommendation for `doctor`:** none of the four CLIs prints (verified for Claude/Codex/OpenCode; Gemini per docs) a server's tools without a model turn. The reliable, agent-neutral check is to spawn `node /abs/dist/index.js` yourself and send `initialize` then `tools/list` over stdio (newline-delimited JSON-RPC). This takes about 10 lines, and `probe/index.js` was smoke-tested this way. Then use each CLI's `mcp list`/`get` only to confirm the server is registered.

---

## 1. Claude Code (2.1.322)

### MCP: user scope
- Storage: "User-scoped servers are stored in `~/.claude.json`". Scopes are `local` (the default; stored in `~/.claude.json` under the project), `project` (`.mcp.json`), and `user`. [DOC https://code.claude.com/docs/en/mcp]
- [HELP `claude mcp add --help`]:
  ```
  -e, --env <env...>           Set environment variables (e.g. -e KEY=value)
  -s, --scope <scope>          Configuration scope (local, user, or project)
                               (default: "local")
  -t, --transport <transport>  Transport type (stdio, sse, http). Defaults to
                               stdio if not specified.
  ```
  Example from help: `claude mcp add my-server -e API_KEY=xxx -- npx my-mcp-server`
- Commands: `claude mcp add|add-json|get|list|remove [-s scope]|login|logout|reset-project-choices`. [HELP `claude mcp --help`] `list`/`get`: "approved servers are health-checked".
- Format, as observed in the user's `~/.claude.json` (secret values redacted) and matching the doc's `add-json` example `{"type":"stdio","command":...,"args":[...],"env":{...}}`:
  ```json
  { "mcpServers": {
      "my-server": { "type": "stdio", "command": "node",
                     "args": ["/abs/path/dist/index.js"], "env": {} } } }
  ```
  Recommended install method: `claude mcp add-json -s user my-server '{"type":"stdio","command":"node","args":["/abs/path/dist/index.js"]}'`. Avoid hand-editing `~/.claude.json`, because it is a large, constantly rewritten state file.
- Env: [PROBE] The server received **all 53 parent env vars**, including `PROBE_PARENT_VAR` and `PROBE_SECRET_TOKEN`, plus the explicit `env` entries. The cwd was the session cwd. The docs do not state this inheritance. They document `${VAR}` / `${VAR:-default}` expansion in `command`, `args`, `env`, `url`, and `headers`, and say Claude sets `CLAUDE_PROJECT_DIR` for servers. [DOC https://code.claude.com/docs/en/mcp]
- Timeouts: `MCP_TIMEOUT` sets startup (30 s default in `-p` with `--mcp-config` [DOC https://code.claude.com/docs/en/headless]). `MCP_TOOL_TIMEOUT` sets tool calls, and a per-server `"timeout"` (ms) overrides it. `MAX_MCP_OUTPUT_TOKENS` defaults to 25,000. [DOC https://code.claude.com/docs/en/mcp]

### Headless + isolation (`claude -p`)
- [HELP] `--bare`: "Minimal mode: skip hooks (those defined in settings and by installed plugins…), LSP, plugin sync, attribution, auto-memory, background prefetches, keychain reads, and CLAUDE.md auto-discovery. Sets CLAUDE_CODE_SIMPLE=1. Anthropic auth is strictly ANTHROPIC_API_KEY or apiKeyHelper via --settings (OAuth and keychain are never read)… Explicitly provide context via: --system-prompt[-file], --append-system-prompt[-file], --add-dir (CLAUDE.md dirs), --mcp-config, --settings, --agents, --plugin-dir."
- [DOC https://code.claude.com/docs/en/headless] `--bare` skips "auto-discovery of hooks, skills, custom commands, subagents, installed plugins, MCP servers, auto memory, and CLAUDE.md". Under bare, "only servers supplied on the command line connect". "`--bare` is the recommended mode for scripted and SDK calls, and will become the default for `-p` in a future release."
- **Gotcha:** `--bare` needs `ANTHROPIC_API_KEY`, because subscription OAuth is never read. To keep the subscription login but still isolate the run, use the flags below without `--bare`. My probe ran this way with OAuth.
  - `--strict-mcp-config` [HELP]: "Only use MCP servers from --mcp-config, ignoring all other MCP configurations"
  - `--mcp-config <configs...>` [HELP]: "Load MCP servers from JSON files or strings"
  - `--setting-sources <sources>` [HELP]: "Comma-separated list of setting sources to load (user, project, local)." Pass a narrow list to drop user hooks/settings. Empty `""` is accepted (see the re-test below).
  - `--safe-mode` [HELP]: disables "CLAUDE.md, skills, installed plugins, hooks, MCP servers, custom commands and agents…". Re-test: `--mcp-config` servers do NOT load under `--safe-mode`.
  - **Isolation re-test with OAuth [PROBE, haiku, same probe prompt]:**
    | flags | probe tool called? | prompt size signal |
    |---|---|---|
    | `--strict-mcp-config --mcp-config …` only | yes | ~55k tokens/turn (cache create 55,135 on turn 1) |
    | `--safe-mode --strict-mcp-config --mcp-config …` | **no**: "no tool by that name". `--safe-mode` drops `--mcp-config` servers too | 35,909 (smaller) |
    | `--setting-sources project …` | yes | ~53k/turn (cache_read 102,280 over 2 turns) |
    | `--setting-sources "" …` (empty is accepted) | yes | ~53k/turn (cache_read 106,334) |
    **Conclusion:** with OAuth, `--setting-sources "" --strict-mcp-config --mcp-config <json>` is the best recipe that still lets the MCP server load. It drops user/project/local settings files, which is where hooks live, and all other MCP servers. **It does not shrink the prompt**, so CLAUDE.md, auto-memory, and the skill listing appear to still load (UNCONFIRMED which ones exactly). The only full context strip is `--bare`, which requires `ANTHROPIC_API_KEY` or an `apiKeyHelper` in `--settings`. A `--system-prompt` replacement was not tested.
  - `--disable-slash-commands` [HELP]: "Disable all skills and commands"
  - `--no-session-persistence` [HELP]: "sessions will not be saved to disk… (only works with --print)"
  - `--max-budget-usd <amount>` [HELP]: only works with `--print`
  - No `--no-hooks` flag exists. Hooks are skipped only through `--bare`, `--safe-mode`, or `--setting-sources`.
- Auto-approve: `--allowedTools "mcp__probe__probe_ping"` worked [PROBE]. The rule syntax is `mcp__<server>__<tool>`. For plugin-bundled servers it is `mcp__plugin_<plugin>_<server>__<tool>` [DOC https://code.claude.com/docs/en/mcp]. `-p` runs start in Manual mode (v2.1.302+), so an unlisted tool that needs approval is denied [DOC https://code.claude.com/docs/en/headless].
- JSON output [PROBE, haiku, one tool call]: top-level keys include `result`, `total_cost_usd` (0.01164757), `usage` (`input_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`, `output_tokens`, `output_tokens_details.thinking_tokens`, `iterations`), `modelUsage.<model>.{inputTokens,outputTokens,cacheReadInputTokens,cacheCreationInputTokens,costUSD,…}`, `num_turns`, `session_id`, `permission_denials`, `is_error`, `duration_ms`, `duration_api_ms`. [DOC https://code.claude.com/docs/en/headless]: the cost figures are "client-side estimates". With `--continue`/`--resume`, the reported cost covers the whole conversation.
- Doctor signal: with `--output-format stream-json`, the `system/init` event carries `mcp_servers[{name,status}]` and `mcp_server_errors` [DOC https://code.claude.com/docs/en/headless].

### Skills
- `~/.claude/skills/<skill-name>/SKILL.md` (personal). "Claude Code skills follow the Agent Skills open standard". All frontmatter is optional, and `description` is recommended. [DOC https://code.claude.com/docs/en/skills]
- Under `--bare`, skills are not auto-discovered or listed to the model (except `--add-dir`'s `.claude/skills/`) [DOC https://code.claude.com/docs/en/headless], but [HELP] says "Skills still resolve via /skill-name", so explicit `/name` invocation still works.

---

## 2. Codex CLI (0.165.0)

### MCP: user scope
- File: `~/.codex/config.toml`. Project scope is `.codex/config.toml` (trusted projects only). [DOC https://learn.chatgpt.com/docs/extend/mcp?surface=cli]
  ```toml
  [mcp_servers.my-server]
  command = "node"
  args = ["/abs/path/dist/index.js"]
  env = { SOME_FIXED = "value" }          # literal values
  env_vars = ["OPENAI_API_KEY", "MY_VAR"]  # forward these from parent env
  # cwd = "/abs/path"
  startup_timeout_sec = 30                 # default 10
  tool_timeout_sec = 300                   # default 60
  default_tools_approval_mode = "approve"  # needed for codex exec, see below
  # enabled = true; enabled_tools = [...]; disabled_tools = [...]; required = true
  ```
  Doc lines: "`env_vars` … Environment variables to allow and forward". `startup_timeout_sec` "Default: `10`". `tool_timeout_sec` "Default: `60`". `default_tools_approval_mode` values are "`auto`, `prompt`, `writes`, and `approve`". The per-tool override is `tools.<tool>.approval_mode`. [DOC mcp page; https://learn.chatgpt.com/docs/config-file/config-reference.md: env_vars = "Additional environment variables to whitelist for an MCP stdio server"]
- [HELP `codex mcp add --help`]: `Usage: codex mcp add [OPTIONS] <NAME> (--url <URL> | -- <COMMAND>...)` and `--env <KEY=VALUE>  Environment variables to set when launching the server. Only valid with stdio servers`. `add` has no scope flag; it writes `~/.codex/config.toml`.
- Commands [HELP `codex mcp --help`]: `list`, `get`, `add`, `remove`, `login`, `logout`. `codex mcp list --json` means "Output the configured servers as JSON". This shows config only and does not list tools.
- **Env [PROBE]:** With parent `PROBE_PARENT_VAR` set, the server received exactly the 9 default vars `HOME, LANG, LOGNAME, PATH, SHELL, TERM, TMPDIR, USER, __CF_USER_TEXT_ENCODING` plus the `env` entries. The parent var was **not** forwarded. After adding `env_vars=["PROBE_PARENT_VAR"]`, it was forwarded. **Any API key the server needs must be listed in `env_vars` (or `env`).** The docs only imply this ("whitelist").
- Existing user config already uses `env_vars` (for example `october-bus-static`) and `mcp_optional_startup_grace_ms` (a top-level key, not on the doc page I read; UNCONFIRMED semantics).

### Headless (`codex exec`)
- **Approval re-test, one variable at a time [PROBE]:** (A) `--ignore-user-config` with no approval key → **failed**, "MCP tool call requires approval". (B) user config (`approval_policy = "on-request"`) plus only `default_tools_approval_mode="approve"` → **completed**. So the server-level approve key is what matters, whatever the user config says. `codex mcp add` has **no flag** for it, so an installer has to write the TOML key itself (or pass `-c` per run).
- **Gotcha [PROBE]:** The default run (user config) failed the MCP call with `"MCP tool call requires approval, but approval policy is never"`. It succeeded after adding `-c 'mcp_servers.<n>.default_tools_approval_mode="approve"'`, which returned `PONG-7731`. Docs do not cover exec + MCP approvals. The `--dangerously-bypass-approvals-and-sandbox` route was not tested.
- [HELP `codex exec --help`] relevant lines:
  ```
  -m, --model <MODEL>             Model the agent should use
  -s, --sandbox <SANDBOX_MODE>    [possible values: read-only, workspace-write, danger-full-access]
      --skip-git-repo-check       Allow running Codex outside a Git repository
      --ephemeral                 Run without persisting session files to disk
      --ignore-user-config        Do not load `$CODEX_HOME/config.toml`; auth still uses `CODEX_HOME`
      --ignore-rules              Do not load user or project execpolicy `.rules` files
      --json                      Print events to stdout as JSONL
  -o, --output-last-message <FILE>
      --output-schema <FILE>
  -c, --config <key=value>        (TOML-parsed override, dotted path)
  -p, --profile <CONFIG_PROFILE_V2>   Layer $CODEX_HOME/<name>.config.toml on top of the base user config
      --dangerously-bypass-approvals-and-sandbox
      --dangerously-bypass-hook-trust
  ```
  Note: `-p` in Codex means `--profile`, not "print".
- Isolated run that works [PROBE]: `codex exec --skip-git-repo-check --ephemeral --json -s read-only --ignore-user-config -c 'mcp_servers.x.command="node"' -c 'mcp_servers.x.args=["/abs/dist/index.js"]' -c 'mcp_servers.x.env_vars=["MY_VAR"]' -c 'mcp_servers.x.default_tools_approval_mode="approve"' "<prompt>" </dev/null`
  - `--ignore-user-config` still loaded `~/.codex/hooks.json`; an `item.type:"error"` event read "clamping Interrupt hook timeout…". `~/.agents/skills` were also still scanned. Use `--ignore-rules` too, and set `CODEX_HOME` to a temp dir for full isolation (auth then has to be copied in). UNCONFIRMED: the cleanest hook-disable flag.
  - Without `</dev/null`, exec prints "Reading additional input from stdin..." and can hang (stdin is appended as a `<stdin>` block).
- JSON usage [PROBE]: `{"type":"turn.completed","usage":{"input_tokens":64999,"cached_input_tokens":50048,"cache_write_input_tokens":0,"output_tokens":113,"reasoning_output_tokens":0}}`. **No USD cost field.** MCP calls appear as `item.started/completed` with `"type":"mcp_tool_call","server","tool","arguments","result","error","status"`. Event types: "`thread.started`, `turn.started`, `turn.completed`, `turn.failed`, `item.*`, and `error`" [DOC https://learn.chatgpt.com/docs/non-interactive-mode.md]. "`--full-auto` is no longer supported" [same].

### Skills
- [DOC https://learn.chatgpt.com/docs/build-skills]: skills "build on the open agent skills standard". "The `SKILL.md` file must include `name` and `description`." Scopes: REPO `$CWD/.agents/skills`, `$CWD/../.agents/skills`, `$REPO_ROOT/.agents/skills`; USER `$HOME/.agents/skills`; ADMIN `/etc/codex/skills`; SYSTEM bundled. Duplicate names are not merged.
- `~/.codex/skills` (`$CODEX_HOME/skills`) is **not on the doc page**, but the 0.165.0 binary still has a `codex-home-skills` scope, and its bundled installer text says "Installs into `$CODEX_HOME/skills/<skill-name>` (defaults to `~/.codex/skills`)" (from `strings` on the binary). **Prefer `~/.agents/skills/<name>/SKILL.md`.**
- [PROBE] Codex rejects skills with no YAML frontmatter: `failed to load skill /Users/…/.agents/skills/blog-outline/SKILL.md: missing YAML frontmatter delimited by ---`. Frontmatter is mandatory.
- Enable/disable: `[[skills.config]]` entries with `path` or `name` plus `enabled`. `skills.bundled.enabled` defaults to true. [DOC https://learn.chatgpt.com/docs/config-file/config-reference.md]
- Instructions: `~/.codex/AGENTS.md` (or `AGENTS.override.md`). Project docs are merged root→cwd, with `project_doc_max_bytes` defaulting to 32 KiB. [DOC https://learn.chatgpt.com/docs/agent-configuration/agents-md.md]

---

## 3. OpenCode (2.0.29, v2 architecture)

### MCP: user scope
- Global config: `~/.config/opencode/opencode.json(c)`. Precedence goes global < project files < `/etc/opencode/` < managed. [DOC https://opencode.ai/docs/config/] `opencode debug paths` → `config ~/.config/opencode`. [local]
- **v2 key shape is `mcp.servers.<name>`**, not the v1 `mcp.<name>`. [DOC https://opencode.ai/docs/mcp-servers/]
  ```jsonc
  {
    "$schema": "https://opencode.ai/config.json",
    "mcp": { "servers": {
      "my-server": {
        "type": "local",                                 // required
        "command": ["node", "/abs/path/dist/index.js"],  // array: exe + args
        "environment": { "SOME_FIXED": "value" }         // optional, added to inherited env
        // "disabled": true, "timeout": { startup/catalog/execution }
      } } }
  }
  ```
  Doc lines: "`environment` … String variables added to OpenCode's inherited process environment". "`disabled` … Defaults to `false`" (the doc says to use `disabled`, not `enabled`). Timeout defaults: "30 seconds for `startup`, 30 seconds for `catalog`, and 12 hours for `execution`".
- [PROBE] A scratch project `opencode.json` using the v2 shape above loaded and worked in `opencode run --standalone`. The server got **all 53 parent env vars** plus `environment`.
- [HELP `opencode mcp add --help`]:
  ```
  opencode mcp add [flags] <name> [<command...>]
  --env key=value       Environment variable for a local server, as name=value
  --global              Write to the global config instead of the project config
  ```
  **Default writes to the project config.** Use `--global` for user scope. Subcommands: `list`, `add`, `auth`, `logout`. There is no `remove`, so delete the key by hand.
- `opencode debug config` lists every config source and its parsed content, including the project file. This is a good doctor check that the entry is registered. [local] It showed that v2 auto-migrated the user's v1 `provider`/`model` keys to `providers`/`model{providerID,model}` at read time.

### Headless (`opencode run`)
- [HELP `opencode run --help`]:
  ```
  --standalone            Run with a private server instead of the background service
  --model, -m string      Model to use in the format provider/model#variant
  --agent string          Agent to use
  --format choice         Output format (choices: default, json)
  --auto                  Auto-approve permissions that are not explicitly denied
  ```
- [PROBE] `opencode run --standalone --format json --auto -m opencode/deepseek-v4.1-flash-free "<prompt>"` → `PONG-7731`. JSONL events: `step_start`, `tool_use`, `step_finish`, `text`. Each `step_finish.part` has `"cost":0,"tokens":{"input","output","reasoning","cache":{"read","write"}}`.
- **Gotcha:** v2 exposes MCP tools through a code-mode `execute` tool. The model wrote `await tools.probe.probe_ping()`, and `tool_use.part.tool` is `"execute"`, not the MCP tool name. A doctor/eval that greps for the tool name has to look inside `state.input.code` or the output.
- **Gotcha:** OpenCode v2 runs a shared background service (`opencode service status` → `http://127.0.0.1:<port>`). `opencode mcp list` (no `--standalone` flag) answered "No MCP servers configured" from the scratch project dir, even though `debug config` saw the project file. Use `--standalone` for runs, or `opencode reload` after editing global config (not tested, to avoid touching the user's service).

### Skills
- Paths [DOC https://opencode.ai/docs/skills/]: Global `~/.config/opencode/skills`. Global compatibility `~/.claude/skills`, `~/.agents/skills`. Project `.opencode/skills`, `.claude/skills`, `.agents/skills`, searched from cwd up to the project root.
- Frontmatter: everything is optional, but "Skills without a `description` are not advertised to the model". Permissions use `{ "action": "skill", "resource": "<id|glob>", "effect": "allow|ask|deny" }` under `permissions`.
- Local note: the user has a legacy `~/.config/opencode/skill/` (singular) dir. The v2 docs list only `skills/` (plural).
- Instructions: an `instructions` array (files/globs/URLs) is loaded "alongside discovered `AGENTS.md` files" [DOC https://opencode.ai/docs/config/]. UNCONFIRMED: the global AGENTS.md path for v2 (likely `~/.config/opencode/AGENTS.md`).

---

## 4. Gemini CLI (not installed; docs only)

- **Status gotcha:** every geminicli.com page carries the banner "Unpaid tier and Google One users: Gemini CLI was replaced by Antigravity CLI on June 18th, 2026" and "Gemini CLI has not been optimized for Gemini 4 Argon". Treat Gemini CLI support as legacy. Antigravity CLI formats were not researched.
- MCP [DOC https://geminicli.com/docs/tools/mcp-server/]: `~/.gemini/settings.json` (user), `.gemini/settings.json` (project).
  ```json
  { "mcpServers": { "my-server": {
      "command": "node", "args": ["/abs/path/dist/index.js"],
      "env": { "MY_API_KEY": "$MY_API_KEY" },
      "timeout": 600000, "trust": true } } }
  ```
  - `timeout`: "Request timeout in milliseconds (default: 600,000ms = 10 minutes)". `trust`: "When `true`, bypasses all tool call confirmations for this server". `includeTools`/`excludeTools` (exclude wins).
  - Env: inherited, but "redacts sensitive environment variables from the base environment", covering `*TOKEN*`, `*SECRET*`, `*PASSWORD*`, `*KEY*`, `*AUTH*`, `*CREDENTIAL*`, and `GEMINI_API_KEY`. "Explicitly defined variables … are trusted". **API keys must be re-listed in `env`**, e.g. `"OPENAI_API_KEY": "$OPENAI_API_KEY"`.
  - CLI: `gemini mcp add [options] <name> <commandOrUrl> [args...]`, `gemini mcp remove <name>`, `gemini mcp list`. **`-s/--scope` defaults to `project`**, so pass `-s user`.
  - Tool listing: `/mcp` in an interactive session only. "Stdio servers show as Disconnected in untrusted folders."
- Headless [DOC https://geminicli.com/docs/cli/cli-reference/, https://geminicli.com/docs/cli/headless/]: `-p/--prompt` "Forces non-interactive mode". `-o/--output-format text|json|stream-json`; json gives `response`, `stats`, `error`, and per-model token usage appears in the stream-json `result` event. `--approval-mode default|auto_edit|yolo|plan` (`--yolo` is deprecated). `--allowed-tools` is deprecated in favour of the Policy Engine. `--allowed-mcp-server-names`. `--skip-trust` / env `GEMINI_CLI_TRUST_WORKSPACE=true`: folder trust is on by default (`security.folderTrust.enabled` = true), and untrusted folders don't start stdio servers.
- Skills [DOC https://geminicli.com/docs/cli/skills/, https://geminicli.com/docs/reference/configuration/]: Agent Skills are supported, in "`~/.gemini/skills/` or the `~/.agents/skills/` alias". `skills.enabled` defaults to `true`. Commands: `gemini skills list|install [--consent]|uninstall`.
- Instructions [DOC https://geminicli.com/docs/cli/gemini-md/]: global `~/.gemini/GEMINI.md`, hierarchical. To read AGENTS.md, set `"context": {"fileName": ["AGENTS.md","GEMINI.md"]}`.

---

## Cross-cutting gotchas (ranked by how likely they break an install)
1. **Codex exec blocks MCP tool calls** unless the server sets `default_tools_approval_mode = "approve"` (or the run bypasses approvals). [PROBE]
2. **Codex strips the env** to 9 default vars (`HOME LANG LOGNAME PATH SHELL TERM TMPDIR USER __CF_USER_TEXT_ENCODING`). List the server's keys in `env_vars`. **Gemini redacts `*KEY*`/`*TOKEN*`** vars, so re-list them in `env`. Claude and OpenCode pass the full env through. [PROBE ×3, DOC Gemini]
3. **OpenCode v2 needs `mcp.servers.<n>` + `"type":"local"` + `command` as an array.** `opencode mcp add` writes to the project config unless you pass `--global`.
4. **Default scopes differ:** Claude `local`, Gemini `project`, OpenCode project. Pass `-s user` / `-s user` / `--global`. Codex always writes user config.
5. **Codex startup timeout is 10 s** by default. A Node server with a slow cold start should set `startup_timeout_sec`. Claude defaults to 30 s (`MCP_TIMEOUT`), OpenCode to 30 s startup, Gemini to a 10-minute request timeout.
6. **Skills dirs:** `~/.agents/skills` is read by Codex, OpenCode, and Gemini. Claude needs `~/.claude/skills` (which OpenCode also reads). Installing to both covers all four. Codex hard-fails a SKILL.md that has no `---` frontmatter, and requires `name` and `description`.
7. **`claude --bare` drops OAuth.** It needs `ANTHROPIC_API_KEY`. For subscription users, use `--setting-sources "" --strict-mcp-config --mcp-config …` (no hooks/settings, only your server; context is not reduced). `--safe-mode` is not usable because it also drops `--mcp-config`.
8. **Codex has no USD cost** in `--json`, only tokens. Claude has `total_cost_usd` (an estimate). OpenCode has a per-step `cost` (0 for free models).
9. Codex `exec` needs `</dev/null` in scripts. `-p` means profile in Codex but print in Claude, and prompt in Gemini.

Raw `--help` captures: `<scratch>/research/raw/`. Probe server and outputs: `<scratch>/probe/` (env files hold var NAMES only).
