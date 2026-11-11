# Expert: Agent-Tooling Engineer — optimize-anything upgrade
Metric: ease of setup and correct use across Claude Code, Codex CLI, OpenCode, Gemini CLI. Verified 2026-11-11.

## 1. Source digest (URL → facts)

**Skill packaging**
- agentskills.io/specification: required `name` (≤64 chars, lowercase/digits/hyphens, no `--`, **must match dir name**) and `description` (≤1024 chars). Optional fields: `license`, `compatibility` (≤500), `metadata`, `allowed-tools` (experimental). Disclosure levels: metadata ~100 tok, body <5000 tok, keep SKILL.md <500 lines, refs one level deep.
- code.claude.com/docs/en/skills: follows the standard plus extensions (`when_to_use`, `context: fork`, `paths`, `hooks`…). Locations: `~/.claude/skills`, `.claude/skills`, plugin `skills/`. **`.agents/skills` is not listed.** Listing cap is 1,536 chars per skill, total budget 1% of context; descriptions get dropped when over budget. "Outside Claude Code (claude.ai, Skills API) only name/description/license/compatibility/metadata/allowed-tools are allowed; **any other key is a hard error**." Trigger testing: `claude plugin eval` with a `tool_used: Skill` grader. Use `${CLAUDE_SKILL_DIR}` for script paths.
- learn.chatgpt.com/docs/build-skills (Codex): scans `.agents/skills` (repo → root), `~/.agents/skills`, `/etc/codex/skills`. `$skill` invokes explicitly, otherwise implicit match on description. Optional `agents/openai.yaml` holds `allow_implicit_invocation` and `dependencies.tools` (can declare an MCP server). Skill list ≤2% of context (8k chars fallback); descriptions shortened first.
- opencode.ai/docs/skills: reads `.opencode/skills`, `.claude/skills`, `.agents/skills` (+ global equivalents). Model loads a skill via the `skill` tool (`{"id": ...}`). ID comes from the path, not `name`. Permission rules allow/ask/deny.
- geminicli.com/docs/cli/skills: Built-in < Extension < User (`~/.gemini/skills` or `~/.agents/skills`) < Workspace (`.gemini/skills` or `.agents/skills`). Model calls `activate_skill`, which **asks the user for consent before loading**. `gemini skills install <git-url>`. Page notes Gemini CLI was replaced by Antigravity CLI for free and Google One users on 2026-06-18.
- geminicli.com/docs/extensions/reference: `gemini-extension.json` takes `mcpServers` (use `${extensionPath}`), `contextFileName`, `excludeTools`, and `settings[]{name,envVar,sensitive}`. **Extensions only receive the env vars declared in settings.** Skills go in `skills/<name>/SKILL.md`.
- agents.md: AGENTS.md is supported by Codex, OpenCode, Gemini CLI (via contextFileName), Cursor, Copilot and others. code.claude.com/docs/en/memory: Claude Code reads AGENTS.md natively.
- **Conclusion:** one spec-only SKILL.md is portable. Discovery paths differ: `.agents/skills` covers Codex, OpenCode and Gemini; Claude Code needs `.claude/skills` or a plugin.

**MCP config and limits per client**
- code.claude.com/docs/en/mcp: `MAX_MCP_OUTPUT_TOKENS` defaults to 25k; warning at 10k; text over 50k chars is saved to a file. Tool timeout ~28h, idle timeout 30 min for stdio (progress notifications reset it), auto-background after 2 min. Supports `${VAR:-default}` and `${CLAUDE_PLUGIN_ROOT}` in plugin `.mcp.json`. Plugin tool names look like `mcp__plugin_<p>_<s>__<tool>`. **MCP tasks are public beta**: protocol 2026-07-28, main conversation only (not subagents), not on Bedrock.
- learn.chatgpt.com/docs/extend/mcp (Codex): `[mcp_servers.x]` takes command/args/env/**`env_vars` (allowlist to forward)**/cwd. **`tool_timeout_sec` defaults to 60s**, startup to 10s. Per-tool `output_token_limit`. Tasks/progress support not documented (?).
- geminicli.com/docs/tools/mcp-server: default timeout 600s; `$VAR` expansion (undefined becomes empty); `trust`, `includeTools`.
- opencode.ai/docs/mcp-servers: `type:"local"`, `command:[...]`, `environment`, `{env:NAME}` syntax (shell `$NAME` is NOT expanded). Execution timeout 12h. `protocol` can be `legacy|auto|2026-07-28`.
- modelcontextprotocol.io/specification/2026-07-28/basic/utilities/tasks: Tasks are now an **extension** (`io.modelcontextprotocol/tasks`). Server returns `resultType:"task"` and the client polls `tasks/get` with `pollIntervalMs`; `tasks/update` and `tasks/cancel` exist. The 2025-11-25 version (experimental) used `execution.taskSupport`. modelcontextprotocol.io/extensions/client-matrix: **no client is marked as supporting Tasks** (community-maintained list; Claude Code's own docs say beta).
- modelcontextprotocol.io/specification/2025-11-25/server/tools: return input-validation failures as **tool execution errors (`isError:true`) so the model can self-correct**. `structuredContent` SHOULD also carry a serialized TextContent copy. `resource_link` content is allowed. Use `{"type":"object","additionalProperties":false}` for tools with no params.
- anthropic.com/engineering/writing-tools-for-agents: consolidate around workflows; avoid overlapping tools; add a `response_format` concise/detailed param (~⅓ tokens); truncate with guidance on how to narrow; errors should name the fix; evaluate tools on held-out tasks.

**Optimizer formulation**
- gepa-ai.github.io/gepa/api/optimize_anything/optimize_anything: `optimize_anything(seed_candidate=None, *, evaluator, batch_evaluator, dataset, valset, objective, background, test_set, config)`. Config: `max_evals=100`, `max_token_cost`, `max_concurrency=8`, `stop_at_score`, `sandbox=True`. **Warns that the run is unbounded if neither budget is set.** Evaluator returns `(score, info)` with higher = better. Seedless mode needs `objective`. `test_set` is scored outside the budget.
- gepa blog 2026/02/18 (introducing optimize_anything): ASI = "anything that would help an expert diagnose failures" (errors, outputs, runtime, per-aspect `scores`). "A single average **hides which aspects are strong and which are weak**." Example budgets were 100–350 metric calls. Multi-task mode converges faster than single-task.
- gepa blog 2026/07/22 (omni): engines `gepa | autoresearch | meta_harness` (coding-agent proposer) behind one task/evaluator. Budgets `max_token_cost` and `stop_at_score` are honored by all engines.
- gepa blog 2026/09/29 (optimize cheap, deploy strong): cheap task LM + strong reflector + ship to a stronger model. HotpotQA scored +9.0 pts at ~14× lower cost. Prompt-insensitive tasks show no gain.
- dspy.ai/current/api/optimizers/GEPA/overview: metric should return `dspy.Prediction(score, feedback)`; without feedback GEPA falls back to a score-only message. Scalar score still gates acceptance even with objective frontiers.
- github.com/algorithmicsuperintelligence/openevolve: `EVOLVE-BLOCK-START/END` scopes what gets mutated; evaluator artifacts (errors) feed the next prompt.

**Skill triggering**
- agentskills.io/skill-creation/optimizing-descriptions: imperative "Use this skill when…", intent over mechanics, "pushy". Eval set: ~20 queries (8–10 positive, 8–10 **near-miss** negatives), 3 runs each, trigger-rate threshold 0.5, **60/40 train/val split**, pick the best iteration by val, ~5 iterations. Agents skip skills for tasks they can do alone.
- platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices: write the description in third person, what + when. Name the MCP server with the tool (`Server:tool`). Avoid time-sensitive info, avoid offering many options, build ≥3 evals first.
- Community (LOW evidence): activation rates of 41–100% depending on wording and harness (https://dev.to/lizechengnet/why-claude-code-skills-dont-trigger-and-how-to-fix-them-in-2026-o7h, https://besthub.dev/articles/why-claude-code-skills-fail-to-activate-and-how-to-achieve-100-reliability-291371b95c53, https://mcp.directory/blog/why-your-claude-skill-isnt-activating-2026-fixes). Reported bug: `claude -p` showed 0% recall in skill-creator loops (claudeissues.com/issue/36570).
- platform.claude.com/docs/en/about-claude/model-deprecations: `claude-opus-4-6` and `claude-sonnet-4-6` (engine defaults) are **active** until at least Feb 2027. **`temperature` was removed in API version 2026-10-21** (returns 400); `Config.temperature=1.0` breaks if the SDK pins that version (?).

**Repo facts that block correct use** (read-only audit)
- `mcp-server/src/process-manager.ts`: status is only updated on `optimization_complete`, so `check_optimization` shows `running` with no progress for the whole run. The disk fallback in `getStatus` reports `completed` for any run with a checkpoint, including crashed or in-progress runs. `split('\n')` per chunk plus an empty `catch{}` silently drops JSON lines split across chunks. `ENGINE_DIR` is `__dirname/../../engine` (only works from a git clone). Env and PATH are inherited implicitly. Event and run-dir paths are fixed under `~/.optimize-anything`.
- `tools.ts`: evaluator schema is flat (shell without `command` is accepted); `config` is untyped; there is no budget field; no outputSchema.
- `engine/.../config.py`: `max_iterations=100`, `max_metric_calls=None`, Opus proposer, no cost cap. This is unbounded-ish and expensive by default.
- `evaluators/shell_eval.py:29`: `{{candidate}}` is raw-substituted into a shell command (quoting breakage and injection). `{{candidate_file}}` is the safe path, but Recipe 2 and the README teach `echo '{{candidate}}'`.
- `skill/SKILL.md`: the description triggers on bare "optimize/improve", which over-triggers on near misses like "optimize this SQL query". Install instructions are Claude-only. No budget or valset guidance.

## 2. Candidate designs (axes 1–6 + added: interface surface, packaging, setup path)

| # | Design | 1 search | 2 selection | 3 feedback | 4 model routing | 5 budget | 6 artifact scope | Interface | Packaging | Setup path |
|---|---|---|---|---|---|---|---|---|---|---|
| A | Status quo | reflective mutation | Pareto (instance) | score+ASI | Opus proposer, Sonnet judge | max_iterations=100 | dict components | 5 async tools, opaque status | `skill/` Claude-only | clone + npm build + abs path |
| B | Portable thin | = A | = A | = A | = A | **required** `budget{max_metric_calls|max_cost_usd, stop_at_score}` | = A | 5 tools, truthful status, typed schema | `.agents/skills/` + Claude plugin + Gemini ext + Codex/OpenCode snippets | `uvx`/`npx` one-liner + `doctor` |
| C | B + formulation guardrails | = A | Pareto; **final pick by valset** | score + **per-example/per-aspect scores** + ASI | cheap task/judge, strong proposer (preset) | required + dry-run cost estimate | dict + optional EVOLVE-BLOCK-style region | B + `dry_run` (seed eval, warnings) + long-poll status + file/resource_link output | as B + `references/` + `assets/` templates | as B |
| D | CLI-first skill, MCP optional | = A | = A | = A | = A | required | = A | shell: `optimize-anything run --config … --events &`, then `status` | spec-only skill + bundled config templates | `uv tool install` only |
| E | MCP Tasks-native | = A | = A | = A | = A | required | = A | 1 tool, `resultType:"task"` | MCP only | needs client on 2026-07-28 |
| F | Host-agent-as-proposer | agent writes candidates in its own loop | Pareto in engine | ASI returned to agent | **host agent's model, no API key** | per-session eval cap | any file | `evaluate_candidate` / `next_minibatch` tools | skill drives the loop | no key needed |

## 3. Scores (1–5, higher = better; effort: 5 = least work)

| # | valset score | sample eff. | cost | train/val gap | impl effort | evidence | My metric (setup + correct use) |
|---|---|---|---|---|---|---|---|
| A | baseline | baseline | 2 (unbounded Opus) | ? (no val guidance; skill omits valset) | 5 | – | 1 |
| B | ? same engine | ? same engine | 4 (budget required) | ? | 4 | strong (vendor docs) | 4 |
| C | ? likely ↑ via fewer mis-formulated runs, not via search | ? | 4 (dry-run catches broken evaluators before spend) | 4 (val-based final pick, warn on missing valset; GEPA/agentskills practice) | 3 | medium-strong | **5** |
| D | ? | ? | 4 | ? | 4 | strong (all 4 agents have shell) | 4 (no MCP config at all; worse status UX) |
| E | ? | ? | 4 | ? | 3 | weak for portability (matrix: no client listed; Claude beta only, no subagents) | 2 |
| F | ? (agent proposer quality varies; GEPA omni meta_harness is precedent) | ? likely worse/eval-hungry | 3 (moves cost into agent context) | ? | 2 | weak | 3 (no key, but context burn and long loops) |

The "?" cells in the valset, sample-efficiency and gap columns are mostly because these designs change the interface, not the search algorithm. The optimizer-side experts own those numbers. The only interface effects I can defend are fewer wasted runs (C's dry run) and val-based selection (C).

## 4. Ranked top 3 upgrades (by my metric)

### #1 Portable install with zero hardcoding, plus a CLI fallback (B + D)
- **Step 0: publish.** The `npx …-mcp` and `uvx --from optimize-anything` snippets below assume npm and PyPI releases that **don't exist yet**; the README only documents clone + build + absolute path. Until they're published, every snippet uses the form that works today: `node <ABS>/mcp-server/dist/index.js` with env `OPTIMIZE_ANYTHING_ENGINE=<ABS>/engine`.
- **Skill location**: canonical `skills/optimize-anything/SKILL.md` (name = dir). Ship it to `.agents/skills/optimize-anything/` (Codex, OpenCode, Gemini) and `.claude/skills/optimize-anything/` (symlink or plugin). Frontmatter uses **spec fields only** (`name, description, license, compatibility: "Requires uv and ANTHROPIC_API_KEY", metadata`). Put Claude-only extras in the plugin, not the shared file.
- **Engine resolution**: `OPTIMIZE_ANYTHING_ENGINE` env, else `uvx --from optimize-anything optimize-anything`, else `<pkg>/../engine`. Make the runs dir configurable via `OPTIMIZE_ANYTHING_HOME`.
- **Startup `doctor`** (CLI subcommand + check on first tool call). Verify `uv` is on PATH, `ANTHROPIC_API_KEY` is visible, the engine imports, and the runs dir is writable. On failure return `isError:true` naming the fix for **this** client, e.g. "Codex: add `env_vars=[\"ANTHROPIC_API_KEY\",\"PATH\"]` to [mcp_servers.optimize-anything]".
- **Per-agent snippets** (README + `references/install.md`):
  - Claude plugin `.mcp.json`: `{"command":"node","args":["${CLAUDE_PLUGIN_ROOT}/mcp-server/dist/index.js"],"env":{"ANTHROPIC_API_KEY":"${ANTHROPIC_API_KEY}"}}`
  - Codex `config.toml`: `[mcp_servers.optimize-anything] command="npx" args=["-y","optimize-anything-mcp"] env_vars=["ANTHROPIC_API_KEY","PATH"]`. With long-poll (≤50s, see #3) under the 60s `tool_timeout_sec` default, no override is needed.
  - Gemini `gemini-extension.json`: `mcpServers` with `${extensionPath}`, `settings:[{"name":"Anthropic key","envVar":"ANTHROPIC_API_KEY","sensitive":true}]`, plus `skills/optimize-anything/`.
  - OpenCode: `"mcp":{"servers":{"optimize-anything":{"type":"local","command":["npx","-y","optimize-anything-mcp"],"environment":{"ANTHROPIC_API_KEY":"{env:ANTHROPIC_API_KEY}"}}}}`
- **CLI fallback in the skill**: "If the optimize-anything MCP tools aren't available, run `optimize-anything run --config cfg.json --events > run.log &` and poll `optimize-anything status --run-dir …`." A broken MCP install stops being a dead end. Caveat for Codex: the default `workspace-write` sandbox has network off unless `[sandbox_workspace_write] network_access = true` (developers.openai.com/codex/agent-approvals-security), so the CLI fallback can't reach the Anthropic API there without that setting. MCP servers reportedly run outside the sandbox (agent37.com/blog/codex-sandbox; unverified against official docs).
- **Tool references in the skill**: say "the `optimize-anything` server's `start_optimization` tool". Do **not** hardcode `mcp__…` prefixes, which differ between plugin and plain installs (this departs from Anthropic's `Server:tool` advice on purpose). Optionally declare the MCP server in Codex `agents/openai.yaml` `dependencies.tools`.

### #2 Formulation guardrails in the tool surface (C)
- **`optimize_anything` input schema**:
  - `evaluator` uses `oneOf`: `{type:"python", code}`, `{type:"shell", command, score_pattern}`, `{type:"llm_judge", criteria}`. Reject `{{candidate}}` inside shell `command` with a "use {{candidate_file}}" fix.
  - `budget` is **required**: `{max_metric_calls?: int, max_cost_usd?: number, stop_at_score?: number, timeout_minutes?: number}`, with at least one cap.
  - `dataset`/`valset` are arrays of objects.
  - `config` is typed: `{proposer_model, judge_model, reflection_minibatch_size}`, defaulting to server env, not hardcoded IDs.
  - Add `dry_run: boolean`.
- **`dry_run:true`** (or a separate `validate_optimization`) evaluates the seed on ≤3 examples, **hard-capped at 45s total** to stay under Codex's 60s. On the cap it returns partial results with the warning "timed out at example k; evaluator too slow for interactive dry-run". It returns `{seed_score, per_example_scores, sample_asi, est_cost_usd, est_minutes, warnings[]}`. Warnings to fire:
  - ASI empty
  - all scores identical or at the max (no signal / skip-perfect)
  - generalization intent but no `valset`
  - valset < 10 or overlaps dataset
  - judge model = proposer model
  - score outside the documented range
  - evaluator took longer than the timeout
- **Return contract**: `structuredContent` + text copy, and an `outputSchema` on every tool. Input errors come back as `isError` with fix text, per the MCP tools spec.
- **Skill body** (~120 lines): a "formulate → dry-run → fix → run → poll → verify on held-out → apply" checklist; one recipe per evaluator type with `{{candidate_file}}`; the mistakes list (§5) in `references/mistakes.md`; config templates in `assets/`. No model IDs, no dates.

### #3 Truthful, cheap polling that fits every client's timeouts
- Fix process-manager:
  - **decouple runs from the MCP server process.** Spawn detached (`detached:true`, `unref()`); the engine writes `<run_dir>/events.jsonl` itself instead of using a stdout pipe; record `pid` in `run.json`. Each new agent session starts a fresh server with an empty in-memory map, so status must be rebuilt from disk + PID liveness. Whether today's runs die when the old server exits is unverified, and the fix holds either way.
  - line-buffer stdout
  - persist `events.jsonl` + `status.json` on every `iteration`/`new_best` event
  - mark runs without a terminal event and without a live PID as `failed`/`orphaned`, not `completed`
- `check_optimization(run_id, wait_seconds=0..50, since_event?, response_format="concise"|"detailed")`:
  - long-polls until a new best or a terminal state, capped at 50s (under Codex's 60s), and emits `notifications/progress` while waiting (resets Claude's idle timer)
  - concise output (~150 tokens): `{status, iteration, metric_calls_used/budget, cost_usd, best_val_score, seed_score, last_improvement_iter, last_error, next_poll_hint_s}`
- `get_best_candidate(run_id, include="best"|"frontier", max_chars=8000)`: returns val score, train score, a diff vs seed, and `candidate_path`. Above `max_chars`, return a `resource_link`/path instead of inlining (Claude warns at 10k tokens and spills over 50k chars). Output limits are undocumented (?) for Gemini CLI and OpenCode, and for Codex's global default (only per-tool `output_token_limit` is documented). `max_chars=8000` is chosen to be safe under those unknowns.
- Rename `optimize_anything` to `start_optimization` so the description says it returns right away. Add a "Call check_optimization with wait_seconds=45; do other work between polls" line in the tool description.
- MCP Tasks is an **optional add-on later** (return `resultType:"task"` only when the client declares the extension). Keep start/poll as the baseline.

**Measure #1–#3**: trigger eval (20 queries, 60/40, 3 runs, near misses like "optimize this SQL query", "make my React app faster", "tune hyperparameters with Optuna") per agent. Detection signals: Claude `claude plugin eval` with the `tool_used: Skill` grader (don't trust `claude -p` alone, see issue 36570); OpenCode a `skill` tool call; Gemini `activate_skill` (needs headless auto-approve); Codex **?** (signal not documented). Track correct use with a task eval: % of agent-built runs that pass dry-run with no warnings, have a budget set, and pick the final candidate by val.

## 5. Formulation mistakes to warn about in the skill
1. **One averaged scalar hides weak aspects.** Return per-example/per-aspect `scores` (GEPA blog 2026/02/18).
2. **Score-only evaluator.** Without ASI/feedback the proposer is blind; DSPy falls back to a generic score-only message (dspy.ai GEPA overview). Include errors, failing input, expected vs got, runtime.
3. **No budget means an unbounded run** (GEPA API reference warns; repo default is 100 Opus iterations).
4. **No held-out valset in generalization mode, or picking the last instead of the best-by-val iteration** (agentskills optimizing-descriptions; Anthropic tools blog says to use held-out tests).
5. **Train/val leakage or overlap; valset too small to rank candidates** (agentskills: stratified, fixed split).
6. **Lower-is-better metrics passed raw.** Score is higher-is-better, so negate or invert cost/latency (GEPA API).
7. **Hard constraints only stated in `objective` text.** Enforce them in the evaluator (score 0 + ASI on violation). The GEPA Can't-Be-Late example states MUST constraints in background, and its evaluator scores them.
8. **Noisy/nondeterministic evaluator scored once.** Repeat or seed (agentskills recommends 3 runs per query).
9. **Inline `{{candidate}}` in shell commands.** Quoting breaks and injection (repo `shell_eval.py:29`). Use `{{candidate_file}}`.
10. **Expensive task model during search.** Optimize with a cheap task LM and a strong reflector, then deploy strong (GEPA blog 2026/09/29); skip this on prompt-insensitive tasks.
11. **Judge = proposer model / vague judge rubric.** Use a concrete rubric with a parseable `SCORE:`. (Engineering judgment; the "Bridging the Subjectivity Gap" GEPA post is relevant but not read in full.)
12. **Mutating the whole file when only a region matters.** Split it into named components, or scope an EVOLVE-BLOCK-style region (OpenEvolve README).
13. **Seed already perfect on the minibatch.** `skip_perfect_score` means no learning signal; harden the eval set first (repo `config.py`).

## 6. Advise AGAINST
- **MCP Tasks as the only interface.** It is an extension; the client matrix lists no supporting client; Claude Code has it in beta for the main conversation only (not subagents, not Bedrock).
- **Blocking tool calls for the whole run.** Codex's 60s default timeout kills them; Claude auto-backgrounds after 2 min.
- **Claude-only frontmatter in the shared SKILL.md.** claude.ai and the Skills API hard-fail on unknown keys. Keep `when_to_use`, `context` etc. out.
- **Broad trigger words** ("optimize", "improve", "evolve"). They over-trigger on near misses and waste listing budget. Lead with the specific use: iterative search over a text artifact with an evaluator.
- **Hardcoded `mcp__…` tool names, model IDs, absolute paths, or dated notes in the skill.**
- **Dumping checkpoints or full frontiers into tool output** (25k-token cap; 10k warning).
- **More overlapping tools.** Fold `dry_run` into start, keep ~5 tools (Anthropic tools blog).
- **Relying on inherited env.** Gemini extensions only receive declared vars (documented). Codex documents an `env_vars` allowlist; that undeclared vars, including `PATH`, aren't forwarded is *inferred*. Declare `ANTHROPIC_API_KEY` explicitly everywhere.
- **Passing `temperature`** if the SDK moves to API version 2026-10-21 (400 error; model-deprecations page). Omit it by default.
- **Host-agent-as-proposer (F) as the default.** Interesting for no-key setups, but it burns the agent's context and has no evidence base yet. Offer it later as an engine option, mirroring GEPA omni's `meta_harness`.
