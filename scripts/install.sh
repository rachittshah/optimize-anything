#!/usr/bin/env bash
# One-command install of the optimize-anything MCP server + skill for Claude Code, Codex CLI,
# OpenCode and Gemini CLI. Idempotent; backs up every config it changes to
# ~/.optimize-anything/backups/<timestamp>/. API keys are never written (names only).
#
#   scripts/install.sh [--agents claude,codex,opencode,gemini] [--dry-run] [--link]
#   scripts/install.sh --uninstall [--agents ...]
#   scripts/install.sh --doctor [--agents ...] [--live claude,codex,...]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AGENTS=""
LIVE=""
MODE="install"
DRY=""
LINK=""

usage() { sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --agents) AGENTS="${2:-}"; shift 2 ;;
    --agents=*) AGENTS="${1#*=}"; shift ;;
    --live) LIVE="${2:-}"; shift 2 ;;
    --live=*) LIVE="${1#*=}"; shift ;;
    --dry-run) DRY="--dry-run"; shift ;;
    --link) LINK="--link"; shift ;;
    --uninstall) MODE="uninstall"; shift ;;
    --doctor) MODE="doctor"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

die() { echo "error: $*" >&2; exit 1; }
step() { echo "==> $*"; }

# ---- prerequisites
NODE="$(command -v node || true)"
[ -n "$NODE" ] || die "node not found (need Node >= 18): https://nodejs.org"
NODE_MAJOR="$("$NODE" -p 'process.versions.node.split(".")[0]')"
[ "$NODE_MAJOR" -ge 18 ] || die "node $("$NODE" --version) is too old; need >= 18"
command -v npm >/dev/null || die "npm not found"
UV="${OA_UV:-$(command -v uv || true)}"
[ -n "$UV" ] || die "uv not found: curl -LsSf https://astral.sh/uv/install.sh | sh"

engine_python() {
  if [ -x "$REPO/engine/.venv/bin/python" ]; then echo "$REPO/engine/.venv/bin/python"; else echo ""; fi
}

if [ "$MODE" = "doctor" ]; then
  PY="$(engine_python)"
  [ -n "$PY" ] || die "engine venv missing; run scripts/install.sh first"
  exec "$PY" "$REPO/scripts/doctor.py" ${AGENTS:+--agents "$AGENTS"} ${LIVE:+--live "$LIVE"}
fi

# ---- build engine + MCP server
if [ "$MODE" = "install" ]; then
  if [ -n "$DRY" ]; then
    step "would run: (cd engine && uv sync) and (cd mcp-server && npm ci && npm run build)"
  else
    step "engine: uv sync"
    (cd "$REPO/engine" && "$UV" sync --inexact --quiet)
    step "mcp-server: npm ci && npm run build"
    (cd "$REPO/mcp-server" && { npm ci --no-audit --no-fund --silent || npm install --no-audit --no-fund --silent; } && npm run --silent build)
    [ -f "$REPO/mcp-server/dist/index.js" ] || die "build did not produce mcp-server/dist/index.js"
  fi
fi

PY="$(engine_python)"
if [ -z "$PY" ]; then
  [ -n "$DRY" ] || die "engine venv missing after uv sync"
  PY="$("$UV" python find 3.12 2>/dev/null || true)"
  [ -n "$PY" ] || die "no Python >= 3.12 found for the dry run; run without --dry-run once"
fi
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' \
  || die "engine Python is $("$PY" --version 2>&1); need >= 3.12 (uv python install 3.12)"

# ---- register with agents + install skill
step "${MODE}${DRY:+ (dry run)}: agents=${AGENTS:-all found on PATH}"
set +e
"$PY" "$REPO/scripts/oa_install.py" --repo "$REPO" --node "$NODE" --uv "$UV" \
  ${AGENTS:+--agents "$AGENTS"} $DRY $LINK $([ "$MODE" = "uninstall" ] && echo --uninstall)
RC=$?
set -e
if [ "$MODE" = "install" ] && [ -z "$DRY" ] && [ $RC -eq 0 ]; then
  echo "done. Verify with: scripts/install.sh --doctor"
fi
exit $RC
