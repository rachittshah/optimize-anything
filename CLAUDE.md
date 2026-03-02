# optimize-anything

Universal text artifact optimizer using LLM-powered iterative search.

## Project Structure
- `engine/` — Python optimization engine (uv)
- `mcp-server/` — TypeScript MCP server (thin wrapper)
- `skill/` — Claude Code skill

## Commands
- `cd engine && uv sync` — install Python deps
- `cd engine && uv run pytest` — run tests
- `cd engine && uv run optimize-anything --help` — CLI
- `cd mcp-server && npm install && npm run build` — build MCP server

## Architecture
Python engine does all optimization logic (Pareto frontier, reflective mutation, evaluation).
TS MCP server spawns Python subprocess and manages run lifecycle.
Three evaluator types: Python code, shell command, LLM-as-judge.
Three optimization modes: single-task, multi-task, generalization.
