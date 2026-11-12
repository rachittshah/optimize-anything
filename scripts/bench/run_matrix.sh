#!/usr/bin/env bash
# Run one bench variant over every (task, seed) pair in parallel.
# Usage: scripts/bench/run_matrix.sh <variant-name> [extra run_bench.py args...]
# Example:
#   scripts/bench/run_matrix.sh main --engine-src /path/to/main-worktree/engine/src
#   scripts/bench/run_matrix.sh sota --config-json '{"preset":"cheap"}'
# Env: TASKS (default: all 3), SEEDS (default: "0 1 2"), BUDGET (max metric calls, default 300),
#      MODEL (proposer, default claude-haiku-5-5), OUTDIR (default scripts/bench/results).
set -euo pipefail
variant="$1"; shift
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
tasks="${TASKS:-regex_emails parse_dates ticket_router}"
seeds="${SEEDS:-0 1 2}"
budget="${BUDGET:-300}"
model="${MODEL:-claude-haiku-5-5}"
outdir="${OUTDIR:-$repo/scripts/bench/results}"
mkdir -p "$outdir/logs"
outdir="$(cd "$outdir" && pwd)"  # absolute: jobs cd into engine/
out="$outdir/$variant.jsonl"
: > "$out.tmp"

pids=()
for t in $tasks; do
  for s in $seeds; do
    part="$outdir/logs/$variant.$t.$s.jsonl"
    rm -f "$part"
    (cd "$repo/engine" && uv run python ../scripts/bench/run_bench.py \
        --tasks "$t" --seeds "$s" --max-iterations 200 --max-metric-calls "$budget" \
        --proposer claude --proposer-model "$model" --variant-name "$variant" \
        --out "$part" "$@" > "$outdir/logs/$variant.$t.$s.log" 2>&1) &
    pids+=($!)
  done
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
cat "$outdir"/logs/"$variant".*.jsonl > "$out" 2>/dev/null || true
rm -f "$out.tmp"
echo "variant=$variant runs=$(wc -l < "$out" | tr -d ' ') failed_jobs=$fail -> $out"
