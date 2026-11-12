#!/usr/bin/env bash
# Reproduce the A/B rounds reported in RESULTS.md.
# Usage: scripts/bench/ab_rounds.sh <round>   (round = 1 | 2 | sonnet)
# Round 1 needs a worktree of the original engine:  git worktree add --detach "$MAIN_WT" 1f77560
set -euo pipefail
cd "$(dirname "$0")/../.."
round="$1"
export SEEDS="${SEEDS:-0 1 2 3 4}"
P='"promotion":"paired_se"'
M=scripts/bench/run_matrix.sh
case "$round" in
  1)
    export OUTDIR=scripts/bench/results/ab1
    $M main --engine-src "${MAIN_WT:?set MAIN_WT to a worktree of the original engine}/engine/src"
    $M core
    $M promo --config-json "{$P}"
    $M halving --config-json '{"val_eval":"halving"}'
    $M prescreen --config-json '{"prescreen":true}'
    $M cegis --config-json '{"counterexample_mining":true}'
    $M section --config-json '{"mutation_operator":"section_edit"}'
    $M bandit --config-json '{"operator_bandit":true}'
    ;;
  2)
    export OUTDIR=scripts/bench/results/ab2
    $M promo_memory --config-json "{$P,\"proposal_memory\":true}"
    $M promo_editcap --config-json "{$P,\"max_edit_ratio\":0.6}"
    $M promo_halving --config-json "{$P,\"val_eval\":\"halving\"}"
    $M promo_bandit --config-json "{$P,\"operator_bandit\":true}"
    $M promo_cegis --config-json "{$P,\"counterexample_mining\":true}"
    $M promo_rep --config-json "{$P}"
    ;;
  sonnet)
    export OUTDIR=scripts/bench/results/ab_sonnet MODEL=claude-sonnet-5-5
    $M main --engine-src "${MAIN_WT:?set MAIN_WT}/engine/src"
    $M core
    $M promo --config-json "{$P}"
    ;;
  *) echo "unknown round: $round" >&2; exit 2 ;;
esac
echo "ROUND_${round}_DONE"
