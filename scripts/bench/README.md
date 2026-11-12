# Engine benchmark

3 deterministic tasks (`regex_emails`, `parse_dates`, `ticket_router` [2 components]), fixed 24/18/18
train/val/test splits in `data/` (regenerate with `make_data.py`). The engine sees train (minibatches) and val
(selection); the harness scores the final best candidate on test itself, outside `total_evals`.

```bash
cd engine   # WT = any path, e.g. a scratch dir; remove later with: git worktree remove <WT>
git worktree add --detach <WT> main          # baseline engine checkout
uv run python ../scripts/bench/run_bench.py --engine-src <WT>/engine/src \
  --tasks all --seeds 0 1 2 --max-iterations 10 --variant-name main --out ../scripts/bench/results/main.jsonl
uv run python ../scripts/bench/run_bench.py \
  --tasks all --seeds 0 1 2 --max-iterations 10 --variant-name sota --config-json '{"foo": 1}' \
  --out ../scripts/bench/results/sota.jsonl
python3 ../scripts/bench/summarize.py ../scripts/bench/results/*.jsonl
```

`--proposer mock` = no LLM. `--background full` hands the proposer the whole hidden spec (makes tasks easy).
Unknown `--config-json` keys are dropped and listed in `config_dropped`. `engine_module` records which engine ran.

`--out` appends: use a fresh file per variant. `seed` is not a Config field on main, so seeds vary only LLM sampling.
--engine-src reuses this tree's venv: the baseline needs its deps (e.g. `anthropic`) installed there.
