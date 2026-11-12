// Run status derivation. Pure: callers read the files and pass a snapshot in.
import { compactEvent, EngineEvent } from './lines.js';

export type RunState = 'running' | 'completed' | 'failed' | 'stopped';

export interface RunSnapshot {
  runId: string;
  result: any | null; // parsed result.json
  checkpoint: any | null; // parsed checkpoint.json
  events: EngineEvent[]; // tail of events.jsonl (oldest first)
  pidAlive: boolean | null; // null = no pid file (e.g. a run from before detach support)
  stopped: boolean; // <run_dir>/stopped marker
  exitCode: number | null; // <run_dir>/exit_code written by the wrapper shell
  stderrTail: string; // last lines of stderr.log
  startedAt: number | null; // ms epoch
  now: number; // ms epoch
}

export interface RunStatus {
  run_id: string;
  status: RunState;
  detail?: string;
  iteration?: number;
  best_score?: number | null;
  seed_score?: number | null;
  total_evals?: number;
  cost_usd?: number;
  stop_reason?: string | null;
  elapsed_s?: number;
  last_events?: Record<string, unknown>[];
  error?: string;
}

function num(x: unknown): number | undefined {
  return typeof x === 'number' && Number.isFinite(x) ? x : undefined;
}

/** Best score the engine reports: checkpoint.best_idx when valid (differs from argmax under
 *  promotion=paired_se), else argmax of agg_scores. */
export function checkpointBest(cp: any): { idx: number; score: number } | null {
  const scores: unknown = cp?.agg_scores;
  if (!Array.isArray(scores) || scores.length === 0) return null;
  const bi = cp.best_idx;
  if (Number.isInteger(bi) && bi >= 0 && bi < scores.length && typeof scores[bi] === 'number') {
    return { idx: bi, score: scores[bi] };
  }
  let idx = 0;
  for (let i = 1; i < scores.length; i++) if ((scores[i] as number) > (scores[idx] as number)) idx = i;
  return { idx, score: scores[idx] as number };
}

function lastOf(events: EngineEvent[], type: string): EngineEvent | undefined {
  for (let i = events.length - 1; i >= 0; i--) if (events[i].type === type) return events[i];
  return undefined;
}

export function deriveStatus(s: RunSnapshot): RunStatus {
  const complete = lastOf(s.events, 'optimization_complete');
  const out: RunStatus = { run_id: s.runId, status: 'running' };

  // State. result.json is written only after a run finishes normally.
  if (s.result) {
    out.status = 'completed';
  } else if (s.stopped) {
    out.status = 'stopped';
    out.detail = 'stopped by stop_optimization; best-so-far is in checkpoint.json';
  } else if (s.pidAlive) {
    out.status = 'running';
  } else if (s.exitCode === 0 && complete) {
    out.status = 'completed'; // finished but result.json missing (unusual)
  } else {
    out.status = 'failed';
    const why =
      s.exitCode !== null && s.exitCode !== 0
        ? `engine exited with code ${s.exitCode}`
        : s.pidAlive === null
          ? 'no live process and no pid file'
          : 'engine process is gone';
    out.detail = s.checkpoint
      ? `interrupted: ${why}; checkpoint at iteration ${s.checkpoint.iteration ?? '?'} but no result.json`
      : `${why} before the first checkpoint`;
    if (s.stderrTail) out.error = s.stderrTail.slice(-1500);
  }

  // Progress. Events are fresher than the checkpoint (written every few iterations).
  const lastIter = lastOf(s.events, 'iteration_start');
  const iters = [num(s.result?.total_iterations), num(lastIter?.iteration), num(s.checkpoint?.iteration)].filter(
    (x): x is number => x !== undefined,
  );
  if (iters.length) out.iteration = Math.max(...iters);

  const cpBest = checkpointBest(s.checkpoint);
  const lastBest = lastOf(s.events, 'new_best_found');
  out.best_score =
    num(s.result?.best_score) ?? num(complete?.best_score) ?? cpBest?.score ?? num(lastBest?.score) ?? null;
  out.seed_score = num(s.result?.seed_score) ?? num(s.checkpoint?.seed_score) ?? num(complete?.seed_score) ?? null;
  const evals = num(s.result?.total_evals) ?? num(complete?.total_evals) ?? num(s.checkpoint?.total_evals);
  if (evals !== undefined) out.total_evals = evals;

  let cost = num(s.result?.total_cost_usd) ?? num(complete?.total_cost_usd);
  if (cost === undefined) {
    for (let i = s.events.length - 1; i >= 0 && cost === undefined; i--) cost = num(s.events[i].cost_usd);
  }
  cost = cost ?? num(s.checkpoint?.cost?.total_usd);
  if (cost !== undefined) out.cost_usd = Math.round(cost * 10000) / 10000;

  out.stop_reason = (s.result?.stop_reason as string) ?? (complete?.stop_reason as string) ?? null;
  if (s.startedAt) out.elapsed_s = Math.round((s.now - s.startedAt) / 1000);
  out.last_events = s.events.slice(-3).map((e) => compactEvent(e));
  return out;
}

export function isTerminal(st: RunState): boolean {
  return st !== 'running';
}

/** Long-poll exit test: stop waiting on a terminal state or a better best score. */
export function shouldReturnEarly(before: RunStatus, now: RunStatus): boolean {
  if (isTerminal(now.status)) return true;
  const a = before.best_score;
  const b = now.best_score;
  return typeof b === 'number' && (typeof a !== 'number' || b > a);
}
