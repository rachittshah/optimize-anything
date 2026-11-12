import { test } from 'node:test';
import assert from 'node:assert/strict';
import { deriveStatus, shouldReturnEarly, checkpointBest } from '../dist/status.js';

const base = { runId: 'r1', result: null, checkpoint: null, events: [], pidAlive: true, stopped: false,
  exitCode: null, stderrTail: '', startedAt: 1000, now: 11000 };
const cp = { iteration: 3, agg_scores: [0.2, 0.9, 0.7], best_idx: 2, total_evals: 12, seed_score: 0.2, cost: { total_usd: 0.5 } };

test('running process with checkpoint -> running, best from best_idx (not argmax)', () => {
  const s = deriveStatus({ ...base, checkpoint: cp });
  assert.equal(s.status, 'running');
  assert.equal(s.best_score, 0.7);
  assert.equal(s.iteration, 3);
  assert.equal(s.total_evals, 12);
  assert.equal(s.cost_usd, 0.5);
  assert.equal(s.elapsed_s, 10);
});

test('checkpoint but no result.json and no live process -> failed/interrupted, not completed', () => {
  const s = deriveStatus({ ...base, checkpoint: cp, pidAlive: false });
  assert.equal(s.status, 'failed');
  assert.match(s.detail, /interrupted/);
});

test('legacy run dir without pid file -> failed/interrupted', () => {
  const s = deriveStatus({ ...base, checkpoint: cp, pidAlive: null });
  assert.equal(s.status, 'failed');
  assert.match(s.detail, /no pid file/);
});

test('non-zero exit code before first checkpoint reports stderr', () => {
  const s = deriveStatus({ ...base, pidAlive: false, exitCode: 1, stderrTail: 'Traceback: boom' });
  assert.equal(s.status, 'failed');
  assert.match(s.detail, /code 1.*before the first checkpoint/);
  assert.match(s.error, /boom/);
});

test('result.json -> completed with stop_reason; stopped marker -> stopped', () => {
  const done = deriveStatus({ ...base, pidAlive: false, exitCode: 0, checkpoint: cp,
    result: { best_score: 0.8, total_iterations: 5, total_evals: 20, total_cost_usd: 1.23456, stop_reason: 'budget', seed_score: 0.2 } });
  assert.equal(done.status, 'completed');
  assert.equal(done.best_score, 0.8);
  assert.equal(done.stop_reason, 'budget');
  assert.equal(done.iteration, 5);
  assert.equal(done.cost_usd, 1.2346);
  const st = deriveStatus({ ...base, pidAlive: false, stopped: true, checkpoint: cp });
  assert.equal(st.status, 'stopped');
});

test('events are fresher than the checkpoint and only the last 3 are returned, without asi', () => {
  const events = [
    { type: 'iteration_start', iteration: 4, candidate_idx: 0, cost_usd: 0.6 },
    { type: 'evaluation_complete', candidate_idx: 0, score: 0.5, asi: { x: 1 } },
    { type: 'proposal_complete', iteration: 4, component_name: 't', cost_usd: 0.7 },
    { type: 'candidate_rejected', iteration: 4, reason: 'minibatch' },
  ];
  const s = deriveStatus({ ...base, checkpoint: cp, events });
  assert.equal(s.iteration, 4);
  assert.equal(s.cost_usd, 0.7);
  assert.equal(s.last_events.length, 3);
  assert.equal(s.last_events[0].asi, undefined);
});

test('checkpointBest falls back to argmax when best_idx is missing or invalid', () => {
  assert.deepEqual(checkpointBest({ agg_scores: [0.1, 0.9, 0.3] }), { idx: 1, score: 0.9 });
  assert.deepEqual(checkpointBest({ agg_scores: [0.1, 0.9], best_idx: 7 }), { idx: 1, score: 0.9 });
  assert.equal(checkpointBest({}), null);
});

test('long-poll returns early on terminal state or new best only', () => {
  const a = { run_id: 'r', status: 'running', best_score: 0.5 };
  assert.equal(shouldReturnEarly(a, { ...a }), false);
  assert.equal(shouldReturnEarly(a, { ...a, best_score: 0.6 }), true);
  assert.equal(shouldReturnEarly(a, { ...a, status: 'completed' }), true);
  assert.equal(shouldReturnEarly({ ...a, best_score: null }, { ...a, best_score: 0 }), true);
});
