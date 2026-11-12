import { test } from 'node:test';
import assert from 'node:assert/strict';
import { validateStart, DEFAULT_BUDGET } from '../dist/guardrails.js';
import { capText } from '../dist/output.js';

const py = { type: 'python', code: 'def evaluate(candidate, example):\n    return 1.0, {}' };
const seed = { text: 'hello' };

test('valid python run passes and gets the default budget with a note', () => {
  const v = validateStart({ seed_candidate: seed, evaluator: py });
  assert.deepEqual(v.errors, []);
  assert.equal(v.runConfig.config.max_metric_calls, DEFAULT_BUDGET.max_metric_calls);
  assert.equal(v.runConfig.config.max_cost_usd, DEFAULT_BUDGET.max_cost_usd);
  assert.match(v.notes.join(' '), /No budget given/);
});

test('explicit budget is kept and no default is added', () => {
  const v = validateStart({ seed_candidate: seed, evaluator: py, config: { max_iterations: 2, backend: 'mock' } });
  assert.deepEqual(v.errors, []);
  assert.equal(v.runConfig.config.max_metric_calls, undefined);
  assert.equal(v.runConfig.config.max_iterations, 2);
});

test('python without evaluate() is rejected with a fix', () => {
  const v = validateStart({ seed_candidate: seed, evaluator: { type: 'python', code: 'print(1)' } });
  assert.match(v.errors[0], /def evaluate\(candidate, example\)/);
});

test('shell with inline {{candidate}} is rejected and points to {{candidate_file}}', () => {
  const v = validateStart({ seed_candidate: seed, evaluator: { type: 'shell', command: "echo '{{candidate}}' | wc -w" } });
  assert.equal(v.errors.length, 1);
  assert.match(v.errors[0], /\{\{candidate_file\}\}/);
  assert.match(v.errors[0], /quote/);
});

test('shell without command or without candidate reference is rejected', () => {
  assert.equal(validateStart({ seed_candidate: seed, evaluator: { type: 'shell' } }).errors.length, 1);
  assert.match(validateStart({ seed_candidate: seed, evaluator: { type: 'shell', command: 'python score.py' } }).errors[0], /never references/);
  assert.deepEqual(validateStart({ seed_candidate: seed, evaluator: { type: 'shell', command: 'python s.py {{candidate_file}}', score_pattern: 'S=(\\d+)' } }).errors, []);
  assert.match(validateStart({ seed_candidate: seed, evaluator: { type: 'shell', command: 'x {{candidate_file}}', score_pattern: '(' } }).errors[0], /regex/);
});

test('llm_judge needs criteria or rubric', () => {
  assert.equal(validateStart({ seed_candidate: seed, evaluator: { type: 'llm_judge' } }).errors.length, 1);
  assert.deepEqual(validateStart({ seed_candidate: seed, evaluator: { type: 'llm_judge', criteria: 'concise' } }).errors, []);
  assert.deepEqual(validateStart({ seed_candidate: seed, evaluator: { type: 'llm_judge', rubric: ['clarity'] } }).errors, []);
});

test('unknown evaluator type, missing seed and objective, bad seed values', () => {
  assert.match(validateStart({ seed_candidate: seed, evaluator: { type: 'regex' } }).errors[0], /evaluator.type/);
  assert.match(validateStart({ evaluator: py }).errors[0], /seed_candidate/);
  assert.deepEqual(validateStart({ evaluator: py, objective: 'write a haiku' }).errors, []);
  assert.match(validateStart({ seed_candidate: { a: 1 }, evaluator: py }).errors[0], /must be a string/);
});

test('generalization objective with dataset but no valset warns', () => {
  const v = validateStart({ seed_candidate: seed, evaluator: py, dataset: [{ q: 1 }], objective: 'generalize to unseen questions' });
  assert.deepEqual(v.errors, []);
  assert.match(v.warnings.join(' '), /valset/);
  const quiet = validateStart({ seed_candidate: seed, evaluator: py, dataset: [{ q: 1 }], objective: 'score well on these' });
  assert.equal(quiet.warnings.length, 0);
});

test('preset and testset pass through; bad preset and bad budget rejected', () => {
  const v = validateStart({ seed_candidate: seed, evaluator: py, preset: 'cheap', testset: [{ q: 1 }] });
  assert.equal(v.runConfig.config.preset, 'cheap');
  assert.deepEqual(v.runConfig.testset, [{ q: 1 }]);
  assert.match(validateStart({ seed_candidate: seed, evaluator: py, preset: 'turbo' }).errors[0], /preset/);
  assert.match(validateStart({ seed_candidate: seed, evaluator: py, config: { max_cost_usd: -1 } }).errors[0], /positive/);
});

test('capText keeps responses under the cap', () => {
  const t = capText('x'.repeat(20000));
  assert.ok(t.length <= 8000);
  assert.match(t, /truncated/);
  assert.equal(capText('short'), 'short');
});

test('fitList drops rows so the JSON stays valid and under the cap', async () => {
  const { fitList } = await import('../dist/output.js');
  const big = { runs: Array.from({ length: 200 }, (_, i) => ({ run_id: 'r' + i, detail: 'x'.repeat(100) })), total: 200 };
  const out = fitList(big, 'runs');
  const text = JSON.stringify(out, null, 2);
  assert.ok(text.length <= 8000);
  assert.ok(out.omitted > 0 && out.runs.length + out.omitted === 200);
  assert.doesNotThrow(() => JSON.parse(capText(text)));
});
