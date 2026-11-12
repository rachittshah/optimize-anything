import { test } from 'node:test';
import assert from 'node:assert/strict';
import { LineBuffer, parseEventLine, consumeChunk, compactEvent } from '../dist/lines.js';

test('LineBuffer joins a JSON line split across chunks', () => {
  const b = new LineBuffer();
  const line = JSON.stringify({ type: 'new_best_found', score: 0.5 });
  assert.deepEqual(b.push(line.slice(0, 10)), []);
  assert.deepEqual(b.push(line.slice(10) + '\n{"type":"x"'), [line]);
  assert.deepEqual(b.push('}\n'), ['{"type":"x"}']);
  assert.deepEqual(b.flush(), []);
});

test('LineBuffer handles CRLF, empty lines and flush of a partial tail', () => {
  const b = new LineBuffer();
  assert.deepEqual(b.push('a\r\n\nb'), ['a']);
  assert.deepEqual(b.flush(), ['b']);
});

test('parseEventLine accepts only objects with a string type', () => {
  assert.deepEqual(parseEventLine('{"type":"iteration_start","iteration":1}'), { type: 'iteration_start', iteration: 1 });
  assert.equal(parseEventLine('{"best_score": 1}'), null); // final result JSON (no type)
  assert.equal(parseEventLine('{'), null); // first line of indented result JSON
  assert.equal(parseEventLine('[1,2]'), null);
  assert.equal(parseEventLine('{"type": 3}'), null);
  assert.equal(parseEventLine('not json'), null);
});

test('consumeChunk never advances past a partial trailing line', () => {
  const full = '{"type":"a"}\n{"type":"b"}\n{"type":"c"';
  const r1 = consumeChunk(Buffer.from(full), 100);
  assert.deepEqual(r1.events.map((e) => e.type), ['a', 'b']);
  assert.equal(r1.nextOffset, 100 + full.lastIndexOf('\n') + 1);
  const r2 = consumeChunk(Buffer.from('{"type":"c"'), 0);
  assert.equal(r2.nextOffset, 0);
  assert.equal(r2.events.length, 0);
});

test('consumeChunk skips the multi-line result JSON printed at the end', () => {
  const out = '{"type":"optimization_complete","best_score":1}\n{\n  "best_score": 1\n}\n';
  const r = consumeChunk(Buffer.from(out), 0);
  assert.deepEqual(r.lines, ['{"type":"optimization_complete","best_score":1}']);
  assert.equal(r.nextOffset, Buffer.byteLength(out));
});

test('consumeChunk counts bytes, not chars, for the offset (UTF-8)', () => {
  const s = '{"type":"a","t":"é…"}\n';
  assert.equal(consumeChunk(Buffer.from(s), 0).nextOffset, Buffer.byteLength(s));
});

test('compactEvent drops asi and clips long strings', () => {
  const c = compactEvent({ type: 'evaluation_complete', score: 1, asi: { big: 'x'.repeat(1000) }, reason: 'y'.repeat(500) });
  assert.equal(c.asi, undefined);
  assert.ok(String(c.reason).length < 200);
});
