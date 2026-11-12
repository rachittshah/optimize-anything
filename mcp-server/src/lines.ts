// Line splitting for engine stdout. Pure: no fs, no globals.

/** Accumulates text chunks and yields only complete lines. A chunk that ends mid-line
 *  keeps the partial tail until the rest arrives (the old per-chunk JSON.parse dropped it). */
export class LineBuffer {
  private tail = '';

  push(chunk: string): string[] {
    const text = this.tail + chunk;
    const parts = text.split('\n');
    this.tail = parts.pop() ?? '';
    return parts.map((l) => l.replace(/\r$/, '')).filter((l) => l.length > 0);
  }

  /** Bytes (UTF-8) of the partial line still buffered. */
  pendingBytes(): number {
    return Buffer.byteLength(this.tail, 'utf8');
  }

  /** Remaining partial line (use at end of stream). */
  flush(): string[] {
    const rest = this.tail.replace(/\r$/, '');
    this.tail = '';
    return rest ? [rest] : [];
  }
}

export interface EngineEvent {
  type: string;
  [key: string]: unknown;
}

/** An engine event is a single-line JSON object with a string `type`. Anything else on
 *  stdout (the final indented result JSON, stray prints) is ignored. */
export function parseEventLine(line: string): EngineEvent | null {
  const s = line.trim();
  if (!s.startsWith('{') || !s.endsWith('}')) return null;
  try {
    const obj = JSON.parse(s);
    if (obj && typeof obj === 'object' && !Array.isArray(obj) && typeof obj.type === 'string') {
      return obj as EngineEvent;
    }
  } catch {
    /* not an event line */
  }
  return null;
}

/** Split a byte chunk read from `offset` into complete event lines. Returns the events and
 *  the new offset (end of the last complete line), so a partial trailing line is re-read
 *  next time instead of being lost. */
export function consumeChunk(chunk: Buffer, offset: number): { events: EngineEvent[]; lines: string[]; nextOffset: number } {
  const lastNl = chunk.lastIndexOf(0x0a);
  if (lastNl < 0) return { events: [], lines: [], nextOffset: offset };
  const complete = chunk.subarray(0, lastNl + 1).toString('utf8');
  const buf = new LineBuffer();
  const lines = buf.push(complete);
  const events: EngineEvent[] = [];
  const kept: string[] = [];
  for (const line of lines) {
    const ev = parseEventLine(line);
    if (ev) {
      events.push(ev);
      kept.push(line.trim());
    }
  }
  return { events, lines: kept, nextOffset: offset + lastNl + 1 };
}

/** Event without bulky fields (asi, candidate text), long strings clipped. */
export function compactEvent(ev: EngineEvent, maxStr = 160): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(ev)) {
    if (k === 'asi' || k === 'candidate' || k === 'components') continue;
    if (typeof v === 'string') out[k] = v.length > maxStr ? v.slice(0, maxStr) + '…' : v;
    else if (v !== null && typeof v === 'object') {
      const s = JSON.stringify(v);
      out[k] = s.length > maxStr ? s.slice(0, maxStr) + '…' : v;
    } else out[k] = v;
  }
  return out;
}
