// Tool responses stay under ~8k chars (Claude warns at 10k tokens; other clients' limits are undocumented).
export const MAX_RESPONSE_CHARS = 8000;

export function capText(text: string, max = MAX_RESPONSE_CHARS): string {
  if (text.length <= max) return text;
  const note = `\n…[truncated ${text.length - max} of ${text.length} chars; narrow the request (e.g. list_optimization_runs limit) or read files in the run dir]`;
  return text.slice(0, max - note.length) + note;
}

/** Drop trailing items of obj[key] until the pretty JSON fits; records how many were omitted. */
export function fitList<T extends Record<string, any>>(obj: T, key: keyof T, max = MAX_RESPONSE_CHARS): T {
  const items = [...(obj[key] as unknown[])];
  let out: any = { ...obj, [key]: items };
  while (items.length && JSON.stringify(out, null, 2).length > max) {
    items.pop();
    out = { ...obj, [key]: items, omitted: (obj[key] as unknown[]).length - items.length,
            note: 'response size cap reached; pass a smaller limit to see fewer, or read the runs dir' };
  }
  return out;
}
