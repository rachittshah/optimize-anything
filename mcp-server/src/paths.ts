// Where things live. Env overrides first, because agents pass different env (Codex strips most).
import { existsSync, statSync } from 'fs';
import { join, dirname, delimiter } from 'path';
import { homedir } from 'os';
import { fileURLToPath } from 'url';

const HERE = dirname(fileURLToPath(import.meta.url)); // <repo>/mcp-server/dist

export function oaHome(): string {
  return join(homedir(), '.optimize-anything');
}

export function runsDir(): string {
  return process.env.OA_RUNS_DIR || join(oaHome(), 'runs');
}

export function dryRunsDir(): string {
  return join(oaHome(), 'dryruns');
}

/** Engine dir: $OA_ENGINE_DIR, else <repo>/engine relative to this file's dist dir. */
export function engineDir(): string {
  const candidates = [process.env.OA_ENGINE_DIR, join(HERE, '..', '..', 'engine')].filter(Boolean) as string[];
  for (const c of candidates) if (existsSync(join(c, 'pyproject.toml'))) return c;
  throw new Error(
    `optimize-anything engine not found (looked in: ${candidates.join(', ')}). ` +
      'Set OA_ENGINE_DIR=<repo>/engine in this MCP server entry (scripts/install.sh does this).',
  );
}

function isExecutable(p: string): boolean {
  try {
    return statSync(p).isFile();
  } catch {
    return false;
  }
}

/** uv binary: $OA_UV, else PATH, else common install locations. */
export function uvPath(): string {
  const env = process.env.OA_UV;
  if (env && isExecutable(env)) return env;
  for (const dir of (process.env.PATH || '').split(delimiter)) {
    if (dir && isExecutable(join(dir, 'uv'))) return join(dir, 'uv');
  }
  for (const p of [join(homedir(), '.local', 'bin', 'uv'), join(homedir(), '.cargo', 'bin', 'uv'), '/opt/homebrew/bin/uv', '/usr/local/bin/uv']) {
    if (isExecutable(p)) return p;
  }
  throw new Error(
    'uv not found on PATH. Install uv (https://docs.astral.sh/uv/) or set OA_UV=/abs/path/to/uv in this MCP server entry.',
  );
}
