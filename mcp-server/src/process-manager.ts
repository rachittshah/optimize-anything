// Run lifecycle on disk. Runs are detached from the MCP server: everything a later server
// (or another agent session) needs is in <run_dir>:
//   config.json, pid, started_at, stdout.log, stderr.log, exit_code, events.jsonl,
//   .events_offset, stopped, checkpoint.json, result.json (engine), best.json
import { spawn, execFileSync } from 'child_process';
import {
  readFileSync, existsSync, writeFileSync, mkdirSync, readdirSync, openSync, closeSync,
  readSync, fstatSync, appendFileSync, statSync, unlinkSync, rmSync,
} from 'fs';
import { join } from 'path';
import { randomUUID } from 'crypto';
import { consumeChunk, parseEventLine, EngineEvent } from './lines.js';
import { deriveStatus, RunStatus, RunSnapshot, shouldReturnEarly, checkpointBest } from './status.js';
import { runsDir, dryRunsDir, engineDir, uvPath } from './paths.js';

const RUN_ID = /^[A-Za-z0-9_-]{1,64}$/;
const INLINE_CANDIDATE_MAX = 6000;

function readJson(p: string): any | null {
  try {
    return JSON.parse(readFileSync(p, 'utf-8'));
  } catch {
    return null;
  }
}

function readText(p: string): string | null {
  try {
    return readFileSync(p, 'utf-8');
  } catch {
    return null;
  }
}

function tailText(p: string, maxBytes: number): string {
  if (!existsSync(p)) return '';
  const fd = openSync(p, 'r');
  try {
    const size = fstatSync(fd).size;
    const start = Math.max(0, size - maxBytes);
    const buf = Buffer.alloc(size - start);
    readSync(fd, buf, 0, buf.length, start);
    return buf.toString('utf8');
  } finally {
    closeSync(fd);
  }
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export class ProcessManager {
  runDir(runId: string): string {
    if (!RUN_ID.test(runId)) throw new Error(`invalid run_id ${JSON.stringify(runId)}`);
    return join(runsDir(), runId);
  }

  exists(runId: string): boolean {
    return RUN_ID.test(runId) && existsSync(this.runDir(runId));
  }

  /** Start a detached engine process. runConfig comes from validateStart(). */
  startRun(runConfig: Record<string, any>): { runId: string; runDir: string } {
    const engine = engineDir();
    const uv = uvPath();
    const runId = randomUUID().slice(0, 8);
    const dir = this.runDir(runId);
    mkdirSync(dir, { recursive: true });
    const configPath = join(dir, 'config.json');
    writeFileSync(configPath, JSON.stringify({ ...runConfig, config: { ...runConfig.config, run_dir: dir } }, null, 2));
    writeFileSync(join(dir, 'started_at'), String(Date.now()));

    const out = openSync(join(dir, 'stdout.log'), 'a');
    const err = openSync(join(dir, 'stderr.log'), 'a');
    // sh waits for the engine and records its exit code, so a later server can tell a crash
    // from a clean exit. detached => own process group (pgid = sh pid) for stop_optimization.
    const child = spawn(
      '/bin/sh',
      ['-c', '"$0" "$@"; echo $? > "$OA_EXIT_FILE"', uv, 'run', '--project', engine, 'optimize-anything', 'run', '--config', configPath, '--events'],
      {
        cwd: engine,
        detached: true,
        stdio: ['ignore', out, err],
        env: { ...process.env, OA_EXIT_FILE: join(dir, 'exit_code'), PYTHONUNBUFFERED: '1' },
      },
    );
    closeSync(out);
    closeSync(err);
    if (!child.pid) throw new Error('failed to spawn the engine process');
    writeFileSync(join(dir, 'pid'), String(child.pid));
    child.on('error', (e) => appendFileSync(join(dir, 'stderr.log'), `spawn error: ${e.message}\n`));
    child.unref();
    return { runId, runDir: dir };
  }

  /** Is the run's process group leader alive, and is it really our process (pid reuse)? */
  pidAlive(runId: string): boolean | null {
    const dir = this.runDir(runId);
    const raw = readText(join(dir, 'pid'));
    if (raw === null) return null;
    const pid = parseInt(raw, 10);
    if (!Number.isInteger(pid) || pid <= 1) return false;
    try {
      process.kill(pid, 0);
    } catch (e: any) {
      if (e?.code !== 'EPERM') return false;
    }
    try {
      const cmd = execFileSync('ps', ['-ww', '-o', 'command=', '-p', String(pid)], { encoding: 'utf8' });
      return cmd.includes(dir);
    } catch {
      return false;
    }
  }

  /** Move complete event lines from stdout.log to events.jsonl. Offset-based, so a
   *  restarted server resumes without duplicates and split lines are never dropped. */
  ingest(runId: string): void {
    const dir = this.runDir(runId);
    const stdoutPath = join(dir, 'stdout.log');
    if (!existsSync(stdoutPath)) return;
    const lock = join(dir, '.ingest.lock');
    let lockFd: number;
    try {
      lockFd = openSync(lock, 'wx');
    } catch {
      try {
        if (Date.now() - statSync(lock).mtimeMs > 10_000) unlinkSync(lock); // stale lock
      } catch { /* raced */ }
      return; // another server is ingesting right now
    }
    try {
      const offPath = join(dir, '.events_offset');
      const offset = parseInt(readText(offPath) ?? '0', 10) || 0;
      const fd = openSync(stdoutPath, 'r');
      let chunk: Buffer;
      try {
        const size = fstatSync(fd).size;
        if (size <= offset) return;
        chunk = Buffer.alloc(Math.min(size - offset, 8 * 1024 * 1024));
        readSync(fd, chunk, 0, chunk.length, offset);
      } finally {
        closeSync(fd);
      }
      const { lines, nextOffset } = consumeChunk(chunk, offset);
      if (lines.length) appendFileSync(join(dir, 'events.jsonl'), lines.join('\n') + '\n');
      if (nextOffset !== offset) writeFileSync(offPath, String(nextOffset));
    } finally {
      closeSync(lockFd);
      try { unlinkSync(lock); } catch { /* already gone */ }
    }
  }

  readEvents(runId: string, maxBytes = 64 * 1024): EngineEvent[] {
    const text = tailText(join(this.runDir(runId), 'events.jsonl'), maxBytes);
    const lines = text.split('\n');
    if (text.length >= maxBytes) lines.shift(); // first line may be cut
    return lines.map(parseEventLine).filter((e): e is EngineEvent => e !== null);
  }

  snapshot(runId: string): RunSnapshot {
    const dir = this.runDir(runId);
    this.ingest(runId);
    const exitRaw = readText(join(dir, 'exit_code'));
    const started = parseInt(readText(join(dir, 'started_at')) ?? '', 10);
    let startedAt: number | null = Number.isFinite(started) ? started : null;
    if (startedAt === null) {
      try { startedAt = statSync(dir).birthtimeMs || statSync(dir).mtimeMs; } catch { /* none */ }
    }
    return {
      runId,
      result: readJson(join(dir, 'result.json')),
      checkpoint: readJson(join(dir, 'checkpoint.json')),
      events: this.readEvents(runId),
      pidAlive: this.pidAlive(runId),
      stopped: existsSync(join(dir, 'stopped')),
      exitCode: exitRaw !== null && exitRaw.trim() !== '' ? parseInt(exitRaw, 10) : null,
      stderrTail: tailText(join(dir, 'stderr.log'), 2000).trim(),
      startedAt,
      now: Date.now(),
    };
  }

  getStatus(runId: string): RunStatus | null {
    if (!this.exists(runId)) return null;
    return deriveStatus(this.snapshot(runId));
  }

  /** Long-poll: wait up to waitSeconds (max 50) for completion or a new best. */
  async waitStatus(runId: string, waitSeconds: number): Promise<RunStatus | null> {
    const first = this.getStatus(runId);
    if (!first) return null;
    const deadline = Date.now() + Math.min(Math.max(waitSeconds, 0), 50) * 1000;
    let cur = first;
    while (Date.now() < deadline && !shouldReturnEarly(first, cur)) {
      await sleep(Math.min(500, Math.max(0, deadline - Date.now())));
      cur = this.getStatus(runId) ?? cur;
    }
    return cur;
  }

  getBestCandidate(runId: string): Record<string, unknown> | null {
    if (!this.exists(runId)) return null;
    const dir = this.runDir(runId);
    const result = readJson(join(dir, 'result.json'));
    let candidate: Record<string, string> | null = null;
    let score: number | null = null;
    let source = '';
    if (result?.best_candidate) {
      candidate = result.best_candidate;
      score = result.best_score ?? null;
      source = 'result.json';
    } else {
      const cp = readJson(join(dir, 'checkpoint.json'));
      const best = checkpointBest(cp);
      if (cp && best) {
        candidate = cp.candidates?.[best.idx]?.components ?? null;
        score = best.score;
        source = `checkpoint.json (candidate ${best.idx}; run not finished)`;
      }
    }
    if (!candidate) return null;
    const bestPath = join(dir, 'best.json');
    const full = { run_id: runId, score, source, candidate };
    writeFileSync(bestPath, JSON.stringify(full, null, 2));
    if (JSON.stringify(candidate).length <= INLINE_CANDIDATE_MAX) return { ...full, path: bestPath };
    const preview: Record<string, string> = {};
    for (const [k, v] of Object.entries(candidate)) {
      preview[k] = v.length > 500 ? `${v.slice(0, 500)}… [${v.length} chars total]` : v;
    }
    return {
      run_id: runId, score, source, path: bestPath, truncated: true,
      note: `Candidate too large to inline; read the full JSON from ${bestPath} (key "candidate").`,
      preview,
    };
  }

  /** SIGTERM the run's process group (pid file), then SIGKILL if it lingers. */
  async stopRun(runId: string): Promise<{ stopped: boolean; detail: string }> {
    if (!this.exists(runId)) return { stopped: false, detail: 'run not found' };
    const dir = this.runDir(runId);
    if (existsSync(join(dir, 'result.json'))) return { stopped: false, detail: 'run already completed' };
    const alive = this.pidAlive(runId);
    if (!alive) return { stopped: false, detail: alive === null ? 'no pid file (not started by this server version)' : 'process is not running' };
    const pid = parseInt(readText(join(dir, 'pid'))!, 10);
    writeFileSync(join(dir, 'stopped'), String(Date.now()));
    try { process.kill(-pid, 'SIGTERM'); } catch { /* gone */ }
    for (let i = 0; i < 30 && this.pidAlive(runId); i++) await sleep(100);
    if (this.pidAlive(runId)) {
      try { process.kill(-pid, 'SIGKILL'); } catch { /* gone */ }
    }
    return { stopped: true, detail: 'process group terminated; best-so-far is in checkpoint.json (get_best_candidate)' };
  }

  listRuns(limit = 20): { runs: RunStatus[]; total: number; runs_dir: string } {
    const base = runsDir();
    if (!existsSync(base)) return { runs: [], total: 0, runs_dir: base };
    const dirs = readdirSync(base)
      .filter((d) => RUN_ID.test(d))
      .map((d) => { try { return { d, m: statSync(join(base, d)).mtimeMs }; } catch { return null; } })
      .filter((x): x is { d: string; m: number } => x !== null && statSync(join(base, x.d)).isDirectory())
      .sort((a, b) => b.m - a.m);
    const runs = dirs.slice(0, limit).map(({ d }) => {
      const st = deriveStatus(this.snapshot(d));
      delete st.last_events;
      delete st.error;
      return st;
    });
    return { runs, total: dirs.length, runs_dir: base };
  }

  /** Seed-only check: evaluates the seed on ≤3 examples, hard-killed at 50s. */
  async dryRun(runConfig: Record<string, any>): Promise<Record<string, unknown>> {
    const engine = engineDir();
    const uv = uvPath();
    const dir = join(dryRunsDir(), randomUUID().slice(0, 8));
    mkdirSync(dir, { recursive: true });
    const configPath = join(dir, 'config.json');
    writeFileSync(configPath, JSON.stringify({ ...runConfig, config: { ...runConfig.config, run_dir: dir } }, null, 2));
    return await new Promise((resolve) => {
      const child = spawn(uv, ['run', '--project', engine, 'optimize-anything', 'run', '--config', configPath, '--dry-run'], {
        cwd: engine, detached: true, stdio: ['ignore', 'pipe', 'pipe'], env: { ...process.env, PYTHONUNBUFFERED: '1' },
      });
      let stdout = '';
      let stderr = '';
      child.stdout!.on('data', (d) => { stdout += d.toString(); });
      child.stderr!.on('data', (d) => { stderr += d.toString(); });
      const timer = setTimeout(() => {
        try { process.kill(-child.pid!, 'SIGKILL'); } catch { /* gone */ }
      }, 50_000);
      child.on('close', (code, signal) => {
        clearTimeout(timer);
        rmSync(dir, { recursive: true, force: true }); // scratch only; nothing to keep
        const start = stdout.indexOf('{');
        let parsed: any = null;
        if (start >= 0) {
          try { parsed = JSON.parse(stdout.slice(start)); } catch { /* fallthrough */ }
        }
        if (parsed && typeof parsed === 'object') resolve(parsed);
        else resolve({
          dry_run: true, ok: false,
          error: signal ? 'dry run killed after 50s (evaluator too slow for an interactive check)' : `dry run exited with code ${code}`,
          stderr: stderr.slice(-1500),
        });
      });
    });
  }
}
