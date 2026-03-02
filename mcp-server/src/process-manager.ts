// mcp-server/src/process-manager.ts
import { spawn, ChildProcess } from 'child_process';
import { readFileSync, existsSync, writeFileSync, mkdirSync, readdirSync } from 'fs';
import { join, dirname } from 'path';
import { homedir } from 'os';
import { randomUUID } from 'crypto';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export interface RunConfig {
  seed_candidate?: Record<string, string>;
  evaluator: {
    type: 'python' | 'shell' | 'llm_judge';
    code?: string;
    command?: string;
    score_pattern?: string;
    criteria?: string;
    judge_model?: string;
    timeout?: number;
  };
  objective?: string;
  background?: string;
  dataset?: any[];
  valset?: any[];
  config?: Record<string, any>;
}

export interface RunStatus {
  run_id: string;
  status: 'running' | 'completed' | 'failed' | 'stopped';
  iteration?: number;
  best_score?: number;
  num_candidates?: number;
  total_evals?: number;
  result?: any;
  error?: string;
}

const RUNS_DIR = join(homedir(), '.optimize-anything', 'runs');
const ENGINE_DIR = join(__dirname, '..', '..', 'engine');

export class ProcessManager {
  private processes: Map<string, ChildProcess> = new Map();
  private statuses: Map<string, RunStatus> = new Map();

  async startRun(config: RunConfig): Promise<string> {
    const runId = randomUUID().slice(0, 8);
    const runDir = join(RUNS_DIR, runId);
    mkdirSync(runDir, { recursive: true });

    const configPath = join(runDir, 'config.json');
    const fullConfig = {
      ...config,
      config: { ...config.config, run_dir: runDir },
    };
    writeFileSync(configPath, JSON.stringify(fullConfig, null, 2));

    const proc = spawn('uv', ['run', 'optimize-anything', 'run', '--config', configPath, '--events'], {
      cwd: ENGINE_DIR,
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    this.processes.set(runId, proc);
    this.statuses.set(runId, { run_id: runId, status: 'running' });

    let lastEvent: any = null;

    proc.stdout?.on('data', (data: Buffer) => {
      const lines = data.toString().split('\n').filter(Boolean);
      for (const line of lines) {
        try {
          lastEvent = JSON.parse(line);
          if (lastEvent.type === 'optimization_complete') {
            this.statuses.set(runId, {
              run_id: runId,
              status: 'completed',
              best_score: lastEvent.best_score,
              total_evals: lastEvent.total_evals,
              iteration: lastEvent.total_iterations,
            });
          }
        } catch {}
      }
    });

    proc.stderr?.on('data', (data: Buffer) => {
      console.error(`[${runId}] ${data.toString()}`);
    });

    proc.on('close', (code) => {
      this.processes.delete(runId);
      const current = this.statuses.get(runId);
      if (current && current.status === 'running') {
        this.statuses.set(runId, {
          ...current,
          status: code === 0 ? 'completed' : 'failed',
          error: code !== 0 ? `Process exited with code ${code}` : undefined,
        });
      }
    });

    return runId;
  }

  getStatus(runId: string): RunStatus | null {
    // Check in-memory first
    const status = this.statuses.get(runId);
    if (status) return status;

    // Check checkpoint on disk
    const checkpointPath = join(RUNS_DIR, runId, 'checkpoint.json');
    if (existsSync(checkpointPath)) {
      try {
        const data = JSON.parse(readFileSync(checkpointPath, 'utf-8'));
        return {
          run_id: runId,
          status: 'completed',
          iteration: data.iteration,
          best_score: Math.max(...(data.agg_scores || [0])),
          num_candidates: data.candidates?.length || 0,
          total_evals: data.total_evals,
        };
      } catch {}
    }

    return null;
  }

  getBestCandidate(runId: string): { candidate: Record<string, string>; score: number } | null {
    const checkpointPath = join(RUNS_DIR, runId, 'checkpoint.json');
    if (!existsSync(checkpointPath)) return null;

    try {
      const data = JSON.parse(readFileSync(checkpointPath, 'utf-8'));
      const scores: number[] = data.agg_scores || [];
      const bestIdx = scores.indexOf(Math.max(...scores));
      return {
        candidate: data.candidates[bestIdx]?.components || {},
        score: scores[bestIdx] || 0,
      };
    } catch {
      return null;
    }
  }

  stopRun(runId: string): boolean {
    const proc = this.processes.get(runId);
    if (proc) {
      proc.kill('SIGTERM');
      this.processes.delete(runId);
      const current = this.statuses.get(runId);
      if (current) {
        this.statuses.set(runId, { ...current, status: 'stopped' });
      }
      return true;
    }
    return false;
  }

  listRuns(): RunStatus[] {
    const runs: RunStatus[] = [];
    for (const status of this.statuses.values()) {
      runs.push(status);
    }

    // Also check disk
    if (existsSync(RUNS_DIR)) {
      for (const dir of readdirSync(RUNS_DIR)) {
        if (!this.statuses.has(dir)) {
          const status = this.getStatus(dir);
          if (status) runs.push(status);
        }
      }
    }

    return runs;
  }
}
