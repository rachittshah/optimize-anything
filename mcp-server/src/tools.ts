export const TOOLS = [
  {
    name: 'optimize_anything',
    description:
      'Start an optimization run of a text artifact (prompt, code, config...) by iterative LLM-powered search ' +
      'against your evaluator. Returns a run_id immediately; then call check_optimization with wait_seconds=45 ' +
      'to follow progress. Set dry_run=true first to score the seed on up to 3 examples (≤45s) and catch ' +
      'evaluator errors before spending budget. A budget is required (defaults: max_metric_calls=150, max_cost_usd=5).',
    inputSchema: {
      type: 'object' as const,
      properties: {
        seed_candidate: {
          type: 'object',
          description: 'Initial candidate as {component_name: text}. E.g. {"system_prompt": "You are..."}',
          additionalProperties: { type: 'string' },
        },
        evaluator: {
          type: 'object',
          description:
            'python: {type:"python", code:"def evaluate(candidate, example): return score, {diagnostics}"}. ' +
            'shell: {type:"shell", command:"python score.py {{candidate_file}}", score_pattern?} (never inline {{candidate}}). ' +
            'llm_judge: {type:"llm_judge", criteria:"..." | rubric:[...], judge_model?}. Higher score = better; ' +
            'return diagnostics (errors, expected vs got) so the proposer can fix failures.',
          properties: {
            type: { type: 'string', enum: ['python', 'shell', 'llm_judge'] },
            code: { type: 'string' },
            command: { type: 'string' },
            score_pattern: { type: 'string' },
            criteria: { type: 'string' },
            rubric: { type: 'array' },
            judge_model: { type: 'string' },
            timeout: { type: 'number', description: 'Seconds per evaluation (default 30)' },
          },
          required: ['type'],
        },
        objective: { type: 'string', description: 'What to optimize for (natural language)' },
        background: { type: 'string', description: 'Domain knowledge and hard constraints' },
        dataset: { type: 'array', description: 'Training examples (multi-task / generalization mode)' },
        valset: { type: 'array', description: 'Held-out validation examples used to pick the best candidate (generalization mode)' },
        testset: { type: 'array', description: 'Held-out test examples, scored only at the end (never used for selection)' },
        preset: { type: 'string', enum: ['default', 'cheap', 'quality'], description: 'Model preset' },
        dry_run: { type: 'boolean', description: 'Only evaluate the seed on ≤3 examples (≤45s) and report scores/errors; no optimization' },
        config: {
          type: 'object',
          description:
            'Engine config. Budget: max_metric_calls, max_iterations, max_cost_usd (at least one; else defaults apply). ' +
            'Others: backend (auto|anthropic|openai|claude-cli|codex-cli|mock), proposer_model, judge_model, ' +
            'reflection_minibatch_size, promotion, val_eval, timeout_seconds, no_improvement_patience...',
          properties: {
            max_metric_calls: { type: 'number' },
            max_iterations: { type: 'number' },
            max_cost_usd: { type: 'number' },
            backend: { type: 'string' },
          },
        },
      },
      required: ['evaluator'],
    },
  },
  {
    name: 'check_optimization',
    description:
      'Status of a run: status (running|completed|failed|stopped), iteration, best_score, seed_score, total_evals, ' +
      'cost_usd, stop_reason, elapsed_s, last 3 events. With wait_seconds>0 it waits (max 50) and returns early ' +
      'on completion or a new best score.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        run_id: { type: 'string', description: 'The run ID returned by optimize_anything' },
        wait_seconds: { type: 'number', description: 'Long-poll up to this many seconds (0-50, default 0)' },
      },
      required: ['run_id'],
    },
  },
  {
    name: 'get_best_candidate',
    description:
      'Best candidate so far (final once the run completed). Always written to <run_dir>/best.json; returned ' +
      'inline when small, else the path plus a preview.',
    inputSchema: {
      type: 'object' as const,
      properties: { run_id: { type: 'string', description: 'The run ID' } },
      required: ['run_id'],
    },
  },
  {
    name: 'stop_optimization',
    description: 'Stop a running optimization (kills its process group). The best-so-far stays available via get_best_candidate.',
    inputSchema: {
      type: 'object' as const,
      properties: { run_id: { type: 'string', description: 'The run ID to stop' } },
      required: ['run_id'],
    },
  },
  {
    name: 'list_optimization_runs',
    description: 'List recent optimization runs (newest first) with status and best score.',
    inputSchema: {
      type: 'object' as const,
      properties: { limit: { type: 'number', description: 'Max runs to return (default 20)' } },
    },
  },
];
