// mcp-server/src/tools.ts
export const TOOLS = [
  {
    name: 'optimize_anything',
    description: 'Start an optimization run. Optimizes any text artifact through iterative LLM-powered search.',
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
          description: 'Evaluator config: {type: "python"|"shell"|"llm_judge", code?, command?, score_pattern?, criteria?, judge_model?, timeout?}',
          properties: {
            type: { type: 'string', enum: ['python', 'shell', 'llm_judge'] },
            code: { type: 'string' },
            command: { type: 'string' },
            score_pattern: { type: 'string' },
            criteria: { type: 'string' },
            judge_model: { type: 'string' },
            timeout: { type: 'number' },
          },
          required: ['type'],
        },
        objective: { type: 'string', description: 'What to optimize for (natural language)' },
        background: { type: 'string', description: 'Domain knowledge and constraints' },
        dataset: { type: 'array', description: 'Training examples (for multi-task or generalization mode)' },
        valset: { type: 'array', description: 'Validation set (for generalization mode)' },
        config: {
          type: 'object',
          description: 'Engine config: {max_iterations?, max_metric_calls?, model?, selection_strategy?, ...}',
        },
      },
      required: ['evaluator'],
    },
  },
  {
    name: 'check_optimization',
    description: 'Check the current status of an optimization run.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        run_id: { type: 'string', description: 'The run ID returned by optimize_anything' },
      },
      required: ['run_id'],
    },
  },
  {
    name: 'get_best_candidate',
    description: 'Get the current best candidate from an optimization run.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        run_id: { type: 'string', description: 'The run ID' },
      },
      required: ['run_id'],
    },
  },
  {
    name: 'stop_optimization',
    description: 'Stop a running optimization.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        run_id: { type: 'string', description: 'The run ID to stop' },
      },
      required: ['run_id'],
    },
  },
  {
    name: 'list_optimization_runs',
    description: 'List all optimization runs with their status.',
    inputSchema: {
      type: 'object' as const,
      properties: {},
    },
  },
];
