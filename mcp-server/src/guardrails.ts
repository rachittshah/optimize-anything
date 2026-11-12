// Input validation for the optimize_anything tool. Pure: returns errors/warnings, never throws.

export const DEFAULT_BUDGET = { max_metric_calls: 150, max_cost_usd: 5 };
export const PRESETS = ['default', 'cheap', 'quality'];

export interface StartArgs {
  seed_candidate?: unknown;
  evaluator?: any;
  objective?: unknown;
  background?: unknown;
  dataset?: unknown;
  valset?: unknown;
  testset?: unknown;
  preset?: unknown;
  config?: any;
  dry_run?: unknown;
}

export interface Validated {
  errors: string[];
  warnings: string[];
  notes: string[];
  /** Engine config file contents (only meaningful when errors is empty). */
  runConfig: Record<string, any>;
}

const GENERALIZATION =
  /generali[sz]|unseen|held[- ]?out|new (inputs|examples|cases|data)|robust|transfer|out[- ]of[- ]distribution|any (input|example)/i;

export function validateStart(args: StartArgs): Validated {
  const errors: string[] = [];
  const warnings: string[] = [];
  const notes: string[] = [];
  const ev = args.evaluator;

  // Evaluator
  if (!ev || typeof ev !== 'object') {
    errors.push('evaluator is required: {type: "python"|"shell"|"llm_judge", ...}.');
  } else if (ev.type === 'python') {
    if (typeof ev.code !== 'string' || !/def\s+evaluate\s*\(/.test(ev.code)) {
      errors.push(
        'python evaluator needs `code` defining `def evaluate(candidate, example):` that returns ' +
          '(score, asi_dict), e.g. "def evaluate(candidate, example):\\n    return float(...), {\\"error\\": ...}". ' +
          'candidate is the text (a dict when there are several components); example is a dataset item or None.',
      );
    }
  } else if (ev.type === 'shell') {
    const cmd = ev.command;
    if (typeof cmd !== 'string' || !cmd.trim()) {
      errors.push('shell evaluator needs `command`, e.g. "python score.py {{candidate_file}}".');
    } else {
      if (cmd.includes('{{candidate}}')) {
        errors.push(
          'shell `command` uses inline {{candidate}}: the candidate text is pasted raw into the shell, so any ' +
            'quote, $ or newline in it breaks the command or runs as shell code. Use {{candidate_file}} (a temp ' +
            'file path) and read it in your script, e.g. "python score.py {{candidate_file}}" or ' +
            '"wc -w < {{candidate_file}}". Examples are available as {{example_file}} or {{example_json}}.',
        );
      } else if (!cmd.includes('{{candidate_file}}')) {
        errors.push(
          'shell `command` never references the candidate. Add {{candidate_file}} (path to a temp file holding ' +
            'the candidate), e.g. "python score.py {{candidate_file}}".',
        );
      }
      if (ev.score_pattern !== undefined) {
        try {
          new RegExp(ev.score_pattern);
        } catch {
          errors.push(`shell score_pattern is not a valid regex: ${ev.score_pattern}`);
        }
      } else {
        notes.push('shell score_pattern not set: the engine takes the first number on stdout (regex ([\\d.]+)).');
      }
    }
  } else if (ev.type === 'llm_judge') {
    const hasCriteria = typeof ev.criteria === 'string' && ev.criteria.trim().length > 0;
    const hasRubric = Array.isArray(ev.rubric) && ev.rubric.length > 0;
    if (!hasCriteria && !hasRubric) {
      errors.push(
        'llm_judge evaluator needs `criteria` (concrete, checkable text) or `rubric` (list of criterion names or ' +
          '{name, description, anchors}).',
      );
    }
  } else {
    errors.push(`evaluator.type must be "python", "shell" or "llm_judge" (got ${JSON.stringify(ev.type)}).`);
  }

  // Seed / objective
  const seed = args.seed_candidate;
  if (seed === undefined || seed === null) {
    if (typeof args.objective !== 'string' || !args.objective.trim()) {
      errors.push('Provide seed_candidate ({component_name: text}) or an objective to start from an empty artifact.');
    }
  } else if (typeof seed !== 'object' || Array.isArray(seed) || Object.keys(seed as object).length === 0) {
    errors.push('seed_candidate must be a non-empty object {component_name: text}, e.g. {"system_prompt": "You are..."}.');
  } else {
    for (const [k, v] of Object.entries(seed as Record<string, unknown>)) {
      if (typeof v !== 'string') errors.push(`seed_candidate.${k} must be a string (got ${typeof v}).`);
    }
  }

  for (const name of ['dataset', 'valset', 'testset'] as const) {
    const v = args[name];
    if (v !== undefined && v !== null && !Array.isArray(v)) errors.push(`${name} must be an array of examples.`);
    if (Array.isArray(v) && v.length === 0) warnings.push(`${name} is empty; it is ignored.`);
  }

  const config: Record<string, any> =
    args.config && typeof args.config === 'object' && !Array.isArray(args.config) ? { ...args.config } : {};
  if (args.config !== undefined && (typeof args.config !== 'object' || Array.isArray(args.config))) {
    errors.push('config must be an object, e.g. {"max_metric_calls": 150, "max_cost_usd": 5}.');
  }

  // Preset
  if (args.preset !== undefined) {
    if (typeof args.preset !== 'string' || !PRESETS.includes(args.preset)) {
      errors.push(`preset must be one of ${PRESETS.join(', ')}.`);
    } else config.preset = args.preset;
  }

  // Budget
  for (const k of ['max_metric_calls', 'max_iterations', 'max_cost_usd']) {
    if (config[k] !== undefined && !(typeof config[k] === 'number' && config[k] > 0)) {
      errors.push(`config.${k} must be a positive number.`);
    }
  }
  const hasBudget = ['max_metric_calls', 'max_iterations', 'max_cost_usd'].some((k) => typeof config[k] === 'number');
  if (!hasBudget) {
    Object.assign(config, DEFAULT_BUDGET);
    notes.push(
      `No budget given: using max_metric_calls=${DEFAULT_BUDGET.max_metric_calls} and max_cost_usd=${DEFAULT_BUDGET.max_cost_usd}. ` +
        'Set config.max_metric_calls / max_iterations / max_cost_usd to change it.',
    );
  }

  // Formulation warnings
  const hasDataset = Array.isArray(args.dataset) && args.dataset.length > 0;
  const hasValset = Array.isArray(args.valset) && args.valset.length > 0;
  const objective = typeof args.objective === 'string' ? args.objective : '';
  if (hasDataset && !hasValset && GENERALIZATION.test(objective + ' ' + (args.background ?? ''))) {
    warnings.push(
      'The objective sounds like generalization, but no valset was given: candidates are then selected on the ' +
        'same examples they were tuned on, which overstates quality. Pass a held-out valset (and optionally a testset).',
    );
  }
  if (hasValset && !hasDataset) {
    warnings.push('valset without dataset: pass training examples in dataset too.');
  }
  if (hasValset && (args.valset as unknown[]).length < 10) {
    warnings.push(`valset has only ${(args.valset as unknown[]).length} examples; scores will be noisy (aim for 10+).`);
  }

  const runConfig: Record<string, any> = {
    seed_candidate: seed ?? undefined,
    evaluator: ev,
    objective: args.objective,
    background: args.background,
    dataset: args.dataset,
    valset: args.valset,
    testset: args.testset,
    config,
  };
  for (const k of Object.keys(runConfig)) if (runConfig[k] === undefined || runConfig[k] === null) delete runConfig[k];
  return { errors, warnings, notes, runConfig };
}
