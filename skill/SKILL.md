---
name: optimize-anything
description: Use when optimizing any text artifact — prompts, code, configs, agent architectures, policies — through iterative LLM-powered search with evaluation feedback. Also use when the user mentions GEPA, optimize, evolve, or improve via search.
---

# optimize_anything

Optimize any text artifact through iterative LLM-powered search with diagnostic feedback.

## Core Concept

If it can be serialized to a string and its quality measured, it can be optimized. The system proposes improvements via Claude reflection on evaluation diagnostics (ASI), tracks a Pareto frontier of non-dominated solutions, and iterates until budget is exhausted.

## When to Use

- "Optimize this prompt" — prompt optimization
- "Make this code faster" — code optimization with benchmarks
- "Find a better agent architecture" — agent evolution
- "Tune this config" — configuration optimization
- Any task where you have a text artifact + a way to score it

## Three Modes

1. **Single-task**: One problem to solve. No dataset needed.
2. **Multi-task**: Batch of related problems. Cross-transfer of insights.
3. **Generalization**: Train + val split. Optimized artifact must transfer to unseen examples.

## How to Formulate an Optimization

Every optimization needs three things:

| Component | Question | Example |
|-----------|----------|---------|
| **Candidate** | What text are we optimizing? | `{"system_prompt": "You are..."}` |
| **Evaluator** | How do we score it? | Python fn, shell command, or LLM judge criteria |
| **Objective** | What are we optimizing for? | "Maximize accuracy while minimizing verbosity" |

## Evaluator Selection Guide

| Evaluator | When to Use | Example |
|-----------|-------------|---------|
| **Python** | Complex scoring logic, API calls, test suites | `def evaluate(c, e): return run_tests(c)` |
| **Shell** | CLI tools, benchmarks, existing test runners | `echo '{{candidate}}' \| python bench.py` |
| **LLM Judge** | Subjective quality, no code needed | "Rate clarity and specificity, SCORE: X/10" |

## Recipe 1: Optimize a System Prompt

```
Call optimize_anything MCP tool with:
  seed_candidate: {"system_prompt": "<current prompt>"}
  evaluator: {type: "llm_judge", criteria: "Rate helpfulness, accuracy, conciseness. SCORE: X/10"}
  objective: "Maximize response quality"
  dataset: [{"input": "example query 1"}, {"input": "example query 2"}, ...]
  config: {max_iterations: 20}
```

## Recipe 2: Optimize Code for Performance

```
Call optimize_anything MCP tool with:
  seed_candidate: {"algorithm": "<current code>"}
  evaluator: {type: "shell", command: "python bench.py {{candidate_file}}", score_pattern: "Score: ([\\d.]+)"}
  objective: "Maximize speed while maintaining correctness"
  config: {max_iterations: 30}
```

## Recipe 3: Discover Agent Architecture

```
Call optimize_anything MCP tool with:
  seed_candidate: {"agent_code": "<initial harness>"}
  evaluator: {type: "python", code: "def evaluate(c, e): ...run agent, return score..."}
  objective: "Maximize task completion accuracy"
  dataset: [<train_tasks>]
  valset: [<held_out_tasks>]
  config: {max_iterations: 50}
```

## Recipe 4: Optimize Config/Policy

```
Call optimize_anything MCP tool with:
  seed_candidate: {"config": "<yaml or json>"}
  evaluator: {type: "shell", command: "deploy_and_test.sh {{candidate_file}}", score_pattern: "metric: ([\\d.]+)"}
  objective: "Minimize cost while maintaining SLA"
  dataset: [<scenarios>]
  config: {max_iterations: 25}
```

## Workflow

1. Identify the artifact to optimize and decompose into components
2. Choose the right evaluator type
3. Define objective and any background context
4. Call `optimize_anything` MCP tool — returns a run_id
5. Poll with `check_optimization` to monitor progress
6. Retrieve results with `get_best_candidate`
7. Apply the optimized artifact

## Key Concepts

- **ASI (Actionable Side Information)**: Diagnostic feedback from evaluator (errors, traces, scores) that helps the proposer make targeted improvements
- **Pareto Frontier**: Tracks non-dominated solutions across multiple objectives/examples
- **Reflective Mutation**: Claude reads ASI, diagnoses issues, proposes targeted fixes — not blind evolution
- **Minibatch Reflection**: Each iteration focuses on a subset of examples, rotating coverage over time
