import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { ListToolsRequestSchema, CallToolRequestSchema } from '@modelcontextprotocol/sdk/types.js';
import { TOOLS } from './tools.js';
import { ProcessManager } from './process-manager.js';
import { validateStart } from './guardrails.js';
import { capText, fitList } from './output.js';

const server = new Server({ name: 'optimize-anything-mcp', version: '1.1.0' }, { capabilities: { tools: {} } });
const manager = new ProcessManager();

function ok(obj: unknown) {
  return { content: [{ type: 'text', text: capText(JSON.stringify(obj, null, 2)) }] };
}
function fail(obj: unknown) {
  return { content: [{ type: 'text', text: capText(JSON.stringify(obj, null, 2)) }], isError: true };
}

server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: TOOLS }));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name } = request.params;
  const args = (request.params.arguments ?? {}) as Record<string, any>;
  try {
    switch (name) {
      case 'optimize_anything': {
        const v = validateStart(args);
        if (v.errors.length) {
          return fail({ error: 'invalid optimize_anything arguments; nothing was started', fix: v.errors, warnings: v.warnings });
        }
        if (args.dry_run === true) {
          const res = await manager.dryRun(v.runConfig);
          return ok({ ...res, warnings: [...v.warnings, ...((res.warnings as string[]) ?? [])], notes: v.notes });
        }
        const { runId, runDir } = manager.startRun(v.runConfig);
        return ok({
          run_id: runId,
          status: 'started',
          run_dir: runDir,
          budget: Object.fromEntries(
            ['max_metric_calls', 'max_iterations', 'max_cost_usd'].filter((k) => k in v.runConfig.config).map((k) => [k, v.runConfig.config[k]]),
          ),
          warnings: v.warnings,
          notes: v.notes,
          next: 'call check_optimization with run_id and wait_seconds=45 until status is not "running"',
        });
      }

      case 'check_optimization': {
        const runId = String(args.run_id ?? '');
        const wait = typeof args.wait_seconds === 'number' ? args.wait_seconds : 0;
        const status = await manager.waitStatus(runId, wait);
        if (!status) return fail({ error: `run ${runId} not found`, fix: 'use list_optimization_runs to see run ids' });
        return ok(status);
      }

      case 'get_best_candidate': {
        const best = manager.getBestCandidate(String(args.run_id ?? ''));
        if (!best) return fail({ error: 'no candidate yet', fix: 'the seed checkpoint appears after the seed is evaluated; check_optimization shows progress' });
        return ok(best);
      }

      case 'stop_optimization': {
        const runId = String(args.run_id ?? '');
        return ok({ run_id: runId, ...(await manager.stopRun(runId)) });
      }

      case 'list_optimization_runs': {
        const limit = typeof args.limit === 'number' && args.limit > 0 ? Math.min(args.limit, 100) : 20;
        return ok(fitList(manager.listRuns(limit), 'runs'));
      }

      default:
        return fail({ error: `Unknown tool: ${name}` });
    }
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    return fail({ error: msg });
  }
});

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error('optimize-anything MCP server running on stdio');
}

main().catch((error) => {
  console.error('Fatal:', error);
  process.exit(1);
});
