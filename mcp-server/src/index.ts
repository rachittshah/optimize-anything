// mcp-server/src/index.ts
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { ListToolsRequestSchema, CallToolRequestSchema } from '@modelcontextprotocol/sdk/types.js';
import { TOOLS } from './tools.js';
import { ProcessManager, RunConfig } from './process-manager.js';

const server = new Server(
  { name: 'optimize-anything-mcp', version: '1.0.0' },
  { capabilities: { tools: {} } }
);

const manager = new ProcessManager();

server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: TOOLS }));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  try {
    const { name, arguments: args } = request.params;

    switch (name) {
      case 'optimize_anything': {
        const config: RunConfig = {
          seed_candidate: (args as any).seed_candidate,
          evaluator: (args as any).evaluator,
          objective: (args as any).objective,
          background: (args as any).background,
          dataset: (args as any).dataset,
          valset: (args as any).valset,
          config: (args as any).config,
        };
        const runId = await manager.startRun(config);
        return {
          content: [{ type: 'text', text: JSON.stringify({ run_id: runId, status: 'started' }, null, 2) }],
        };
      }

      case 'check_optimization': {
        const status = manager.getStatus((args as any).run_id);
        if (!status) {
          return { content: [{ type: 'text', text: JSON.stringify({ error: 'Run not found' }) }], isError: true };
        }
        return { content: [{ type: 'text', text: JSON.stringify(status, null, 2) }] };
      }

      case 'get_best_candidate': {
        const best = manager.getBestCandidate((args as any).run_id);
        if (!best) {
          return { content: [{ type: 'text', text: JSON.stringify({ error: 'No results found' }) }], isError: true };
        }
        return { content: [{ type: 'text', text: JSON.stringify(best, null, 2) }] };
      }

      case 'stop_optimization': {
        const stopped = manager.stopRun((args as any).run_id);
        return {
          content: [{ type: 'text', text: JSON.stringify({ stopped, run_id: (args as any).run_id }) }],
        };
      }

      case 'list_optimization_runs': {
        const runs = manager.listRuns();
        return { content: [{ type: 'text', text: JSON.stringify(runs, null, 2) }] };
      }

      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    return { content: [{ type: 'text', text: `Error: ${msg}` }], isError: true };
  }
});

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error('optimize-anything MCP server running on stdio');
}

main().catch((error) => { console.error('Fatal:', error); process.exit(1); });
