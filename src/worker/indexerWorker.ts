import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';
import { EmbedDtype, EmbedProviderKind, WorkerConfig } from '../types.js';
import { LocalRagEngine } from './rag.js';
import {
  isRelPathEligible,
  MAX_FILE_BYTES,
  normalizeRelPath,
  resolveContainedPath,
  scanEligibleWorkspaceFiles,
} from './workspace.js';

/**
 * Redirect console.log / stray stdout writes to stderr + log file so stdout stays reserved
 * exclusively for MCP JSON-RPC frames.
 */
function installStdoutIsolation(logFilePath: string): (msg: string) => void {
  fs.mkdirSync(path.dirname(logFilePath), { recursive: true });
  const writeLog = (msg: string) => {
    const line = `[${new Date().toISOString()}] ${msg}\n`;
    try {
      fs.appendFileSync(logFilePath, line, 'utf8');
    } catch {
      // Ignore log disk failure.
    }
    process.stderr.write(line);
  };

  console.log = (...args: unknown[]) => writeLog(args.map(String).join(' '));
  console.info = (...args: unknown[]) => writeLog(args.map(String).join(' '));
  console.warn = (...args: unknown[]) => writeLog(`WARN: ${args.map(String).join(' ')}`);
  console.error = (...args: unknown[]) => writeLog(`ERROR: ${args.map(String).join(' ')}`);

  return writeLog;
}

function parseWorkerConfigFromEnv(): WorkerConfig {
  const workspaceRoot = process.env.AGENTOCODE_WORKSPACE_ROOT;
  if (!workspaceRoot) {
    throw new Error('AGENTOCODE_WORKSPACE_ROOT environment variable is required.');
  }
  const stateDir =
    process.env.AGENTOCODE_STATE_DIR || path.join(workspaceRoot, '.guardian');
  const knowledgeDir =
    process.env.AGENTOCODE_KNOWLEDGE_DIR || path.join(stateDir, 'knowledge_base');
  const modelCacheDir =
    process.env.AGENTOCODE_MODEL_CACHE_DIR || path.join(stateDir, 'hf-cache');

  const embedProvider = (process.env.AGENTOCODE_EMBED_PROVIDER || 'transformers_onnx') as EmbedProviderKind;
  const embedDtype = (process.env.AGENTOCODE_EMBED_DTYPE || 'q8') as EmbedDtype;

  return {
    workspaceRoot: fs.realpathSync(workspaceRoot),
    stateDir,
    knowledgeDir,
    modelCacheDir,
    writeKnowledgeDocs: process.env.AGENTOCODE_WRITE_KNOWLEDGE_DOCS !== '0',
    embedProvider,
    embedModelId: process.env.AGENTOCODE_EMBED_MODEL_ID || 'nomic-ai/nomic-embed-text-v1.5',
    embedModelRevision:
      process.env.AGENTOCODE_EMBED_MODEL_REVISION ?? 'e9b6763023c676ca8431644204f50c2b100d9aab',
    embedDtype,
    ollamaEndpoint: process.env.AGENTOCODE_OLLAMA_ENDPOINT || 'http://127.0.0.1:11434',
    localFilesOnly: process.env.GUARDIAN_LOCAL_FILES_ONLY === '1',
    maxTokens: Number(process.env.AGENTOCODE_MAX_TOKENS || '2048'),
    maxDistance: Number(process.env.GUARDIAN_MAX_DISTANCE || '0.65'),
  };
}

export async function createAgentoMcpServer(config: WorkerConfig, onLog: (msg: string) => void) {
  const engine = new LocalRagEngine(config, undefined, onLog);
  await engine.initialize();

  const server = new McpServer({
    name: 'agentocode-local-rag',
    version: '0.1.0',
  });

  server.registerTool(
    'list_files',
    {
      description:
        'List eligible workspace source files (excluding hidden, VCS, dependency, binary, credential, and oversized files).',
      inputSchema: {
        subpath: z.string().optional().describe('Optional workspace-relative directory filter.'),
        max_entries: z.number().int().min(1).max(1000).optional().describe('Maximum paths to return.'),
      },
    },
    async ({ subpath, max_entries }) => {
      const limit = max_entries ?? 200;
      const normSub = subpath ? normalizeRelPath(subpath) : '';
      if (normSub) {
        resolveContainedPath(config.workspaceRoot, normSub);
      }
      const files = await scanEligibleWorkspaceFiles(config.workspaceRoot);
      const filtered = normSub
        ? files.filter((f) => f.relativePath === normSub || f.relativePath.startsWith(`${normSub}/`))
        : files;
      const truncated = filtered.length > limit;
      const listing = filtered.slice(0, limit).map((f) => `${f.relativePath} (${f.sizeBytes} B)`);
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(
              {
                totalEligible: filtered.length,
                truncated,
                files: listing,
              },
              null,
              2,
            ),
          },
        ],
      };
    },
  );

  server.registerTool(
    'read_file',
    {
      description: 'Read a bounded line range from an eligible workspace file safely inside workspace boundaries.',
      inputSchema: {
        path: z.string().describe('Workspace-relative path to read.'),
        start_line: z.number().int().min(1).optional().describe('1-based starting line number.'),
        end_line: z.number().int().min(1).optional().describe('1-based inclusive ending line number.'),
      },
    },
    async ({ path: relPath, start_line, end_line }) => {
      const contained = resolveContainedPath(config.workspaceRoot, relPath);
      if (!isRelPathEligible(contained.relativePath)) {
        throw new Error(`File excluded by workspace policy: ${contained.relativePath}`);
      }
      const stat = await fsp.stat(contained.realPath);
      if (!stat.isFile()) {
        throw new Error(`Target is not a file: ${contained.relativePath}`);
      }
      if (stat.size > MAX_FILE_BYTES) {
        throw new Error(`File exceeds ${MAX_FILE_BYTES} bytes limit: ${contained.relativePath}`);
      }
      const raw = await fsp.readFile(contained.realPath, 'utf8');
      const lines = raw.split(/\r?\n/);
      const start = Math.max(1, start_line ?? 1);
      const end = Math.min(lines.length, end_line ?? Math.min(lines.length, start + 249));
      const slice = lines
        .slice(start - 1, end)
        .map((line, idx) => `${start + idx}: ${line}`)
        .join('\n');

      return {
        content: [
          {
            type: 'text',
            text: `File: ${contained.relativePath} (lines ${start}-${end} of ${lines.length})\n\n${slice}`,
          },
        ],
      };
    },
  );

  server.registerTool(
    'rag_search',
    {
      description:
        'Search the local knowledge index with live-hash staleness rejection and explicit lexical fallback reporting.',
      inputSchema: {
        query: z.string().min(1).describe('Natural language or symbol query.'),
        top_k: z.number().int().min(1).max(20).optional().describe('Maximum hits to return.'),
      },
    },
    async ({ query, top_k }) => {
      const res = await engine.ragSearch(query, top_k ?? 6);
      return {
        content: [{ type: 'text', text: JSON.stringify(res, null, 2) }],
      };
    },
  );

  server.registerTool(
    'read_concept',
    {
      description: 'Read a deterministic OKF v0.2 knowledge concept document by concept_id or workspace-relative path.',
      inputSchema: {
        concept: z.string().min(1).describe('Concept ID (e.g. file:src/index.ts#chunk0) or workspace-relative path.'),
      },
    },
    async ({ concept }) => {
      const text = await engine.readConcept(concept);
      return {
        content: [{ type: 'text', text }],
      };
    },
  );

  server.registerTool(
    'knowledge_status',
    {
      description: 'Inspect local embedding model state, concept/vector counts, token-cap exclusions, and failures.',
      inputSchema: {},
    },
    async () => {
      const status = await engine.getStatus();
      return {
        content: [{ type: 'text', text: JSON.stringify(status, null, 2) }],
      };
    },
  );

  server.registerTool(
    'sync_knowledge',
    {
      description: 'Run local incremental AST/structural indexing and local embedding synchronization (zero Groq calls).',
      inputSchema: {
        force: z.boolean().optional().describe('Recompute all eligible files even when file hashes match.'),
      },
    },
    async ({ force }) => {
      const status = await engine.syncKnowledge(Boolean(force));
      return {
        content: [{ type: 'text', text: JSON.stringify(status, null, 2) }],
      };
    },
  );

  server.registerTool(
    'preload_model',
    {
      description: 'Load and warm the configured local embedding model in memory.',
      inputSchema: {},
    },
    async () => {
      const status = await engine.preloadModel();
      return {
        content: [{ type: 'text', text: JSON.stringify(status, null, 2) }],
      };
    },
  );

  return { server, engine };
}

async function main(): Promise<void> {
  const config = parseWorkerConfigFromEnv();
  const logPath = path.join(config.stateDir, 'mcp-worker.log');
  const onLog = installStdoutIsolation(logPath);

  onLog(`Starting AgentoCode local worker (pid=${process.pid}, provider=${config.embedProvider}).`);
  const { server, engine } = await createAgentoMcpServer(config, onLog);
  const transport = new StdioServerTransport();

  const shutdown = async () => {
    onLog('Shutting down AgentoCode local worker...');
    await server.close().catch(() => undefined);
    await engine.dispose().catch(() => undefined);
    process.exit(0);
  };

  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
  process.stdin.on('end', shutdown);

  await server.connect(transport);
  onLog('AgentoCode local worker connected over MCP stdio.');
}

// Allow direct invocation as child worker while keeping module importable in tests.
if (typeof require !== 'undefined' && require.main === module) {
  main().catch((err) => {
    process.stderr.write(`Fatal worker error: ${err instanceof Error ? err.stack || err.message : String(err)}\n`);
    process.exit(1);
  });
}

