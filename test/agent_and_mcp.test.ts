import assert from 'node:assert/strict';
import fsp from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { AgentLoop, AgentToolExecutor, buildAgentToolDefinitions } from '../src/agent/agentLoop.js';
import { ChatCompletionClient, parseSseDataLine } from '../src/agent/groqClient.js';
import { readAgentoConfig, resolveStorageDirectories } from '../src/config/settings.js';
import { McpWorkerClient } from '../src/mcp/workerClient.js';
import { EditProposal } from '../src/types.js';

test('buildAgentToolDefinitions blocks raw write_file and delete_file while adding IDE diff tools', () => {
  const defs = buildAgentToolDefinitions([
    { name: 'rag_search', description: 'search' },
    { name: 'write_file', description: 'unsafe direct write' },
    { name: 'delete_file', description: 'unsafe direct delete' },
  ]);

  const names = defs.map((d) => d.function.name);
  assert.ok(names.includes('rag_search'));
  assert.ok(!names.includes('write_file'));
  assert.ok(!names.includes('delete_file'));
  assert.ok(names.includes('propose_file_edit'));
  assert.ok(names.includes('ide_get_active_context'));
  assert.ok(names.includes('ide_get_diagnostics'));
});

test('AgentLoop orchestrates MCP search -> propose_file_edit -> final answer and rejects rogue write_file', async () => {
  let step = 0;
  const mockClient: ChatCompletionClient = {
    async streamChatCompletion({ onDelta }) {
      step++;
      if (step === 1) {
        return {
          content: '',
          finishReason: 'tool_calls',
          toolCalls: [
            {
              id: 'call_1',
              type: 'function',
              function: { name: 'rag_search', arguments: JSON.stringify({ query: 'calc' }) },
            },
            {
              id: 'call_rogue',
              type: 'function',
              function: { name: 'write_file', arguments: JSON.stringify({ path: 'calc.py', content: 'x' }) },
            },
          ],
        };
      }
      if (step === 2) {
        return {
          content: '',
          finishReason: 'tool_calls',
          toolCalls: [
            {
              id: 'call_2',
              type: 'function',
              function: {
                name: 'propose_file_edit',
                arguments: JSON.stringify({
                  path: 'calc.py',
                  content: 'def abs_val(x):\n    return abs(x)\n',
                  rationale: 'Add abs_val helper',
                }),
              },
            },
          ],
        };
      }
      onDelta?.('Created a pending diff proposal for calc.py.');
      return {
        content: 'Created a pending diff proposal for calc.py.',
        finishReason: 'stop',
        toolCalls: [],
      };
    },
  };

  const proposals: EditProposal[] = [];
  const toolOutcomes: Array<{ name: string; status: string }> = [];

  const executor: AgentToolExecutor = {
    async listMcpTools() {
      return [{ name: 'rag_search' }, { name: 'write_file' }];
    },
    async callMcpTool(name) {
      return `mock result for ${name}`;
    },
    async getActiveContext() {
      return 'Active File: calc.py';
    },
    async getDiagnostics() {
      return 'No errors.';
    },
    async proposeFileEdit(args) {
      const proposal: EditProposal = {
        id: 'prop-1',
        relativePath: args.path,
        uriString: `file:///${args.path}`,
        originalContent: '',
        proposedContent: args.content,
        isNewFile: false,
        rationale: args.rationale,
        status: 'pending',
        createdAt: new Date().toISOString(),
      };
      proposals.push(proposal);
      return { summary: 'Proposal created', proposal };
    },
  };

  const loop = new AgentLoop(mockClient, executor);
  const res = await loop.runTurn({
    apiKey: 'gsk_test',
    model: 'openai/gpt-oss-120b',
    history: [],
    userPrompt: 'Add abs_val',
    maxIterations: 5,
    callbacks: {
      onDelta: () => undefined,
      onToolStart: () => undefined,
      onToolDone: (t) => toolOutcomes.push({ name: t.name, status: t.status }),
      onProposalCreated: () => undefined,
    },
  });

  assert.equal(res.assistantText, 'Created a pending diff proposal for calc.py.');
  assert.equal(proposals.length, 1);
  assert.deepEqual(toolOutcomes, [
    { name: 'rag_search', status: 'done' },
    { name: 'write_file', status: 'error' },
    { name: 'propose_file_edit', status: 'done' },
  ]);
});

test('SSE parser and settings storage resolution work accurately', () => {
  const chunk = parseSseDataLine('data: {"choices":[{"delta":{"content":"hello"}}]}');
  assert.ok(chunk && chunk !== 'DONE');
  assert.equal(chunk.choices?.[0]?.delta?.content, 'hello');
  assert.equal(parseSseDataLine('data: [DONE]'), 'DONE');
  assert.equal(parseSseDataLine(': keep-alive'), null);

  const cfg = readAgentoConfig({
    get<T>(_k: string, def: T): T {
      return def;
    },
  });
  const dirs = resolveStorageDirectories({
    workspaceRoot: '/repo',
    extensionStoragePath: '/ext-storage/ws1',
    config: cfg,
  });
  assert.equal(dirs.stateDir, path.join('/ext-storage/ws1', 'guardian'));
  assert.equal(dirs.knowledgeDir, path.join('/ext-storage/ws1', 'guardian', 'knowledge_base'));
});

test('compiled MCP worker runs over real JSON-RPC stdio in lexical_only mode without stdout leaks', async () => {
  const ws = await fsp.mkdtemp(path.join(os.tmpdir(), 'agentocode-mcp-e2e-'));
  const workerBundle = path.resolve(process.cwd(), 'dist', 'indexerWorker.cjs');

  try {
    await fsp.writeFile(path.join(ws, 'sample.py'), 'def greet(name):\n    return f"hi {name}"\n', 'utf8');

    const client = new McpWorkerClient(() => ({
      command: process.execPath,
      args: [workerBundle],
      cwd: process.cwd(),
      env: {
        AGENTOCODE_WORKSPACE_ROOT: ws,
        AGENTOCODE_STATE_DIR: path.join(ws, '.guardian'),
        AGENTOCODE_KNOWLEDGE_DIR: path.join(ws, '.guardian', 'knowledge_base'),
        AGENTOCODE_WRITE_KNOWLEDGE_DOCS: '1',
        AGENTOCODE_EMBED_PROVIDER: 'lexical_only',
        GUARDIAN_LOCAL_FILES_ONLY: '1',
      },
    }));

    try {
      const tools = await client.listTools();
      const toolNames = tools.map((t) => t.name);
      assert.ok(toolNames.includes('list_files'));
      assert.ok(toolNames.includes('read_file'));
      assert.ok(toolNames.includes('rag_search'));
      assert.ok(toolNames.includes('read_concept'));
      assert.ok(toolNames.includes('knowledge_status'));
      assert.ok(toolNames.includes('sync_knowledge'));
      assert.ok(!toolNames.includes('write_file'), 'IDE worker must not expose raw write_file');

      const syncStatus = await client.syncKnowledge(false);
      assert.equal(syncStatus.conceptCount, 1);
      assert.equal(syncStatus.embedProvider, 'lexical_only');

      const search = await client.ragSearch('greet', 3);
      assert.equal(search.retrievalMode, 'lexical_fallback');
      assert.ok(search.results.some((r) => r.path === 'sample.py'));

      const readOut = await client.callToolText('read_file', { path: 'sample.py', start_line: 1, end_line: 5 });
      assert.ok(readOut.includes('def greet(name):'));

      const conceptOut = await client.callToolText('read_concept', { concept: 'sample.py' });
      assert.ok(conceptOut.includes('okf_version: "0.2"'));
    } finally {
      await client.dispose();
    }
  } finally {
    await fsp.rm(ws, { recursive: true, force: true });
  }
});

test('compiled MCP worker loads @huggingface/transformers cleanly under localFilesOnly without CJS/ESM crash', async () => {
  const ws = await fsp.mkdtemp(path.join(os.tmpdir(), 'agentocode-hf-import-'));
  const workerBundle = path.resolve(process.cwd(), 'dist', 'indexerWorker.cjs');

  try {
    const client = new McpWorkerClient(() => ({
      command: process.execPath,
      args: [workerBundle],
      cwd: process.cwd(),
      env: {
        AGENTOCODE_WORKSPACE_ROOT: ws,
        AGENTOCODE_STATE_DIR: path.join(ws, '.guardian'),
        AGENTOCODE_KNOWLEDGE_DIR: path.join(ws, '.guardian', 'knowledge_base'),
        AGENTOCODE_EMBED_PROVIDER: 'transformers_onnx',
        AGENTOCODE_EMBED_MODEL_ID: 'nonexistent-org/missing-local-model',
        AGENTOCODE_EMBED_MODEL_REVISION: '',
        GUARDIAN_LOCAL_FILES_ONLY: '1',
      },
    }));

    try {
      await assert.rejects(
        () => client.preloadModel(),
        (err: Error) => !err.message.includes('ERR_REQUIRE_ESM') && !err.message.includes('Cannot find module'),
      );
      const status = await client.getStatus();
      assert.equal(status.modelState, 'error');
      assert.ok(status.lastError && status.lastError.length > 0);
    } finally {
      await client.dispose();
    }
  } finally {
    await fsp.rm(ws, { recursive: true, force: true });
  }
});
