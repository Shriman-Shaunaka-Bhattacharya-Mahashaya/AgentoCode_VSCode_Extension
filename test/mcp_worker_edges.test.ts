import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { McpWorkerClient } from '../src/mcp/workerClient.js';

test('compiled MCP worker enforces workspace containment on read_file/list_files and survives tool errors', async () => {
  const parent = await fsp.mkdtemp(path.join(os.tmpdir(), 'agentocode-mcp-sec-'));
  const ws = path.join(parent, 'project');
  const outside = path.join(parent, 'outside.txt');
  const workerBundle = path.resolve(process.cwd(), 'dist', 'indexerWorker.cjs');

  await fsp.mkdir(path.join(ws, 'src', 'sub'), { recursive: true });
  await fsp.writeFile(outside, 'TOP_SECRET', 'utf8');
  await fsp.writeFile(path.join(ws, '.env'), 'GROQ_API_KEY=nope\n', 'utf8');
  await fsp.writeFile(path.join(ws, 'src', 'a.ts'), 'export const a = 1;\n', 'utf8');
  await fsp.writeFile(path.join(ws, 'src', 'sub', 'b.ts'), 'export const b = 2;\n', 'utf8');
  await fsp.writeFile(path.join(ws, 'root.py'), 'def main():\n    return 0\n', 'utf8');

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
    await assert.rejects(() => client.callToolText('read_file', { path: '../outside.txt' }), /escapes workspace/);
    await assert.rejects(() => client.callToolText('read_file', { path: '.env' }), /excluded/);
    await assert.rejects(() => client.callToolText('read_file', { path: 'missing.ts' }), /does not exist/);

    // Worker must still be healthy after tool errors.
    const listedSub = JSON.parse(
      await client.callToolText('list_files', { subpath: 'src', max_entries: 1 }),
    ) as { totalEligible: number; truncated: boolean; files: string[] };
    assert.equal(listedSub.totalEligible, 2);
    assert.equal(listedSub.truncated, true);
    assert.equal(listedSub.files.length, 1);

    await client.syncKnowledge(false);
    const indexDoc = await client.callToolText('read_concept', { concept: 'workspace:index' });
    assert.ok(indexDoc.includes('src/sub/b.ts'));

    const dirDoc = await client.callToolText('read_concept', { concept: 'dir:.' });
    assert.ok(dirDoc.includes('root.py'));

    // Deleting a subdirectory prunes stale knowledge_base/directories/ entries on next sync.
    await fsp.rm(path.join(ws, 'src', 'sub'), { recursive: true, force: true });
    await client.syncKnowledge(false);
    const staleDirDoc = path.join(ws, '.guardian', 'knowledge_base', 'directories', 'src', 'sub', 'content.md');
    assert.equal(fs.existsSync(staleDirDoc), false);

    // Force sync completes cleanly even with unchanged files.
    const forced = await client.syncKnowledge(true);
    assert.equal(forced.conceptCount, 2);
  } finally {
    await client.dispose();
    await fsp.rm(parent, { recursive: true, force: true });
  }
});

test('compiled MCP worker keeps stdout strictly JSON-RPC even during errors and stderr logging', async () => {
  const ws = await fsp.mkdtemp(path.join(os.tmpdir(), 'agentocode-raw-wire-'));
  const workerBundle = path.resolve(process.cwd(), 'dist', 'indexerWorker.cjs');

  try {
    await fsp.writeFile(path.join(ws, 'main.ts'), 'export const ok = true;\n', 'utf8');

    const child = spawn(process.execPath, [workerBundle], {
      cwd: process.cwd(),
      env: {
        ...process.env,
        AGENTOCODE_WORKSPACE_ROOT: ws,
        AGENTOCODE_STATE_DIR: path.join(ws, '.guardian'),
        AGENTOCODE_KNOWLEDGE_DIR: path.join(ws, '.guardian', 'knowledge_base'),
        AGENTOCODE_EMBED_PROVIDER: 'transformers_onnx',
        AGENTOCODE_EMBED_MODEL_ID: 'missing/offline-model',
        AGENTOCODE_EMBED_MODEL_REVISION: '',
        GUARDIAN_LOCAL_FILES_ONLY: '1',
      },
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    const stdoutLines: string[] = [];
    let stdoutBuf = '';
    const stderrChunks: string[] = [];

    child.stdout.on('data', (chunk: Buffer) => {
      stdoutBuf += chunk.toString('utf8');
      const parts = stdoutBuf.split(/\r?\n/);
      stdoutBuf = parts.pop() ?? '';
      for (const p of parts) {
        if (p.trim()) {
          stdoutLines.push(p);
        }
      }
    });
    child.stderr.on('data', (chunk: Buffer) => {
      stderrChunks.push(chunk.toString('utf8'));
    });

    const sendRpc = (msg: Record<string, unknown>) => {
      child.stdin.write(`${JSON.stringify(msg)}\n`);
    };

    sendRpc({
      jsonrpc: '2.0',
      id: 1,
      method: 'initialize',
      params: {
        protocolVersion: '2024-11-05',
        capabilities: {},
        clientInfo: { name: 'raw-wire-test', version: '1.0.0' },
      },
    });
    sendRpc({ jsonrpc: '2.0', method: 'notifications/initialized' });
    sendRpc({ jsonrpc: '2.0', id: 2, method: 'tools/call', params: { name: 'preload_model', arguments: {} } });
    sendRpc({ jsonrpc: '2.0', id: 3, method: 'tools/call', params: { name: 'knowledge_status', arguments: {} } });

    await new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('Timed out waiting for raw wire RPC responses')), 10000);
      const checkInterval = setInterval(() => {
        if (stdoutLines.length >= 3) {
          clearInterval(checkInterval);
          clearTimeout(timeout);
          resolve();
        }
      }, 25);
    });

    child.stdin.end();
    await new Promise<void>((resolve) => child.once('exit', () => resolve()));

    // Every non-empty line emitted on stdout MUST parse cleanly as JSON-RPC.
    for (const line of stdoutLines) {
      const parsed = JSON.parse(line) as { jsonrpc?: string };
      assert.equal(parsed.jsonrpc, '2.0');
    }
    assert.ok(stderrChunks.join('').includes('Starting AgentoCode local worker'));
  } finally {
    await fsp.rm(ws, { recursive: true, force: true });
  }
});

test('compiled MCP worker flushes all pipelined responses when stdin closes immediately at EOF', async () => {
  const ws = await fsp.mkdtemp(path.join(os.tmpdir(), 'agentocode-pipe-eof-'));
  const workerBundle = path.resolve(process.cwd(), 'dist', 'indexerWorker.cjs');

  try {
    await fsp.writeFile(path.join(ws, 'calc.ts'), 'export function multiply(a: number, b: number) { return a * b; }\n', 'utf8');

    const child = spawn(process.execPath, [workerBundle], {
      cwd: process.cwd(),
      env: {
        ...process.env,
        AGENTOCODE_WORKSPACE_ROOT: ws,
        AGENTOCODE_STATE_DIR: path.join(ws, '.guardian'),
        AGENTOCODE_KNOWLEDGE_DIR: path.join(ws, '.guardian', 'knowledge_base'),
        AGENTOCODE_EMBED_PROVIDER: 'lexical_only',
        GUARDIAN_LOCAL_FILES_ONLY: '1',
      },
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    let stdoutRaw = '';
    child.stdout.on('data', (chunk: Buffer) => {
      stdoutRaw += chunk.toString('utf8');
    });

    const batch =
      [
        JSON.stringify({
          jsonrpc: '2.0',
          id: 1,
          method: 'initialize',
          params: {
            protocolVersion: '2024-11-05',
            capabilities: {},
            clientInfo: { name: 'eof-batch-test', version: '1.0.0' },
          },
        }),
        JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized' }),
        JSON.stringify({ jsonrpc: '2.0', id: 2, method: 'tools/call', params: { name: 'sync_knowledge', arguments: { force: false } } }),
        JSON.stringify({ jsonrpc: '2.0', id: 3, method: 'tools/call', params: { name: 'rag_search', arguments: { query: 'multiply', top_k: 3 } } }),
      ].join('\n') + '\n';

    // Close stdin immediately after writing the batch, matching piped CI/Docker runs.
    child.stdin.end(batch);

    const exitCode = await new Promise<number | null>((resolve, reject) => {
      const t = setTimeout(() => reject(new Error('Timed out waiting for pipelined worker exit')), 10000);
      child.once('exit', (code) => {
        clearTimeout(t);
        resolve(code);
      });
    });

    assert.equal(exitCode, 0);
    const messages = stdoutRaw
      .trim()
      .split(/\r?\n/)
      .filter(Boolean)
      .map((line) => JSON.parse(line) as { id?: number; result?: unknown });

    assert.equal(messages.length, 3);
    const ragMsg = messages.find((m) => m.id === 3);
    assert.ok(ragMsg && JSON.stringify(ragMsg.result).includes('calc.ts'));
  } finally {
    await fsp.rm(ws, { recursive: true, force: true, maxRetries: 10, retryDelay: 50 });
  }
});

