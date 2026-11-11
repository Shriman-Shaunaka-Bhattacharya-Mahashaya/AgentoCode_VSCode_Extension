import assert from 'node:assert/strict';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { AgentLoop, AgentToolExecutor } from '../src/agent/agentLoop.js';
import { ChatCompletionClient, GroqHttpClient } from '../src/agent/groqClient.js';
import {
  buildWorkerEnvironment,
  DEFAULT_CONFIG,
  resolveNodeRuntime,
  resolveStorageDirectories,
} from '../src/config/settings.js';
import { parseDotEnv, resolveGroqApiKey } from '../src/config/envLoader.js';

test('GroqHttpClient parses chunked SSE text deltas and split tool call arguments', async () => {
  const server = http.createServer(async (req, res) => {
    const chunks: Buffer[] = [];
    for await (const c of req) {
      chunks.push(c as Buffer);
    }
    const payload = JSON.parse(Buffer.concat(chunks).toString('utf8')) as { model: string; stream: boolean };
    assert.equal(payload.model, 'openai/gpt-oss-120b');
    assert.equal(payload.stream, true);
    assert.equal(req.headers.authorization, 'Bearer gsk_local_test');

    res.writeHead(200, { 'Content-Type': 'text/event-stream' });
    res.write('data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n');
    res.write('data: {"choices":[{"delta":{"content":"lo!"}}]}\n\n');
    res.write(
      'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"read_file","arguments":"{\\"path\\":"}}]}}]}\n\n',
    );
    res.write(
      'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"a.ts\\"}"}}]},"finish_reason":"tool_calls"}]}\n\n',
    );
    res.end('data: [DONE]\n\n');
  });

  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const addr = server.address() as { port: number };
  const client = new GroqHttpClient(`http://127.0.0.1:${addr.port}`);

  try {
    const deltas: string[] = [];
    const result = await client.streamChatCompletion({
      apiKey: 'gsk_local_test',
      model: 'openai/gpt-oss-120b',
      messages: [{ role: 'user', content: 'hi' }],
      tools: [],
      onDelta: (d) => deltas.push(d),
    });

    assert.equal(deltas.join(''), 'Hello!');
    assert.equal(result.content, 'Hello!');
    assert.equal(result.finishReason, 'tool_calls');
    assert.equal(result.toolCalls.length, 1);
    assert.equal(result.toolCalls[0].id, 'call_abc');
    assert.equal(result.toolCalls[0].function.name, 'read_file');
    assert.equal(result.toolCalls[0].function.arguments, '{"path":"a.ts"}');
  } finally {
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
});

test('GroqHttpClient surfaces structured HTTP errors and supports AbortSignal cancellation', async () => {
  const server = http.createServer((_req, res) => {
    res.writeHead(401, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { message: 'Invalid API Key' } }));
  });

  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const addr = server.address() as { port: number };
  const client = new GroqHttpClient(`http://127.0.0.1:${addr.port}`);

  try {
    await assert.rejects(
      () =>
        client.streamChatCompletion({
          apiKey: 'bad',
          model: 'openai/gpt-oss-120b',
          messages: [{ role: 'user', content: 'hi' }],
          tools: [],
        }),
      /Groq API \(401\): Invalid API Key/,
    );

    const ac = new AbortController();
    ac.abort();
    await assert.rejects(
      () =>
        client.streamChatCompletion({
          apiKey: 'bad',
          model: 'openai/gpt-oss-120b',
          messages: [{ role: 'user', content: 'hi' }],
          tools: [],
          signal: ac.signal,
        }),
    );
  } finally {
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
});

test('AgentLoop handles malformed JSON tool args, iteration caps, native IDE tools, and abort signals', async () => {
  let calls = 0;
  const runawayClient: ChatCompletionClient = {
    async streamChatCompletion() {
      calls++;
      return {
        content: '',
        finishReason: 'tool_calls',
        toolCalls: [
          {
            id: `c_${calls}`,
            type: 'function',
            function: {
              name: calls === 1 ? 'ide_get_active_context' : 'ide_get_diagnostics',
              arguments: calls === 2 ? '{bad json' : '{"path":"src/app.ts"}',
            },
          },
        ],
      };
    },
  };

  const events: Array<{ name: string; status: string; preview?: string }> = [];
  const executor: AgentToolExecutor = {
    async listMcpTools() {
      return [];
    },
    async callMcpTool() {
      return '';
    },
    async getActiveContext() {
      return 'Active File: src/app.ts';
    },
    async getDiagnostics(p) {
      return `Diagnostics for ${p ?? 'all'}`;
    },
    async proposeFileEdit() {
      throw new Error('unused');
    },
  };

  const loop = new AgentLoop(runawayClient, executor);
  const res = await loop.runTurn({
    apiKey: 'gsk_test',
    model: 'openai/gpt-oss-120b',
    history: [],
    userPrompt: 'Check diagnostics',
    maxIterations: 3,
    callbacks: {
      onDelta: () => undefined,
      onToolStart: () => undefined,
      onToolDone: (t) => events.push({ name: t.name, status: t.status, preview: t.resultPreview }),
      onProposalCreated: () => undefined,
    },
  });

  assert.equal(calls, 3);
  assert.ok(res.assistantText.includes('maxAgentIterations=3'));
  assert.equal(events[0].status, 'done');
  assert.equal(events[1].status, 'error');
  assert.ok(events[1].preview?.includes('Invalid JSON tool arguments'));
  assert.equal(events[2].status, 'done');

  const aborted = new AbortController();
  aborted.abort();
  await assert.rejects(
    () =>
      loop.runTurn({
        apiKey: 'gsk_test',
        model: 'openai/gpt-oss-120b',
        history: [],
        userPrompt: 'Stop immediately',
        maxIterations: 2,
        signal: aborted.signal,
        callbacks: {
          onDelta: () => undefined,
          onToolStart: () => undefined,
          onToolDone: () => undefined,
          onProposalCreated: () => undefined,
        },
      }),
    /canceled by user/i,
  );
});

test('settings helpers handle workspace storage, invalid nodePath, and modelCacheDir propagation', async () => {
  const dir = await fsp.mkdtemp(path.join(os.tmpdir(), 'agentocode-settings-'));
  try {
    const wsStorage = resolveStorageDirectories({
      workspaceRoot: '/repo',
      extensionStoragePath: '/ext-store',
      config: { ...DEFAULT_CONFIG, stateLocation: 'workspace', knowledgeDocsInWorkspace: true },
    });
    assert.equal(wsStorage.stateDir, path.join('/repo', '.guardian'));
    assert.equal(wsStorage.knowledgeDir, path.join('/repo', 'knowledge_base'));

    assert.throws(() => resolveNodeRuntime(path.join(dir, 'missing-node.exe')), /does not exist/);

    const fakeNode = path.join(dir, 'node-custom');
    fs.writeFileSync(fakeNode, 'stub', 'utf8');
    assert.equal(resolveNodeRuntime(fakeNode).command, fakeNode);

    const env = buildWorkerEnvironment({
      workspaceRoot: '/repo',
      stateDir: '/state',
      knowledgeDir: '/state/knowledge_base',
      modelCacheDir: '/global/hf-cache',
      config: { ...DEFAULT_CONFIG, localFilesOnly: true, maxTokens: 1024, maxDistance: 0.5 },
      baseEnv: {},
    });
    assert.equal(env.AGENTOCODE_MODEL_CACHE_DIR, '/global/hf-cache');
    assert.equal(env.GUARDIAN_LOCAL_FILES_ONLY, '1');
    assert.equal(env.AGENTOCODE_MAX_TOKENS, '1024');
    assert.equal(env.GUARDIAN_MAX_DISTANCE, '0.5');
  } finally {
    await fsp.rm(dir, { recursive: true, force: true });
  }
});

test('parseDotEnv and resolveGroqApiKey handle quotes, comments, empty keys, and fallback order', async () => {
  const parsed = parseDotEnv(
    [
      '# Header comment',
      'EMPTY_KEY=',
      'export GROQ_API_KEY="gsk_quoted_123" # inline comment ignored inside quotes? no after quote',
      "SINGLE='abc#123'",
      'PLAIN=val_here # trailing comment',
    ].join('\n'),
  );
  assert.equal(parsed.EMPTY_KEY, '');
  assert.equal(parsed.SINGLE, 'abc#123');
  assert.equal(parsed.PLAIN, 'val_here');

  const dir = await fsp.mkdtemp(path.join(os.tmpdir(), 'agentocode-dotenv-'));
  const wsDir = path.join(dir, 'ws');
  const extDir = path.join(dir, 'ext');
  await fsp.mkdir(wsDir, { recursive: true });
  await fsp.mkdir(extDir, { recursive: true });

  try {
    await fsp.writeFile(path.join(extDir, '.env'), 'GROQ_API_KEY="gsk_ext"\n', 'utf8');
    await fsp.writeFile(path.join(wsDir, '.env'), 'GROQ_API_KEY=gsk_ws\n', 'utf8');

    const wsFirst = await resolveGroqApiKey({
      workspaceRoot: wsDir,
      extensionPath: extDir,
      processEnv: { GROQ_API_KEY: 'gsk_proc' },
      getSecret: async () => 'gsk_secret',
    });
    assert.deepEqual(wsFirst, { apiKey: 'gsk_ws', source: 'workspace_env' });

    await fsp.writeFile(path.join(wsDir, '.env'), 'GROQ_API_KEY=\n', 'utf8');
    const extSecond = await resolveGroqApiKey({
      workspaceRoot: wsDir,
      extensionPath: extDir,
      processEnv: { GROQ_API_KEY: 'gsk_proc' },
      getSecret: async () => 'gsk_secret',
    });
    assert.deepEqual(extSecond, { apiKey: 'gsk_ext', source: 'extension_env' });

    await fsp.unlink(path.join(extDir, '.env'));
    const procThird = await resolveGroqApiKey({
      workspaceRoot: wsDir,
      extensionPath: extDir,
      processEnv: { GROQ_API_KEY: 'gsk_proc' },
      getSecret: async () => 'gsk_secret',
    });
    assert.deepEqual(procThird, { apiKey: 'gsk_proc', source: 'process_env' });

    const secretFourth = await resolveGroqApiKey({
      workspaceRoot: wsDir,
      extensionPath: extDir,
      processEnv: {},
      getSecret: async () => 'gsk_secret',
    });
    assert.deepEqual(secretFourth, { apiKey: 'gsk_secret', source: 'secret_storage' });
  } finally {
    await fsp.rm(dir, { recursive: true, force: true });
  }
});

