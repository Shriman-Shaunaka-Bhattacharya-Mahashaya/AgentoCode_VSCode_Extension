import assert from 'node:assert/strict';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import {
  applyPrefixOnce,
  LexicalOnlyEmbedder,
  OllamaEmbedder,
  resolveModelPrefixes,
} from '../src/worker/embeddings.js';
import {
  computeVectorFingerprint,
  normalizeFloat32,
  PersistentVectorStore,
  VectorFingerprintInput,
} from '../src/worker/vectorStore.js';

function makeFingerprintInput(overrides: Partial<VectorFingerprintInput> = {}): VectorFingerprintInput {
  return {
    embedProvider: 'transformers_onnx',
    modelId: 'nomic-ai/nomic-embed-text-v1.5',
    modelRevision: 'e9b6763023c676ca8431644204f50c2b100d9aab',
    dtype: 'q8',
    dimensions: 4,
    maxTokens: 2048,
    queryPrefix: 'search_query: ',
    documentPrefix: 'search_document: ',
    normalization: 'l2_v1',
    ...overrides,
  };
}

test('computeVectorFingerprint isolates collections across model/dimension/prefix changes', () => {
  const base = computeVectorFingerprint(makeFingerprintInput());
  assert.equal(base, computeVectorFingerprint(makeFingerprintInput()));
  assert.notEqual(base, computeVectorFingerprint(makeFingerprintInput({ modelRevision: 'other' })));
  assert.notEqual(base, computeVectorFingerprint(makeFingerprintInput({ dtype: 'fp32' })));
  assert.notEqual(base, computeVectorFingerprint(makeFingerprintInput({ dimensions: 768 })));
  assert.notEqual(base, computeVectorFingerprint(makeFingerprintInput({ queryPrefix: '' })));
});

test('normalizeFloat32 returns unit vectors and handles zero vectors safely', () => {
  const unit = normalizeFloat32([3, 4]);
  assert.ok(Math.abs(unit[0] - 0.6) < 1e-5);
  assert.ok(Math.abs(unit[1] - 0.8) < 1e-5);

  const zero = normalizeFloat32([0, 0, 0]);
  assert.deepEqual(Array.from(zero), [0, 0, 0]);
});

test('PersistentVectorStore supports save/reload, in-place replacement, deletion compaction, and distance filtering', async () => {
  const dir = await fsp.mkdtemp(path.join(os.tmpdir(), 'agentocode-vec-'));
  try {
    const fpInput = makeFingerprintInput();
    const fp = computeVectorFingerprint(fpInput);
    const store = new PersistentVectorStore(dir, fp, fpInput);
    await store.load();

    store.upsert([
      {
        record: {
          conceptId: 'file:a.ts#chunk0',
          filePath: 'a.ts',
          kind: 'file_chunk',
          title: 'a.ts',
          fileSha256: 'sha-a1',
          contentHash: 'c-a1',
          content: 'alpha',
          lineStart: 1,
          lineEnd: 10,
        },
        vector: new Float32Array([1, 0, 0, 0]),
      },
      {
        record: {
          conceptId: 'file:b.ts#chunk0',
          filePath: 'b.ts',
          kind: 'file_chunk',
          title: 'b.ts',
          fileSha256: 'sha-b1',
          contentHash: 'c-b1',
          content: 'beta',
          lineStart: 1,
          lineEnd: 5,
        },
        vector: new Float32Array([0, 1, 0, 0]),
      },
    ]);

    // In-place replacement under stable concept ID should not increase count.
    store.upsert([
      {
        record: {
          conceptId: 'file:a.ts#chunk0',
          filePath: 'a.ts',
          kind: 'file_chunk',
          title: 'a.ts updated',
          fileSha256: 'sha-a2',
          contentHash: 'c-a2',
          content: 'alpha updated',
          lineStart: 1,
          lineEnd: 12,
        },
        vector: new Float32Array([0.9, 0.1, 0, 0]),
      },
    ]);
    assert.equal(store.size, 2);

    assert.throws(() => store.query(new Float32Array([1, 0]), 2, 0.5), /dimension mismatch/);

    await store.save();

    const reloaded = new PersistentVectorStore(dir, fp, fpInput);
    await reloaded.load();
    assert.equal(reloaded.size, 2);
    assert.equal(reloaded.getRecord('file:a.ts#chunk0')?.fileSha256, 'sha-a2');

    const closeHits = reloaded.query(new Float32Array([1, 0, 0, 0]), 5, 0.25);
    assert.equal(closeHits.length, 1);
    assert.equal(closeHits[0].record.filePath, 'a.ts');

    const removed = reloaded.deleteByFilePath('a.ts');
    assert.equal(removed, 1);
    assert.equal(reloaded.size, 1);
    const remainingHits = reloaded.query(new Float32Array([0, 1, 0, 0]), 5, 0.2);
    assert.equal(remainingHits.length, 1);
    assert.equal(remainingHits[0].record.filePath, 'b.ts');
  } finally {
    await fsp.rm(dir, { recursive: true, force: true });
  }
});

test('PersistentVectorStore quarantines malformed JSON or dimension-mismatched collections', async () => {
  const dir = await fsp.mkdtemp(path.join(os.tmpdir(), 'agentocode-veccorrupt-'));
  try {
    const fpInput = makeFingerprintInput();
    const fp = computeVectorFingerprint(fpInput);
    const colDir = path.join(dir, fp);
    await fsp.mkdir(colDir, { recursive: true });
    await fsp.writeFile(path.join(colDir, 'meta.json'), '{bad json', 'utf8');
    await fsp.writeFile(path.join(colDir, 'records.json'), '[]', 'utf8');
    await fsp.writeFile(path.join(colDir, 'vectors.bin'), Buffer.alloc(0));

    const store = new PersistentVectorStore(dir, fp, fpInput);
    await store.load();
    assert.equal(store.wasCorruptRecovered, true);
    assert.equal(store.size, 0);
    assert.ok(fs.readdirSync(dir).some((name) => name.startsWith(`${fp}.corrupt.`)));
  } finally {
    await fsp.rm(dir, { recursive: true, force: true });
  }
});

test('resolveModelPrefixes and applyPrefixOnce map common embedding models cleanly', () => {
  const nomic = resolveModelPrefixes('nomic-ai/nomic-embed-text-v1.5');
  assert.equal(nomic.defaultDimensions, 768);
  assert.equal(nomic.queryPrefix, 'search_query: ');
  assert.equal(nomic.documentPrefix, 'search_document: ');

  assert.equal(applyPrefixOnce('hello', 'search_query: '), 'search_query: hello');
  assert.equal(applyPrefixOnce('search_query: hello', 'search_query: '), 'search_query: hello');

  assert.equal(resolveModelPrefixes('Qwen/Qwen3-Embedding-4B').defaultDimensions, 2560);
  assert.equal(resolveModelPrefixes('all-MiniLM-L6-v2').defaultDimensions, 384);
  assert.equal(resolveModelPrefixes('BAAI/bge-large-en-v1.5').defaultDimensions, 1024);
});

test('OllamaEmbedder executes local HTTP protocol, preflight length gate, normalization, and error states', async () => {
  const seenInputs: string[] = [];
  let failEmbed = false;

  const server = http.createServer(async (req, res) => {
    if (req.url === '/api/tags') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ models: [{ name: 'nomic-embed-text' }] }));
      return;
    }
    if (req.url === '/api/embed') {
      const chunks: Buffer[] = [];
      for await (const chunk of req) {
        chunks.push(chunk as Buffer);
      }
      const body = JSON.parse(Buffer.concat(chunks).toString('utf8')) as { input: string; truncate?: boolean };
      seenInputs.push(body.input);
      assert.equal(body.truncate, false);

      if (failEmbed) {
        res.writeHead(500, { 'Content-Type': 'text/plain' });
        res.end('ollama error');
        return;
      }
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ embeddings: [[3, 4, 0, 0]] }));
      return;
    }
    res.writeHead(404);
    res.end();
  });

  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const addr = server.address() as { port: number };
  const endpoint = `http://127.0.0.1:${addr.port}`;

  try {
    const embedder = new OllamaEmbedder({
      endpoint,
      modelId: 'nomic-embed-text',
      maxTokens: 20,
    });

    const docRes = await embedder.embedDocument('calc code');
    assert.equal(docRes.ok, true);
    if (docRes.ok) {
      assert.equal(docRes.vector.length, 4);
      assert.ok(Math.abs(docRes.vector[0] - 0.6) < 1e-5);
    }

    const queryRes = await embedder.embedQuery('find calc');
    assert.equal(queryRes.ok, true);

    // Overlength input must fail preflight BEFORE touching /api/embed.
    const countBefore = seenInputs.length;
    const tooBig = await embedder.embedDocument('x'.repeat(200));
    assert.equal(tooBig.ok, false);
    if (!tooBig.ok) {
      assert.equal(tooBig.reason, 'overlength');
    }
    assert.equal(seenInputs.length, countBefore);
    assert.deepEqual(seenInputs, ['search_document: calc code', 'search_query: find calc']);

    const metrics = embedder.getMetrics();
    assert.equal(metrics.embedDocCalls, 1);
    assert.equal(metrics.embedQueryCalls, 1);

    failEmbed = true;
    const errRes = await embedder.embedQuery('boom');
    assert.equal(errRes.ok, false);
    assert.equal(embedder.getState(), 'error');
  } finally {
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
});

test('LexicalOnlyEmbedder reports unloaded state and unavailable embeddings cleanly', async () => {
  const lex = new LexicalOnlyEmbedder();
  await lex.ensureLoaded();
  assert.equal(lex.getState(), 'unloaded');
  assert.equal((await lex.embedDocument('a')).ok, false);
  assert.equal((await lex.embedQuery('a')).ok, false);
});

