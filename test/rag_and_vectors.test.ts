import assert from 'node:assert/strict';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { ModelLifecycleState, WorkerConfig } from '../src/types.js';
import { EmbedderInfo, EmbedderMetrics, EmbedderProvider, EmbedResult } from '../src/worker/embeddings.js';
import { LocalRagEngine } from '../src/worker/rag.js';

class DeterministicTestEmbedder implements EmbedderProvider {
  readonly info: EmbedderInfo;
  private state: ModelLifecycleState = 'unloaded';
  private loadCount = 0;
  embedDocCalls = 0;
  embedQueryCalls = 0;

  constructor(maxTokens = 2048) {
    this.info = {
      provider: 'transformers_onnx',
      modelId: 'test-nomic-fake',
      modelRevision: 'rev1',
      dtype: 'q8',
      dimensions: 16,
      maxTokens,
      queryPrefix: 'search_query: ',
      documentPrefix: 'search_document: ',
    };
  }

  getState(): ModelLifecycleState {
    return this.state;
  }

  getMetrics(): EmbedderMetrics {
    return {
      loadCount: this.loadCount,
      embedDocCalls: this.embedDocCalls,
      embedQueryCalls: this.embedQueryCalls,
      lastError: null,
    };
  }

  async ensureLoaded(): Promise<void> {
    if (this.state !== 'ready') {
      this.loadCount++;
      this.state = 'ready';
    }
  }

  private vectorize(text: string): Float32Array {
    // Simple deterministic character-trigram hash projection into 16 dimensions.
    const vec = new Float32Array(16);
    const clean = text.toLowerCase();
    for (let i = 0; i < clean.length; i++) {
      const bucket = (clean.charCodeAt(i) * 31 + (clean.charCodeAt(i + 1) || 0)) % 16;
      vec[bucket] += 1;
    }
    return vec;
  }

  async embedDocument(text: string): Promise<EmbedResult> {
    await this.ensureLoaded();
    const approxTokens = Math.ceil(text.length / 4);
    if (approxTokens > this.info.maxTokens) {
      return {
        ok: false,
        reason: 'overlength',
        tokenCount: approxTokens,
        message: `Overlength (${approxTokens} > ${this.info.maxTokens})`,
      };
    }
    this.embedDocCalls++;
    return { ok: true, vector: this.vectorize(text), tokenCount: approxTokens };
  }

  async embedQuery(text: string): Promise<EmbedResult> {
    await this.ensureLoaded();
    this.embedQueryCalls++;
    return { ok: true, vector: this.vectorize(text), tokenCount: Math.ceil(text.length / 4) };
  }

  async dispose(): Promise<void> {
    this.state = 'unloaded';
  }
}

function makeConfig(workspaceRoot: string, maxTokens = 2048): WorkerConfig {
  const stateDir = path.join(workspaceRoot, '.guardian');
  return {
    workspaceRoot,
    stateDir,
    knowledgeDir: path.join(stateDir, 'knowledge_base'),
    writeKnowledgeDocs: true,
    embedProvider: 'transformers_onnx',
    embedModelId: 'test-nomic-fake',
    embedModelRevision: 'rev1',
    embedDtype: 'q8',
    ollamaEndpoint: 'http://127.0.0.1:11434',
    localFilesOnly: true,
    maxTokens,
    maxDistance: 0.95,
  };
}

test('incremental lifecycle handles sync, no-op caching, dependency refresh, staleness rejection, and deletion', async () => {
  const ws = await fsp.mkdtemp(path.join(os.tmpdir(), 'agentocode-rag-'));
  try {
    await fsp.writeFile(
      path.join(ws, 'calc.py'),
      'def divide(a, b):\n    if b == 0:\n        raise ZeroDivisionError("zero denominator")\n    return a / b\n',
      'utf8',
    );
    await fsp.writeFile(
      path.join(ws, 'main.py'),
      'from calc import divide\n\ndef run():\n    return divide(10, 2)\n',
      'utf8',
    );

    const embedder = new DeterministicTestEmbedder();
    const engine = new LocalRagEngine(makeConfig(ws), embedder);

    const st1 = await engine.syncKnowledge(false);
    assert.equal(st1.conceptCount, 2);
    assert.equal(st1.indexedCount, 2);
    assert.equal(st1.indexAvailable, true);
    assert.equal(embedder.embedDocCalls, 2);
    assert.ok(fs.existsSync(path.join(ws, '.guardian', 'knowledge_base', 'index.md')));

    // No-op sync makes zero new embedding calls.
    await engine.syncKnowledge(false);
    assert.equal(embedder.embedDocCalls, 2);

    // Semantic search finds calc.py while fresh.
    const searchFresh = await engine.ragSearch('ZeroDivisionError divide', 4);
    assert.equal(searchFresh.retrievalMode, 'semantic');
    assert.equal(searchFresh.staleRejectedCount, 0);
    assert.ok(searchFresh.results.some((r) => r.path === 'calc.py'));

    // Modify calc.py on disk WITHOUT syncing -> ragSearch must reject stale hit!
    await fsp.writeFile(
      path.join(ws, 'calc.py'),
      'def divide(a, b):\n    return 0 if b == 0 else a / b\n',
      'utf8',
    );
    const searchStale = await engine.ragSearch('ZeroDivisionError divide', 4);
    assert.ok(searchStale.staleRejectedCount >= 1);
    assert.ok(!searchStale.results.some((r) => r.path === 'calc.py' && r.retrievalMode === 'semantic'));

    // Re-sync refreshes calc.py AND its dependent main.py (2 doc embed calls).
    const beforeCalls = embedder.embedDocCalls;
    await engine.syncKnowledge(false);
    assert.equal(embedder.embedDocCalls - beforeCalls, 2);

    // Delete calc.py -> cleans up immediately.
    await fsp.unlink(path.join(ws, 'calc.py'));
    const stAfterDelete = await engine.syncKnowledge(false);
    assert.equal(stAfterDelete.conceptCount, 1);
    assert.equal(stAfterDelete.indexedCount, 1);
  } finally {
    await fsp.rm(ws, { recursive: true, force: true });
  }
});

test('overlength chunks stay visibly unindexed without silent truncation and corrupt stores recover cleanly', async () => {
  const ws = await fsp.mkdtemp(path.join(os.tmpdir(), 'agentocode-overlen-'));
  try {
    await fsp.writeFile(path.join(ws, 'ok.ts'), 'export const x = 1;\n', 'utf8');
    await fsp.writeFile(path.join(ws, 'huge.txt'), 'token '.repeat(250), 'utf8');

    const embedder = new DeterministicTestEmbedder(40);
    const cfg = makeConfig(ws, 40);
    const engine1 = new LocalRagEngine(cfg, embedder);
    const st = await engine1.syncKnowledge(false);

    assert.equal(st.indexedCount, 1);
    assert.ok(st.unindexedOverlengthCount >= 1);

    // Corrupt vectors.bin intentionally, then reload a fresh engine -> must quarantine and rebuild.
    const vecDir = path.join(cfg.stateDir, 'vectors', st.vectorFingerprint);
    await fsp.writeFile(path.join(vecDir, 'vectors.bin'), Buffer.from('corrupted-bytes'));

    const notices: string[] = [];
    const engine2 = new LocalRagEngine(cfg, embedder, (m) => notices.push(m));
    const stRecovered = await engine2.syncKnowledge(false);
    assert.ok(notices.some((n) => n.includes('Corrupt vector store quarantined')));
    assert.equal(stRecovered.indexedCount, 1);
    const vectorSubdirs = fs.readdirSync(path.join(cfg.stateDir, 'vectors'));
    assert.ok(vectorSubdirs.some((d) => d.includes('.corrupt.')));
  } finally {
    await fsp.rm(ws, { recursive: true, force: true });
  }
});

test('lexical_only mode reports indexAvailable=false and retrievalMode=lexical_fallback', async () => {
  const ws = await fsp.mkdtemp(path.join(os.tmpdir(), 'agentocode-lex-'));
  try {
    await fsp.writeFile(path.join(ws, 'history.py'), 'def copy_history(items):\n    return list(items)\n', 'utf8');
    const cfg: WorkerConfig = {
      ...makeConfig(ws),
      embedProvider: 'lexical_only',
    };
    const engine = new LocalRagEngine(cfg);
    await engine.syncKnowledge(false);
    const res = await engine.ragSearch('copy_history', 3);
    assert.equal(res.indexAvailable, false);
    assert.equal(res.retrievalMode, 'lexical_fallback');
    assert.ok(res.results.some((r) => r.path === 'history.py'));
  } finally {
    await fsp.rm(ws, { recursive: true, force: true });
  }
});

