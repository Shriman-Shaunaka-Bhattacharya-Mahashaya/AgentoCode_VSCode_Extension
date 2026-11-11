import { EmbedDtype, EmbedProviderKind, ModelLifecycleState } from '../types.js';
import { normalizeFloat32 } from './vectorStore.js';

export type EmbedResult =
  | { ok: true; vector: Float32Array; tokenCount: number }
  | { ok: false; reason: 'overlength' | 'unavailable'; tokenCount?: number; message: string };

export interface EmbedderInfo {
  provider: EmbedProviderKind;
  modelId: string;
  modelRevision: string;
  dtype: string;
  dimensions: number;
  maxTokens: number;
  queryPrefix: string;
  documentPrefix: string;
}

export interface EmbedderMetrics {
  loadCount: number;
  embedDocCalls: number;
  embedQueryCalls: number;
  lastError: string | null;
}

export interface EmbedderProvider {
  readonly info: EmbedderInfo;
  getState(): ModelLifecycleState;
  getMetrics(): EmbedderMetrics;
  ensureLoaded(): Promise<void>;
  embedDocument(text: string): Promise<EmbedResult>;
  embedQuery(text: string): Promise<EmbedResult>;
  dispose(): Promise<void>;
}

/**
 * Determine task prefixes for Nomic/Qwen retrieval models without double-prefixing.
 */
export function resolveModelPrefixes(modelId: string): { queryPrefix: string; documentPrefix: string; defaultDimensions: number } {
  const lower = modelId.toLowerCase();
  if (lower.includes('nomic')) {
    return {
      queryPrefix: 'search_query: ',
      documentPrefix: 'search_document: ',
      defaultDimensions: 768,
    };
  }
  if (lower.includes('qwen3-embedding-4b')) {
    return {
      queryPrefix: 'Instruct: Given a code search query, retrieve relevant code or documentation\nQuery: ',
      documentPrefix: '',
      defaultDimensions: 2560,
    };
  }
  if (lower.includes('minilm') || lower.includes('bge-small') || lower.includes('e5-small')) {
    return {
      queryPrefix: '',
      documentPrefix: '',
      defaultDimensions: 384,
    };
  }
  if (lower.includes('bge-large') || lower.includes('e5-large')) {
    return {
      queryPrefix: '',
      documentPrefix: '',
      defaultDimensions: 1024,
    };
  }
  return {
    queryPrefix: '',
    documentPrefix: '',
    defaultDimensions: 768,
  };
}

export function applyPrefixOnce(text: string, prefix: string): string {
  if (!prefix) {
    return text;
  }
  return text.startsWith(prefix) ? text : `${prefix}${text}`;
}

class AsyncMutex {
  private chain: Promise<void> = Promise.resolve();

  run<T>(fn: () => Promise<T>): Promise<T> {
    const next = this.chain.then(fn, fn);
    this.chain = next.then(
      () => undefined,
      () => undefined,
    );
    return next;
  }
}

interface LoadedTransformersPipeline {
  (text: string, options?: { pooling?: string; normalize?: boolean }): Promise<{
    data: ArrayLike<number>;
    dims?: number[];
  }>;
  tokenizer: {
    encode(text: string, options?: { add_special_tokens?: boolean }): Promise<number[]> | number[];
  };
  dispose?: () => Promise<void>;
}

/**
 * Local ONNX embedding provider using @huggingface/transformers with serialized batch-1 inference
 * and strict preflight token-cap enforcement without silent truncation.
 */
export class TransformersOnnxEmbedder implements EmbedderProvider {
  readonly info: EmbedderInfo;
  private state: ModelLifecycleState = 'unloaded';
  private loadCount = 0;
  private embedDocCalls = 0;
  private embedQueryCalls = 0;
  private lastError: string | null = null;
  private pipelineInstance: LoadedTransformersPipeline | null = null;
  private loadPromise: Promise<void> | null = null;
  private readonly mutex = new AsyncMutex();

  constructor(
    private readonly options: {
      modelId: string;
      modelRevision: string;
      dtype: EmbedDtype;
      localFilesOnly: boolean;
      maxTokens: number;
      cacheDir?: string;
      onLog?: (line: string) => void;
    },
  ) {
    const prefixes = resolveModelPrefixes(options.modelId);
    this.info = {
      provider: 'transformers_onnx',
      modelId: options.modelId,
      modelRevision: options.modelRevision,
      dtype: options.dtype,
      dimensions: prefixes.defaultDimensions,
      maxTokens: options.maxTokens,
      queryPrefix: prefixes.queryPrefix,
      documentPrefix: prefixes.documentPrefix,
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
      lastError: this.lastError,
    };
  }

  async ensureLoaded(): Promise<void> {
    if (this.pipelineInstance && this.state === 'ready') {
      return;
    }
    if (this.loadPromise) {
      return this.loadPromise;
    }

    this.loadPromise = (async () => {
      this.state = 'loading';
      this.lastError = null;
      try {
        this.options.onLog?.(
          `Loading local ONNX model ${this.options.modelId} (revision=${this.options.modelRevision || 'default'}, dtype=${this.options.dtype}, localOnly=${this.options.localFilesOnly})...`,
        );
        const transformers = await import('@huggingface/transformers');
        transformers.env.allowRemoteModels = !this.options.localFilesOnly;
        transformers.env.useFSCache = true;
        if (this.options.cacheDir) {
          transformers.env.cacheDir = this.options.cacheDir;
        }

        const pipe = await transformers.pipeline('feature-extraction', this.options.modelId, {
          revision: this.options.modelRevision || undefined,
          dtype: this.options.dtype,
          local_files_only: this.options.localFilesOnly,
        });

        this.pipelineInstance = pipe as unknown as LoadedTransformersPipeline;
        this.loadCount++;
        this.state = 'ready';
        this.options.onLog?.(`Local embedding model ready (loadCount=${this.loadCount}).`);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        this.lastError = msg;
        this.state = 'error';
        this.pipelineInstance = null;
        this.options.onLog?.(`Failed to load local embedding model: ${msg}`);
        throw err;
      } finally {
        this.loadPromise = null;
      }
    })();

    return this.loadPromise;
  }

  async embedDocument(text: string): Promise<EmbedResult> {
    return this.embedWithPrefix(text, this.info.documentPrefix, 'doc');
  }

  async embedQuery(text: string): Promise<EmbedResult> {
    return this.embedWithPrefix(text, this.info.queryPrefix, 'query');
  }

  private async embedWithPrefix(
    rawText: string,
    prefix: string,
    kind: 'doc' | 'query',
  ): Promise<EmbedResult> {
    try {
      await this.ensureLoaded();
    } catch (err: unknown) {
      return {
        ok: false,
        reason: 'unavailable',
        message: err instanceof Error ? err.message : String(err),
      };
    }

    const pipe = this.pipelineInstance;
    if (!pipe) {
      return { ok: false, reason: 'unavailable', message: 'Embedding pipeline not loaded.' };
    }

    const inputText = applyPrefixOnce(rawText, prefix);
    return this.mutex.run(async () => {
      const encoded = await pipe.tokenizer.encode(inputText, { add_special_tokens: true });
      const tokenCount = encoded.length;
      if (tokenCount > this.info.maxTokens) {
        return {
          ok: false,
          reason: 'overlength',
          tokenCount,
          message: `Input uses ${tokenCount} tokens, exceeding verified operational cap of ${this.info.maxTokens} tokens.`,
        };
      }

      const tensor = await pipe(inputText, { pooling: 'mean', normalize: true });
      const vector = normalizeFloat32(tensor.data);
      if (vector.length !== this.info.dimensions) {
        // Record exact output width if a custom model differs from default guess.
        this.info.dimensions = vector.length;
      }

      if (kind === 'doc') {
        this.embedDocCalls++;
      } else {
        this.embedQueryCalls++;
      }
      return { ok: true, vector, tokenCount };
    });
  }

  async dispose(): Promise<void> {
    if (this.pipelineInstance?.dispose) {
      await this.pipelineInstance.dispose();
    }
    this.pipelineInstance = null;
    this.state = 'unloaded';
  }
}

/**
 * Local Ollama embedding provider calling /api/embed on localhost.
 */
export class OllamaEmbedder implements EmbedderProvider {
  readonly info: EmbedderInfo;
  private state: ModelLifecycleState = 'unloaded';
  private loadCount = 0;
  private embedDocCalls = 0;
  private embedQueryCalls = 0;
  private lastError: string | null = null;
  private readonly mutex = new AsyncMutex();

  constructor(
    private readonly options: {
      endpoint: string;
      modelId: string;
      maxTokens: number;
    },
  ) {
    const prefixes = resolveModelPrefixes(options.modelId);
    this.info = {
      provider: 'ollama',
      modelId: options.modelId,
      modelRevision: 'ollama-local',
      dtype: 'ollama',
      dimensions: prefixes.defaultDimensions,
      maxTokens: options.maxTokens,
      queryPrefix: prefixes.queryPrefix,
      documentPrefix: prefixes.documentPrefix,
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
      lastError: this.lastError,
    };
  }

  async ensureLoaded(): Promise<void> {
    if (this.state === 'ready') {
      return;
    }
    this.state = 'loading';
    try {
      const base = this.options.endpoint.replace(/\/+$/, '');
      const res = await fetch(`${base}/api/tags`);
      if (!res.ok) {
        throw new Error(`Ollama health check failed (${res.status})`);
      }
      this.loadCount++;
      this.state = 'ready';
      this.lastError = null;
    } catch (err: unknown) {
      this.lastError = err instanceof Error ? err.message : String(err);
      this.state = 'error';
      throw err;
    }
  }

  async embedDocument(text: string): Promise<EmbedResult> {
    return this.embedWithPrefix(text, this.info.documentPrefix, 'doc');
  }

  async embedQuery(text: string): Promise<EmbedResult> {
    return this.embedWithPrefix(text, this.info.queryPrefix, 'query');
  }

  private async embedWithPrefix(rawText: string, prefix: string, kind: 'doc' | 'query'): Promise<EmbedResult> {
    const inputText = applyPrefixOnce(rawText, prefix);
    // Conservative whitespace/char token preflight since Ollama /api/embed does not expose tokenizer counts.
    const estimatedTokens = Math.ceil(inputText.length / 3.2);
    if (estimatedTokens > this.info.maxTokens) {
      return {
        ok: false,
        reason: 'overlength',
        tokenCount: estimatedTokens,
        message: `Estimated ${estimatedTokens} tokens exceeds maxTokens=${this.info.maxTokens}.`,
      };
    }

    return this.mutex.run(async () => {
      try {
        await this.ensureLoaded();
        const base = this.options.endpoint.replace(/\/+$/, '');
        const res = await fetch(`${base}/api/embed`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model: this.info.modelId, input: inputText, truncate: false }),
        });
        if (!res.ok) {
          const msg = await res.text().catch(() => res.statusText);
          throw new Error(`Ollama /api/embed failed (${res.status}): ${msg}`);
        }
        const json = (await res.json()) as { embeddings?: number[][] };
        const rawVec = json.embeddings?.[0];
        if (!Array.isArray(rawVec) || rawVec.length === 0) {
          throw new Error('Ollama returned an empty embedding vector.');
        }
        const vector = normalizeFloat32(rawVec);
        this.info.dimensions = vector.length;
        if (kind === 'doc') {
          this.embedDocCalls++;
        } else {
          this.embedQueryCalls++;
        }
        return { ok: true, vector, tokenCount: estimatedTokens };
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        this.lastError = msg;
        this.state = 'error';
        return { ok: false, reason: 'unavailable', message: msg };
      }
    });
  }

  async dispose(): Promise<void> {
    this.state = 'unloaded';
  }
}

/**
 * Explicit lexical-only mode where zero neural embeddings or downloads occur.
 */
export class LexicalOnlyEmbedder implements EmbedderProvider {
  readonly info: EmbedderInfo = {
    provider: 'lexical_only',
    modelId: 'lexical-only',
    modelRevision: 'none',
    dtype: 'none',
    dimensions: 0,
    maxTokens: 2048,
    queryPrefix: '',
    documentPrefix: '',
  };

  getState(): ModelLifecycleState {
    return 'unloaded';
  }

  getMetrics(): EmbedderMetrics {
    return { loadCount: 0, embedDocCalls: 0, embedQueryCalls: 0, lastError: null };
  }

  async ensureLoaded(): Promise<void> {
    // No model to load in lexical_only mode.
  }

  async embedDocument(_text: string): Promise<EmbedResult> {
    return { ok: false, reason: 'unavailable', message: 'Embedding provider set to lexical_only.' };
  }

  async embedQuery(_text: string): Promise<EmbedResult> {
    return { ok: false, reason: 'unavailable', message: 'Embedding provider set to lexical_only.' };
  }

  async dispose(): Promise<void> {
    // No-op.
  }
}

