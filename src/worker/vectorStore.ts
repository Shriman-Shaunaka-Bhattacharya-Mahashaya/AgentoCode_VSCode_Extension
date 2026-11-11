import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { atomicWriteFileWithRetry } from './workspace.js';

export interface VectorRecordMetadata {
  conceptId: string;
  filePath: string;
  kind: 'file_chunk' | 'directory' | 'workspace_index';
  title: string;
  fileSha256: string;
  contentHash: string;
  content: string;
  lineStart?: number;
  lineEnd?: number;
}

export interface VectorFingerprintInput {
  embedProvider: string;
  modelId: string;
  modelRevision: string;
  dtype: string;
  dimensions: number;
  maxTokens: number;
  queryPrefix: string;
  documentPrefix: string;
  normalization: string;
}

export interface VectorQueryResult {
  record: VectorRecordMetadata;
  distance: number;
}

interface StoredCollectionHeader {
  schemaVersion: number;
  fingerprint: string;
  fingerprintInput: VectorFingerprintInput;
  dimensions: number;
  recordCount: number;
  updatedAt: string;
}

const SCHEMA_VERSION = 1;

/**
 * Compute a deterministic collection fingerprint so incompatible model/dimension/prefix
 * configurations stay in isolated collections without clobbering legacy stores.
 */
export function computeVectorFingerprint(input: VectorFingerprintInput): string {
  const canonical = JSON.stringify({
    embedProvider: input.embedProvider,
    modelId: input.modelId,
    modelRevision: input.modelRevision,
    dtype: input.dtype,
    dimensions: input.dimensions,
    maxTokens: input.maxTokens,
    queryPrefix: input.queryPrefix,
    documentPrefix: input.documentPrefix,
    normalization: input.normalization,
    serialization: 'f32_le_v1',
  });
  return crypto.createHash('sha256').update(canonical).digest('hex').slice(0, 16);
}

/**
 * L2-normalize a vector in float32 space.
 */
export function normalizeFloat32(values: ArrayLike<number>): Float32Array {
  const out = new Float32Array(values.length);
  let sumSq = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    out[i] = v;
    sumSq += v * v;
  }
  const norm = Math.sqrt(sumSq);
  if (norm > 0) {
    const inv = 1 / norm;
    for (let i = 0; i < out.length; i++) {
      out[i] *= inv;
    }
  }
  return out;
}

/**
 * Pure-TypeScript persistent vector store backed by float32 binary blobs + JSON metadata,
 * with atomic persistence and automatic corrupt-store quarantine & rebuild.
 */
export class PersistentVectorStore {
  private readonly collectionDir: string;
  private readonly metaPath: string;
  private readonly recordsPath: string;
  private readonly vectorsPath: string;

  private records: VectorRecordMetadata[] = [];
  private vectors: Float32Array = new Float32Array(0);
  private idToIndex = new Map<string, number>();
  private dimensions: number;
  private corruptRecovered = false;
  private lastRecoveryNotice: string | null = null;

  constructor(
    private readonly vectorsRootDir: string,
    readonly fingerprint: string,
    private readonly fingerprintInput: VectorFingerprintInput,
  ) {
    this.dimensions = fingerprintInput.dimensions;
    this.collectionDir = path.join(vectorsRootDir, fingerprint);
    this.metaPath = path.join(this.collectionDir, 'meta.json');
    this.recordsPath = path.join(this.collectionDir, 'records.json');
    this.vectorsPath = path.join(this.collectionDir, 'vectors.bin');
  }

  get size(): number {
    return this.records.length;
  }

  get recoveryNotice(): string | null {
    return this.lastRecoveryNotice;
  }

  get wasCorruptRecovered(): boolean {
    return this.corruptRecovered;
  }

  async load(): Promise<void> {
    if (!fs.existsSync(this.collectionDir)) {
      this.records = [];
      this.vectors = new Float32Array(0);
      this.idToIndex.clear();
      return;
    }

    const hasAnyFile =
      fs.existsSync(this.metaPath) || fs.existsSync(this.recordsPath) || fs.existsSync(this.vectorsPath);
    if (!hasAnyFile) {
      return;
    }

    try {
      const [metaRaw, recordsRaw, vecBuf] = await Promise.all([
        fsp.readFile(this.metaPath, 'utf8'),
        fsp.readFile(this.recordsPath, 'utf8'),
        fsp.readFile(this.vectorsPath),
      ]);

      const meta = JSON.parse(metaRaw) as StoredCollectionHeader;
      const records = JSON.parse(recordsRaw) as VectorRecordMetadata[];
      if (
        meta.schemaVersion !== SCHEMA_VERSION ||
        meta.fingerprint !== this.fingerprint ||
        !Array.isArray(records) ||
        meta.recordCount !== records.length
      ) {
        throw new Error('Vector collection header/record count mismatch.');
      }
      if (meta.dimensions !== this.dimensions) {
        throw new Error(`Expected ${this.dimensions} dimensions, found ${meta.dimensions}.`);
      }

      const expectedBytes = records.length * this.dimensions * 4;
      if (vecBuf.byteLength !== expectedBytes) {
        throw new Error(`Expected ${expectedBytes} vector bytes, found ${vecBuf.byteLength}.`);
      }

      // Copy into an aligned Float32Array independent of Buffer pooling offsets.
      const f32 = new Float32Array(records.length * this.dimensions);
      const view = new DataView(vecBuf.buffer, vecBuf.byteOffset, vecBuf.byteLength);
      for (let i = 0; i < f32.length; i++) {
        f32[i] = view.getFloat32(i * 4, true);
      }

      this.records = records;
      this.vectors = f32;
      this.rebuildIndex();
    } catch (err: unknown) {
      const reason = err instanceof Error ? err.message : String(err);
      await this.quarantineCorruptStore(reason);
    }
  }

  private async quarantineCorruptStore(reason: string): Promise<void> {
    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    const quarantineDir = path.join(this.vectorsRootDir, `${this.fingerprint}.corrupt.${stamp}`);
    try {
      await fsp.rename(this.collectionDir, quarantineDir);
    } catch {
      // If rename fails, wipe the broken collection dir so recovery can continue.
      await fsp.rm(this.collectionDir, { recursive: true, force: true });
    }
    this.records = [];
    this.vectors = new Float32Array(0);
    this.idToIndex.clear();
    this.corruptRecovered = true;
    this.lastRecoveryNotice = `Corrupt vector store quarantined (${reason}); rebuilt fresh store.`;
  }

  private rebuildIndex(): void {
    this.idToIndex.clear();
    for (let i = 0; i < this.records.length; i++) {
      this.idToIndex.set(this.records[i].conceptId, i);
    }
  }

  getRecord(conceptId: string): VectorRecordMetadata | undefined {
    const idx = this.idToIndex.get(conceptId);
    return idx === undefined ? undefined : this.records[idx];
  }

  getAllRecords(): readonly VectorRecordMetadata[] {
    return this.records;
  }

  /**
   * Insert or replace records under stable concept IDs.
   */
  upsert(entries: Array<{ record: VectorRecordMetadata; vector: Float32Array }>): void {
    if (entries.length === 0) {
      return;
    }
    const nextRecords = [...this.records];
    const nextMap = new Map(this.idToIndex);
    const vecChunks: Float32Array[] = [];

    // Start from existing vectors copied so in-place replacements stay simple and safe.
    const existingCount = this.records.length;
    const addedCount = entries.filter((e) => !nextMap.has(e.record.conceptId)).length;
    const merged = new Float32Array((existingCount + addedCount) * this.dimensions);
    merged.set(this.vectors, 0);

    let writeCount = existingCount;
    for (const { record, vector } of entries) {
      if (vector.length !== this.dimensions) {
        throw new Error(`Vector dimension mismatch for ${record.conceptId}: expected ${this.dimensions}, got ${vector.length}`);
      }
      const normVec = normalizeFloat32(vector);
      const existingIdx = nextMap.get(record.conceptId);
      if (existingIdx !== undefined) {
        nextRecords[existingIdx] = record;
        merged.set(normVec, existingIdx * this.dimensions);
      } else {
        const idx = writeCount++;
        nextRecords.push(record);
        nextMap.set(record.conceptId, idx);
        merged.set(normVec, idx * this.dimensions);
      }
    }
    void vecChunks;

    this.records = nextRecords;
    this.vectors = merged;
    this.idToIndex = nextMap;
  }

  /**
   * Delete records matching any of the provided concept IDs. Returns count removed.
   */
  deleteByIds(ids: Iterable<string>): number {
    const removeSet = new Set(ids);
    if (removeSet.size === 0 || this.records.length === 0) {
      return 0;
    }
    const keptRecords: VectorRecordMetadata[] = [];
    const keptIndices: number[] = [];
    for (let i = 0; i < this.records.length; i++) {
      if (!removeSet.has(this.records[i].conceptId)) {
        keptRecords.push(this.records[i]);
        keptIndices.push(i);
      }
    }
    const removed = this.records.length - keptRecords.length;
    if (removed === 0) {
      return 0;
    }

    const nextVectors = new Float32Array(keptRecords.length * this.dimensions);
    for (let newIdx = 0; newIdx < keptIndices.length; newIdx++) {
      const oldIdx = keptIndices[newIdx];
      const srcOffset = oldIdx * this.dimensions;
      nextVectors.set(this.vectors.subarray(srcOffset, srcOffset + this.dimensions), newIdx * this.dimensions);
    }

    this.records = keptRecords;
    this.vectors = nextVectors;
    this.rebuildIndex();
    return removed;
  }

  /**
   * Remove all vectors associated with a given workspace relative path.
   */
  deleteByFilePath(filePath: string): number {
    const ids = this.records.filter((r) => r.filePath === filePath).map((r) => r.conceptId);
    return this.deleteByIds(ids);
  }

  /**
   * Query nearest neighbors using cosine distance (1 - cosineSimilarity).
   */
  query(queryVector: Float32Array, topK: number, maxDistance: number): VectorQueryResult[] {
    if (this.records.length === 0 || topK <= 0) {
      return [];
    }
    if (queryVector.length !== this.dimensions) {
      throw new Error(`Query vector dimension mismatch: expected ${this.dimensions}, got ${queryVector.length}`);
    }
    const q = normalizeFloat32(queryVector);
    const candidates: VectorQueryResult[] = [];

    for (let i = 0; i < this.records.length; i++) {
      const offset = i * this.dimensions;
      let dot = 0;
      for (let d = 0; d < this.dimensions; d++) {
        dot += q[d] * this.vectors[offset + d];
      }
      const distance = Math.max(0, 1 - dot);
      if (distance <= maxDistance) {
        candidates.push({ record: this.records[i], distance });
      }
    }

    candidates.sort((a, b) => a.distance - b.distance);
    return candidates.slice(0, topK);
  }

  async save(): Promise<void> {
    await fsp.mkdir(this.collectionDir, { recursive: true });
    const header: StoredCollectionHeader = {
      schemaVersion: SCHEMA_VERSION,
      fingerprint: this.fingerprint,
      fingerprintInput: this.fingerprintInput,
      dimensions: this.dimensions,
      recordCount: this.records.length,
      updatedAt: new Date().toISOString(),
    };

    const vecBuf = Buffer.allocUnsafe(this.records.length * this.dimensions * 4);
    const view = new DataView(vecBuf.buffer, vecBuf.byteOffset, vecBuf.byteLength);
    for (let i = 0; i < this.records.length * this.dimensions; i++) {
      view.setFloat32(i * 4, this.vectors[i], true);
    }

    // Write vectors + records before updating meta header atomically.
    await atomicWriteFileWithRetry(this.vectorsPath, vecBuf);
    await atomicWriteFileWithRetry(this.recordsPath, JSON.stringify(this.records));
    await atomicWriteFileWithRetry(this.metaPath, JSON.stringify(header, null, 2));
  }
}

