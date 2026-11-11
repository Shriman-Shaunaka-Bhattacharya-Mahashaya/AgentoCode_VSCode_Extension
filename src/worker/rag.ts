import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import {
  KnowledgeStatusSnapshot,
  RagSearchHit,
  RagSearchResponse,
  WorkerConfig,
} from '../types.js';
import {
  EmbedderProvider,
  LexicalOnlyEmbedder,
  OllamaEmbedder,
  TransformersOnnxEmbedder,
} from './embeddings.js';
import {
  analyzeSourceFile,
  FileStructuralAnalysis,
  renderChunkEmbeddingText,
  renderDirectoryKnowledgeDocument,
  renderFileKnowledgeDocument,
} from './facts.js';
import {
  computeVectorFingerprint,
  PersistentVectorStore,
  VectorRecordMetadata,
} from './vectorStore.js';
import {
  atomicWriteFileWithRetry,
  knowledgeRelPathForDir,
  knowledgeRelPathForFile,
  normalizeRelPath,
  resolveContainedPath,
  scanEligibleWorkspaceFiles,
  sha256Hex,
} from './workspace.js';

interface ManifestFileEntry {
  sha256: string;
  sizeBytes: number;
  language: string;
  parserUsed: string;
  imports: string[];
  resolvedDeps: string[];
  conceptIds: string[];
  unitCount: number;
  status: 'indexed' | 'overlength' | 'lexical_only' | 'error';
  overlengthChunks: number;
  errorReason?: string;
  updatedAt: string;
}

interface ManifestState {
  schemaVersion: number;
  workspaceRoot: string;
  files: Record<string, ManifestFileEntry>;
  lastSyncAt: string | null;
}

const MANIFEST_SCHEMA_VERSION = 1;

/**
 * Create the configured local embedding provider (Transformers ONNX, Ollama, or lexical_only).
 */
export function createConfiguredEmbedder(
  config: WorkerConfig,
  onLog?: (msg: string) => void,
): EmbedderProvider {
  if (config.embedProvider === 'lexical_only') {
    return new LexicalOnlyEmbedder();
  }
  if (config.embedProvider === 'ollama') {
    return new OllamaEmbedder({
      endpoint: config.ollamaEndpoint,
      modelId: config.embedModelId || 'nomic-embed-text',
      maxTokens: config.maxTokens,
    });
  }
  return new TransformersOnnxEmbedder({
    modelId: config.embedModelId || 'nomic-ai/nomic-embed-text-v1.5',
    modelRevision: config.embedModelRevision,
    dtype: config.embedDtype,
    localFilesOnly: config.localFilesOnly,
    maxTokens: config.maxTokens,
    cacheDir: config.modelCacheDir || path.join(config.stateDir, 'hf-cache'),
    onLog,
  });
}

/**
 * Resolve static import specifiers (TS/JS relative paths or Python module names) to workspace-relative file paths.
 */
export function resolveImportToWorkspaceFile(
  importerRelPath: string,
  specifier: string,
  allFileSet: Set<string>,
): string | null {
  if (!specifier) {
    return null;
  }
  const importerDir = path.posix.dirname(importerRelPath);

  // TS/JS relative imports
  if (specifier.startsWith('.')) {
    const clean = specifier.replace(/\.(js|jsx|mjs|cjs)$/, '');
    const joined = normalizeRelPath(path.posix.join(importerDir === '.' ? '' : importerDir, clean));
    const candidates = [
      specifier.startsWith('./') || specifier.startsWith('../')
        ? normalizeRelPath(path.posix.join(importerDir === '.' ? '' : importerDir, specifier))
        : joined,
      joined,
      `${joined}.ts`,
      `${joined}.tsx`,
      `${joined}.js`,
      `${joined}.jsx`,
      `${joined}.mjs`,
      `${joined}/index.ts`,
      `${joined}/index.tsx`,
      `${joined}/index.js`,
    ];
    for (const candidate of candidates) {
      if (allFileSet.has(candidate)) {
        return candidate;
      }
    }
    return null;
  }

  // Python module dot-paths
  const pySlash = specifier.replace(/\./g, '/');
  const pyCandidates = [
    `${pySlash}.py`,
    `${pySlash}/__init__.py`,
    importerDir !== '.' ? `${importerDir}/${pySlash}.py` : '',
  ].filter(Boolean);
  for (const candidate of pyCandidates) {
    if (allFileSet.has(candidate)) {
      return candidate;
    }
  }
  return null;
}

/**
 * Local-only incremental RAG & knowledge lifecycle manager (zero LLM API calls).
 */
export class LocalRagEngine {
  private readonly manifestPath: string;
  private readonly vectorsRootDir: string;
  private readonly embedder: EmbedderProvider;
  private readonly vectorStore: PersistentVectorStore | null;
  private readonly fingerprint: string;

  private manifest: ManifestState;
  private syncRunning = false;
  private queuedCount = 0;
  private syncQueueTail: Promise<void> = Promise.resolve();
  private initialized = false;
  private recentFailures: Array<{ path: string; reason: string }> = [];

  constructor(
    private readonly config: WorkerConfig,
    embedderOverride?: EmbedderProvider,
    private readonly onLog: (msg: string) => void = () => undefined,
  ) {
    this.manifestPath = path.join(config.stateDir, 'manifest.json');
    this.vectorsRootDir = path.join(config.stateDir, 'vectors');
    this.embedder = embedderOverride ?? createConfiguredEmbedder(config, onLog);

    const info = this.embedder.info;
    this.fingerprint =
      info.provider === 'lexical_only'
        ? 'lexical_only'
        : computeVectorFingerprint({
            embedProvider: info.provider,
            modelId: info.modelId,
            modelRevision: info.modelRevision,
            dtype: info.dtype,
            dimensions: info.dimensions,
            maxTokens: info.maxTokens,
            queryPrefix: info.queryPrefix,
            documentPrefix: info.documentPrefix,
            normalization: 'l2_v1',
          });

    this.vectorStore =
      info.provider === 'lexical_only'
        ? null
        : new PersistentVectorStore(this.vectorsRootDir, this.fingerprint, {
            embedProvider: info.provider,
            modelId: info.modelId,
            modelRevision: info.modelRevision,
            dtype: info.dtype,
            dimensions: info.dimensions,
            maxTokens: info.maxTokens,
            queryPrefix: info.queryPrefix,
            documentPrefix: info.documentPrefix,
            normalization: 'l2_v1',
          });

    this.manifest = {
      schemaVersion: MANIFEST_SCHEMA_VERSION,
      workspaceRoot: config.workspaceRoot,
      files: {},
      lastSyncAt: null,
    };
  }

  async initialize(): Promise<void> {
    if (this.initialized) {
      return;
    }
    await fsp.mkdir(this.config.stateDir, { recursive: true });
    await this.loadManifest();
    if (this.vectorStore) {
      await this.vectorStore.load();
      if (this.vectorStore.recoveryNotice) {
        this.onLog(this.vectorStore.recoveryNotice);
      }
    }
    this.initialized = true;
  }

  private async loadManifest(): Promise<void> {
    if (!fs.existsSync(this.manifestPath)) {
      return;
    }
    try {
      const raw = await fsp.readFile(this.manifestPath, 'utf8');
      const parsed = JSON.parse(raw) as ManifestState;
      if (parsed.schemaVersion !== MANIFEST_SCHEMA_VERSION || typeof parsed.files !== 'object') {
        throw new Error('Incompatible manifest schema.');
      }
      this.manifest = parsed;
    } catch (err: unknown) {
      const stamp = new Date().toISOString().replace(/[:.]/g, '-');
      const backup = `${this.manifestPath}.corrupt.${stamp}`;
      try {
        await fsp.rename(this.manifestPath, backup);
      } catch {
        // Best-effort backup.
      }
      this.onLog(
        `Malformed manifest preserved at ${path.basename(backup)}; resetting manifest (${err instanceof Error ? err.message : String(err)}).`,
      );
      this.manifest = {
        schemaVersion: MANIFEST_SCHEMA_VERSION,
        workspaceRoot: this.config.workspaceRoot,
        files: {},
        lastSyncAt: null,
      };
    }
  }

  private async saveManifest(): Promise<void> {
    await atomicWriteFileWithRetry(this.manifestPath, JSON.stringify(this.manifest, null, 2));
  }

  async preloadModel(): Promise<KnowledgeStatusSnapshot> {
    await this.initialize();
    await this.embedder.ensureLoaded();
    return this.getStatus();
  }

  /**
   * Queue serialized incremental synchronization without duplicate concurrent runs.
   */
  syncKnowledge(force = false): Promise<KnowledgeStatusSnapshot> {
    this.queuedCount++;
    const run = this.syncQueueTail.then(
      () => this.runSyncInternal(force),
      () => this.runSyncInternal(force),
    );
    this.syncQueueTail = run.then(
      () => undefined,
      () => undefined,
    );
    return run;
  }

  private async runSyncInternal(force: boolean): Promise<KnowledgeStatusSnapshot> {
    await this.initialize();
    this.queuedCount = Math.max(0, this.queuedCount - 1);
    this.syncRunning = true;

    try {
      const scanned = await scanEligibleWorkspaceFiles(this.config.workspaceRoot);
      const scannedByPath = new Map(scanned.map((f) => [f.relativePath, f]));
      const allFileSet = new Set(scannedByPath.keys());

      // Detect deletions immediately and purge knowledge docs + vectors independently of model availability.
      const deletedPaths = Object.keys(this.manifest.files).filter((p) => !scannedByPath.has(p));
      let vectorsModified = false;
      for (const delPath of deletedPaths) {
        delete this.manifest.files[delPath];
        if (this.vectorStore) {
          if (this.vectorStore.deleteByFilePath(delPath) > 0) {
            vectorsModified = true;
          }
        }
        if (this.config.writeKnowledgeDocs) {
          const docPath = path.join(this.config.knowledgeDir, knowledgeRelPathForFile(delPath));
          await fsp.rm(path.dirname(docPath), { recursive: true, force: true }).catch(() => undefined);
        }
      }

      // First pass: detect direct changes & build analyses for changed files.
      const directChanged = new Set<string>();
      const analyses = new Map<string, FileStructuralAnalysis>();
      const resolvedDepsMap = new Map<string, string[]>();

      for (const file of scanned) {
        const prev = this.manifest.files[file.relativePath];
        const needsVectorReindex =
          Boolean(this.vectorStore) &&
          prev?.status !== 'indexed' &&
          prev?.status !== 'overlength';

        if (force || !prev || prev.sha256 !== file.sha256 || needsVectorReindex || (this.vectorStore?.wasCorruptRecovered ?? false)) {
          directChanged.add(file.relativePath);
          const analysis = analyzeSourceFile(file.relativePath, file.text);
          analyses.set(file.relativePath, analysis);
          const resolved = analysis.imports
            .map((spec) => resolveImportToWorkspaceFile(file.relativePath, spec, allFileSet))
            .filter((x): x is string => Boolean(x));
          resolvedDepsMap.set(file.relativePath, Array.from(new Set(resolved)));
        }
      }

      // Propagate updates to files that import a modified or deleted file.
      const changedOrDeleted = new Set<string>([...directChanged, ...deletedPaths]);
      const impactedFiles = new Set<string>(directChanged);
      for (const [relPath, entry] of Object.entries(this.manifest.files)) {
        if (impactedFiles.has(relPath) || !scannedByPath.has(relPath)) {
          continue;
        }
        if (entry.resolvedDeps.some((dep) => changedOrDeleted.has(dep))) {
          impactedFiles.add(relPath);
          const file = scannedByPath.get(relPath)!;
          const analysis = analyzeSourceFile(file.relativePath, file.text);
          analyses.set(file.relativePath, analysis);
          const resolved = analysis.imports
            .map((spec) => resolveImportToWorkspaceFile(file.relativePath, spec, allFileSet))
            .filter((x): x is string => Boolean(x));
          resolvedDepsMap.set(file.relativePath, Array.from(new Set(resolved)));
        }
      }

      // Duplicate / no-op sync short-circuits with zero new embedding calls.
      if (impactedFiles.size === 0 && deletedPaths.length === 0 && !force) {
        return await this.getStatus();
      }

      const nowIso = new Date().toISOString();
      const failureList: Array<{ path: string; reason: string }> = [];

      // Ensure embedding model is loaded once if semantic embeddings are active and files changed.
      let modelReady = false;
      if (this.vectorStore && impactedFiles.size > 0) {
        try {
          await this.embedder.ensureLoaded();
          modelReady = this.embedder.getState() === 'ready';
        } catch (err: unknown) {
          failureList.push({
            path: '<embedding-model>',
            reason: err instanceof Error ? err.message : String(err),
          });
        }
      }

      for (const relPath of Array.from(impactedFiles).sort()) {
        const file = scannedByPath.get(relPath)!;
        const analysis = analyses.get(relPath) ?? analyzeSourceFile(relPath, file.text);
        const resolvedDeps = resolvedDepsMap.get(relPath) ?? [];

        if (this.config.writeKnowledgeDocs) {
          const docContent = renderFileKnowledgeDocument({
            relativePath: relPath,
            sha256: file.sha256,
            sizeBytes: file.sizeBytes,
            analysis,
            updatedAt: nowIso,
            resolvedDependencies: resolvedDeps,
          });
          const docTarget = path.join(this.config.knowledgeDir, knowledgeRelPathForFile(relPath));
          await atomicWriteFileWithRetry(docTarget, docContent);
        }

        const conceptIds: string[] = [];
        let overlengthCount = 0;
        let fileStatus: ManifestFileEntry['status'] = this.vectorStore ? 'lexical_only' : 'lexical_only';
        let fileErr: string | undefined;

        if (this.vectorStore) {
          if (this.vectorStore.deleteByFilePath(relPath) > 0) {
            vectorsModified = true;
          }
          if (modelReady) {
            const upserts: Array<{ record: VectorRecordMetadata; vector: Float32Array }> = [];
            for (let idx = 0; idx < analysis.chunks.length; idx++) {
              const chunk = analysis.chunks[idx];
              const conceptId = `file:${relPath}#chunk${idx}`;
              const embedPayload = renderChunkEmbeddingText({
                relativePath: relPath,
                analysis,
                chunkIndex: idx,
              });
              const embedded = await this.embedder.embedDocument(embedPayload);
              if (embedded.ok) {
                conceptIds.push(conceptId);
                upserts.push({
                  record: {
                    conceptId,
                    filePath: relPath,
                    kind: 'file_chunk',
                    title: `${relPath} — ${chunk.title}`,
                    fileSha256: file.sha256,
                    contentHash: sha256Hex(embedPayload),
                    content: embedPayload,
                    lineStart: chunk.lineStart,
                    lineEnd: chunk.lineEnd,
                  },
                  vector: embedded.vector,
                });
              } else if (embedded.reason === 'overlength') {
                overlengthCount++;
                failureList.push({
                  path: `${relPath}#chunk${idx}`,
                  reason: embedded.message,
                });
              } else {
                fileErr = embedded.message;
                failureList.push({ path: relPath, reason: embedded.message });
              }
            }
            if (upserts.length > 0) {
              this.vectorStore.upsert(upserts);
              vectorsModified = true;
            }
            fileStatus =
              fileErr ? 'error' : overlengthCount > 0 && upserts.length === 0 ? 'overlength' : 'indexed';
          } else {
            fileStatus = 'lexical_only';
            fileErr = this.embedder.getMetrics().lastError ?? 'Embedding model unavailable.';
          }
        }

        this.manifest.files[relPath] = {
          sha256: file.sha256,
          sizeBytes: file.sizeBytes,
          language: analysis.language,
          parserUsed: analysis.parserUsed,
          imports: analysis.imports,
          resolvedDeps,
          conceptIds,
          unitCount: analysis.units.length,
          status: fileStatus,
          overlengthChunks: overlengthCount,
          errorReason: fileErr,
          updatedAt: nowIso,
        };
      }

      if (this.config.writeKnowledgeDocs) {
        await this.updateDirectoryKnowledgeDocs(scanned, nowIso);
      }

      this.manifest.lastSyncAt = nowIso;
      this.recentFailures = failureList.slice(0, 25);

      if (this.vectorStore && vectorsModified) {
        await this.vectorStore.save();
      }
      await this.saveManifest();
      return await this.getStatus();
    } finally {
      this.syncRunning = false;
    }
  }

  private async updateDirectoryKnowledgeDocs(
    scanned: Array<{ relativePath: string; sha256: string }>,
    updatedAt: string,
  ): Promise<void> {
    // Rebuild directories/ cleanly so deleted subdirectories do not leave stale concept docs behind.
    await fsp.rm(path.join(this.config.knowledgeDir, 'directories'), { recursive: true, force: true }).catch(() => undefined);

    const dirFiles = new Map<string, Array<{ relativePath: string; sha256: string; unitCount: number; language: string }>>();
    const dirSubdirs = new Map<string, Set<string>>();

    const ensureDir = (d: string) => {
      if (!dirFiles.has(d)) {
        dirFiles.set(d, []);
      }
      if (!dirSubdirs.has(d)) {
        dirSubdirs.set(d, new Set());
      }
    };

    ensureDir('');
    for (const file of scanned) {
      const entry = this.manifest.files[file.relativePath];
      const parent = normalizeRelPath(path.posix.dirname(file.relativePath));
      ensureDir(parent);
      dirFiles.get(parent)!.push({
        relativePath: file.relativePath,
        sha256: file.sha256,
        unitCount: entry?.unitCount ?? 0,
        language: entry?.language ?? 'text',
      });

      // Register ancestor chain bottom-up.
      let cur = parent;
      while (cur) {
        const up = normalizeRelPath(path.posix.dirname(cur));
        ensureDir(up);
        dirSubdirs.get(up)!.add(cur);
        cur = up;
      }
    }

    for (const [relDir, children] of dirFiles.entries()) {
      const subdirs = Array.from(dirSubdirs.get(relDir) ?? []).sort();
      const doc = renderDirectoryKnowledgeDocument({
        relDir,
        childFiles: children,
        childDirs: subdirs,
        updatedAt,
      });
      await atomicWriteFileWithRetry(
        path.join(this.config.knowledgeDir, knowledgeRelPathForDir(relDir)),
        doc,
      );
    }

    const rootIndex = [
      '---',
      'okf_version: "0.2"',
      'concept_id: "workspace:index"',
      'kind: "workspace_index"',
      `updated_at: "${updatedAt}"`,
      'generator: "local-deterministic-v1"',
      '---',
      '',
      '# AgentoCode Knowledge Index',
      '',
      `- **Indexed files tracked:** ${scanned.length}`,
      `- **Directories:** ${dirFiles.size}`,
      `- **Vector store fingerprint:** \`${this.fingerprint}\``,
      '',
      '## Tracked Files',
      '',
      ...scanned.map((f) => `- \`${f.relativePath}\` (\`${f.sha256.slice(0, 8)}\`)`),
      '',
    ].join('\n');

    await atomicWriteFileWithRetry(path.join(this.config.knowledgeDir, 'index.md'), rootIndex);
  }

  /**
   * Perform semantic search with live-hash freshness verification, falling back explicitly to lexical search
   * whenever the index or embedding model is unavailable.
   */
  async ragSearch(query: string, topK = 6): Promise<RagSearchResponse> {
    await this.initialize();
    const boundedTopK = Math.max(1, Math.min(20, Math.floor(topK)));
    const notices: string[] = [];
    let staleRejectedCount = 0;

    const semanticPossible =
      this.vectorStore !== null &&
      this.vectorStore.size > 0 &&
      this.embedder.info.provider !== 'lexical_only';

    let semanticResults: RagSearchHit[] = [];
    let semanticReady = false;

    if (semanticPossible && this.vectorStore) {
      const embeddedQuery = await this.embedder.embedQuery(query);
      if (embeddedQuery.ok) {
        semanticReady = true;
        const candidates = this.vectorStore.query(
          embeddedQuery.vector,
          boundedTopK * 3,
          this.config.maxDistance,
        );
        const seenKeys = new Set<string>();

        for (const cand of candidates) {
          // Verify freshness against live workspace source on disk.
          let isFresh = false;
          try {
            const contained = resolveContainedPath(this.config.workspaceRoot, cand.record.filePath);
            const buf = await fsp.readFile(contained.realPath);
            if (sha256Hex(buf) === cand.record.fileSha256) {
              isFresh = true;
            }
          } catch {
            isFresh = false;
          }

          if (!isFresh) {
            staleRejectedCount++;
            continue;
          }

          const dedupKey = `${cand.record.filePath}:${cand.record.lineStart ?? 0}-${cand.record.lineEnd ?? 0}`;
          if (seenKeys.has(dedupKey)) {
            continue;
          }
          seenKeys.add(dedupKey);

          semanticResults.push({
            conceptId: cand.record.conceptId,
            path: cand.record.filePath,
            kind: cand.record.kind,
            title: cand.record.title,
            distance: Number(cand.distance.toFixed(4)),
            retrievalMode: 'semantic',
            content: cand.record.content,
            lineStart: cand.record.lineStart,
            lineEnd: cand.record.lineEnd,
          });
          if (semanticResults.length >= boundedTopK) {
            break;
          }
        }
      } else {
        notices.push(`Semantic query fallback (${embeddedQuery.message}).`);
      }
    } else if (this.embedder.info.provider === 'lexical_only') {
      notices.push('Embedding provider is set to lexical_only; using lexical source fallback.');
    } else if (!this.vectorStore || this.vectorStore.size === 0) {
      notices.push('Vector index is currently empty; run Sync Knowledge to build local embeddings. Using lexical fallback.');
    }

    if (staleRejectedCount > 0) {
      notices.push(`Rejected ${staleRejectedCount} stale vector hit(s) whose source changed on disk; run Sync Knowledge.`);
    }

    if (semanticReady && semanticResults.length > 0) {
      return {
        query,
        retrievalMode: 'semantic',
        indexAvailable: true,
        candidateCount: semanticResults.length,
        staleRejectedCount,
        notices,
        results: semanticResults,
      };
    }

    // Exact / lexical fallback across live workspace files so answers stay accurate even unindexed.
    const lexicalHits = await this.lexicalSearchWorkspace(query, boundedTopK);
    if (semanticReady && semanticResults.length === 0 && lexicalHits.length > 0) {
      notices.push(`No semantic hits within maxDistance=${this.config.maxDistance}; supplemented with lexical matches.`);
    }

    return {
      query,
      retrievalMode: semanticReady ? 'hybrid' : 'lexical_fallback',
      indexAvailable: semanticReady,
      candidateCount: lexicalHits.length,
      staleRejectedCount,
      notices,
      results: lexicalHits,
    };
  }

  private async lexicalSearchWorkspace(query: string, topK: number): Promise<RagSearchHit[]> {
    const terms = Array.from(
      new Set(
        query
          .toLowerCase()
          .split(/[^a-z0-9_./-]+/)
          .filter((t) => t.length >= 2),
      ),
    );
    const scanned = await scanEligibleWorkspaceFiles(this.config.workspaceRoot);
    const scored: Array<{ score: number; hit: RagSearchHit }> = [];

    for (const file of scanned) {
      const lowerPath = file.relativePath.toLowerCase();
      const lines = file.text.split(/\r?\n/);
      let fileScore = 0;
      const matchingLines: number[] = [];

      for (const term of terms) {
        if (lowerPath.includes(term)) {
          fileScore += 4;
        }
      }

      for (let i = 0; i < lines.length; i++) {
        const lowerLine = lines[i].toLowerCase();
        let lineMatches = 0;
        for (const term of terms) {
          if (lowerLine.includes(term)) {
            lineMatches++;
          }
        }
        if (lineMatches > 0) {
          fileScore += lineMatches;
          if (matchingLines.length < 4) {
            matchingLines.push(i + 1);
          }
        }
      }

      if (fileScore > 0 || terms.length === 0) {
        const anchorLine = matchingLines[0] ?? 1;
        const start = Math.max(1, anchorLine - 5);
        const end = Math.min(lines.length, start + 39);
        scored.push({
          score: fileScore,
          hit: {
            conceptId: `lexical:${file.relativePath}#L${start}-${end}`,
            path: file.relativePath,
            kind: 'file_chunk',
            title: `${file.relativePath} (lexical L${start}-L${end})`,
            distance: null,
            retrievalMode: 'lexical_fallback',
            lineStart: start,
            lineEnd: end,
            content: lines.slice(start - 1, end).join('\n'),
          },
        });
      }
    }

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK).map((s) => s.hit);
  }

  async readConcept(conceptIdOrPath: string): Promise<string> {
    await this.initialize();
    const cleaned = conceptIdOrPath.trim();
    if (!cleaned) {
      throw new Error('Concept identifier or relative path is required.');
    }

    // Check vector metadata first if matching exact conceptId.
    if (this.vectorStore) {
      const rec = this.vectorStore.getRecord(cleaned);
      if (rec) {
        return `# ${rec.title}\n\nConcept ID: ${rec.conceptId}\nFile: ${rec.filePath}\nLines: ${rec.lineStart ?? 1}-${rec.lineEnd ?? 1}\n\n${rec.content}`;
      }
    }

    const strippedPath = cleaned.replace(/^file:/, '').split('#')[0];
    if (cleaned === 'workspace:index' || cleaned === 'index.md') {
      const idxPath = path.join(this.config.knowledgeDir, 'index.md');
      if (fs.existsSync(idxPath)) {
        return fsp.readFile(idxPath, 'utf8');
      }
    }

    const fileDocPath = path.join(this.config.knowledgeDir, knowledgeRelPathForFile(strippedPath));
    if (fs.existsSync(fileDocPath)) {
      return fsp.readFile(fileDocPath, 'utf8');
    }

    const dirDocPath = path.join(
      this.config.knowledgeDir,
      knowledgeRelPathForDir(cleaned.replace(/^dir:/, '')),
    );
    if (fs.existsSync(dirDocPath)) {
      return fsp.readFile(dirDocPath, 'utf8');
    }

    // Fallback to live structural analysis of the requested contained workspace file.
    const contained = resolveContainedPath(this.config.workspaceRoot, strippedPath);
    const stat = await fsp.stat(contained.realPath);
    const text = await fsp.readFile(contained.realPath, 'utf8');
    const analysis = analyzeSourceFile(contained.relativePath, text);
    return renderFileKnowledgeDocument({
      relativePath: contained.relativePath,
      sha256: sha256Hex(text),
      sizeBytes: stat.size,
      analysis,
      updatedAt: new Date().toISOString(),
      resolvedDependencies: [],
    });
  }

  async getStatus(): Promise<KnowledgeStatusSnapshot> {
    await this.initialize();
    const metrics = this.embedder.getMetrics();
    const entries = Object.values(this.manifest.files);
    const overlengthCount = entries.reduce((acc, e) => acc + e.overlengthChunks, 0);
    const failureCount =
      entries.filter((e) => e.status === 'error').length + this.recentFailures.length;
    const indexedVectors = this.vectorStore?.size ?? 0;

    return {
      modelState: this.embedder.getState(),
      embedProvider: this.embedder.info.provider,
      modelId: this.embedder.info.modelId,
      modelRevision: this.embedder.info.modelRevision,
      dtype: this.embedder.info.dtype,
      loadCount: metrics.loadCount,
      embedDocCalls: metrics.embedDocCalls,
      embedQueryCalls: metrics.embedQueryCalls,
      conceptCount: entries.length,
      indexedCount: indexedVectors,
      unindexedOverlengthCount: overlengthCount,
      failureCount,
      syncRunning: this.syncRunning,
      queuedCount: this.queuedCount,
      indexAvailable:
        this.embedder.info.provider !== 'lexical_only' &&
        indexedVectors > 0 &&
        this.embedder.getState() === 'ready',
      vectorFingerprint: this.fingerprint,
      lastSyncAt: this.manifest.lastSyncAt,
      lastError: metrics.lastError,
      workerPid: process.pid,
      recentFailures: this.recentFailures,
    };
  }

  async dispose(): Promise<void> {
    await this.embedder.dispose();
  }
}
