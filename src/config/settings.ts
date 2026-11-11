import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { AgentoExtensionConfig, EmbedDtype, EmbedProviderKind, StateLocationMode } from '../types.js';

export const DEFAULT_CONFIG: AgentoExtensionConfig = {
  generationModel: 'openai/gpt-oss-120b',
  embedProvider: 'transformers_onnx',
  embedModelId: 'nomic-ai/nomic-embed-text-v1.5',
  embedModelRevision: 'e9b6763023c676ca8431644204f50c2b100d9aab',
  embedDtype: 'q8',
  ollamaEndpoint: 'http://127.0.0.1:11434',
  localFilesOnly: false,
  maxTokens: 2048,
  maxDistance: 0.65,
  stateLocation: 'extensionStorage',
  writeKnowledgeDocs: true,
  knowledgeDocsInWorkspace: false,
  nodePath: '',
  autoSyncOnStartup: false,
  autoSyncOnSave: false,
  maxAgentIterations: 12,
};

export interface ConfigurationReader {
  get<T>(section: string, defaultValue: T): T;
}

export function readAgentoConfig(reader: ConfigurationReader): AgentoExtensionConfig {
  return {
    generationModel: reader.get<string>('generationModel', DEFAULT_CONFIG.generationModel),
    embedProvider: reader.get<EmbedProviderKind>('embedProvider', DEFAULT_CONFIG.embedProvider),
    embedModelId: reader.get<string>('embedModelId', DEFAULT_CONFIG.embedModelId),
    embedModelRevision: reader.get<string>('embedModelRevision', DEFAULT_CONFIG.embedModelRevision),
    embedDtype: reader.get<EmbedDtype>('embedDtype', DEFAULT_CONFIG.embedDtype),
    ollamaEndpoint: reader.get<string>('ollamaEndpoint', DEFAULT_CONFIG.ollamaEndpoint),
    localFilesOnly: reader.get<boolean>('localFilesOnly', DEFAULT_CONFIG.localFilesOnly),
    maxTokens: reader.get<number>('maxTokens', DEFAULT_CONFIG.maxTokens),
    maxDistance: reader.get<number>('maxDistance', DEFAULT_CONFIG.maxDistance),
    stateLocation: reader.get<StateLocationMode>('stateLocation', DEFAULT_CONFIG.stateLocation),
    writeKnowledgeDocs: reader.get<boolean>('writeKnowledgeDocs', DEFAULT_CONFIG.writeKnowledgeDocs),
    knowledgeDocsInWorkspace: reader.get<boolean>(
      'knowledgeDocsInWorkspace',
      DEFAULT_CONFIG.knowledgeDocsInWorkspace,
    ),
    nodePath: reader.get<string>('nodePath', DEFAULT_CONFIG.nodePath).trim(),
    autoSyncOnStartup: reader.get<boolean>('autoSyncOnStartup', DEFAULT_CONFIG.autoSyncOnStartup),
    autoSyncOnSave: reader.get<boolean>('autoSyncOnSave', DEFAULT_CONFIG.autoSyncOnSave),
    maxAgentIterations: reader.get<number>('maxAgentIterations', DEFAULT_CONFIG.maxAgentIterations),
  };
}

/**
 * Resolve state directory and knowledge directory without polluting tracked workspace git trees
 * unless explicitly requested by user configuration.
 */
export function resolveStorageDirectories(params: {
  workspaceRoot: string;
  extensionStoragePath?: string;
  config: AgentoExtensionConfig;
}): { stateDir: string; knowledgeDir: string } {
  const { workspaceRoot, extensionStoragePath, config } = params;
  const stateDir =
    config.stateLocation === 'workspace' || !extensionStoragePath
      ? path.join(workspaceRoot, '.guardian')
      : path.join(extensionStoragePath, 'guardian');

  const knowledgeDir = config.knowledgeDocsInWorkspace
    ? path.join(workspaceRoot, 'knowledge_base')
    : path.join(stateDir, 'knowledge_base');

  return { stateDir, knowledgeDir };
}

/**
 * Resolve Node executable for running the local worker outside VS Code's Electron ABI when possible.
 */
export function resolveNodeRuntime(configuredNodePath: string): {
  command: string;
  envExtra: Record<string, string>;
} {
  if (configuredNodePath) {
    if (!fs.existsSync(configuredNodePath)) {
      throw new Error(`Configured agentocode.nodePath does not exist: ${configuredNodePath}`);
    }
    return { command: configuredNodePath, envExtra: {} };
  }

  // Prefer standalone system node on PATH so native onnxruntime-node matches Node ABI cleanly.
  try {
    const probe = process.platform === 'win32' ? 'where.exe' : 'which';
    const out = execFileSync(probe, ['node'], { encoding: 'utf8', timeout: 3000 }).trim();
    const firstMatch = out.split(/\r?\n/).find((line) => line.trim().length > 0);
    if (firstMatch && fs.existsSync(firstMatch.trim())) {
      return { command: firstMatch.trim(), envExtra: {} };
    }
  } catch {
    // Fallback below.
  }

  // Last-resort fallback inside VS Code Extension Host: run Electron executable in Node mode.
  return {
    command: process.execPath,
    envExtra: { ELECTRON_RUN_AS_NODE: '1' },
  };
}

export function buildWorkerEnvironment(params: {
  workspaceRoot: string;
  stateDir: string;
  knowledgeDir: string;
  modelCacheDir?: string;
  config: AgentoExtensionConfig;
  baseEnv: Record<string, string>;
  extraEnv?: Record<string, string>;
}): Record<string, string> {
  const { workspaceRoot, stateDir, knowledgeDir, modelCacheDir, config, baseEnv, extraEnv } = params;
  return {
    ...baseEnv,
    ...(extraEnv ?? {}),
    AGENTOCODE_WORKSPACE_ROOT: workspaceRoot,
    AGENTOCODE_STATE_DIR: stateDir,
    AGENTOCODE_KNOWLEDGE_DIR: knowledgeDir,
    ...(modelCacheDir ? { AGENTOCODE_MODEL_CACHE_DIR: modelCacheDir } : {}),
    AGENTOCODE_WRITE_KNOWLEDGE_DOCS: config.writeKnowledgeDocs ? '1' : '0',
    AGENTOCODE_EMBED_PROVIDER: config.embedProvider,
    AGENTOCODE_EMBED_MODEL_ID: config.embedModelId,
    AGENTOCODE_EMBED_MODEL_REVISION: config.embedModelRevision,
    AGENTOCODE_EMBED_DTYPE: config.embedDtype,
    AGENTOCODE_OLLAMA_ENDPOINT: config.ollamaEndpoint,
    GUARDIAN_LOCAL_FILES_ONLY: config.localFilesOnly ? '1' : '0',
    AGENTOCODE_MAX_TOKENS: String(config.maxTokens),
    GUARDIAN_MAX_DISTANCE: String(config.maxDistance),
  };
}

