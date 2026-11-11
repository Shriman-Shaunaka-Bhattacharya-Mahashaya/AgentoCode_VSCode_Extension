export type EmbedProviderKind = 'transformers_onnx' | 'ollama' | 'lexical_only';

export type ModelLifecycleState = 'unloaded' | 'loading' | 'ready' | 'error';

export type RetrievalMode = 'semantic' | 'lexical_fallback' | 'hybrid';

export type StateLocationMode = 'extensionStorage' | 'workspace';

export type EmbedDtype = 'q8' | 'fp32' | 'fp16' | 'q4';

export interface WorkerConfig {
  workspaceRoot: string;
  stateDir: string;
  knowledgeDir: string;
  modelCacheDir?: string;
  writeKnowledgeDocs: boolean;
  embedProvider: EmbedProviderKind;
  embedModelId: string;
  embedModelRevision: string;
  embedDtype: EmbedDtype;
  ollamaEndpoint: string;
  localFilesOnly: boolean;
  maxTokens: number;
  maxDistance: number;
}

export interface AgentoExtensionConfig {
  generationModel: string;
  embedProvider: EmbedProviderKind;
  embedModelId: string;
  embedModelRevision: string;
  embedDtype: EmbedDtype;
  ollamaEndpoint: string;
  localFilesOnly: boolean;
  maxTokens: number;
  maxDistance: number;
  stateLocation: StateLocationMode;
  writeKnowledgeDocs: boolean;
  knowledgeDocsInWorkspace: boolean;
  nodePath: string;
  autoSyncOnStartup: boolean;
  autoSyncOnSave: boolean;
  maxAgentIterations: number;
}

export interface KnowledgeStatusSnapshot {
  modelState: ModelLifecycleState;
  embedProvider: EmbedProviderKind;
  modelId: string;
  modelRevision: string;
  dtype: string;
  loadCount: number;
  embedDocCalls: number;
  embedQueryCalls: number;
  conceptCount: number;
  indexedCount: number;
  unindexedOverlengthCount: number;
  failureCount: number;
  syncRunning: boolean;
  queuedCount: number;
  indexAvailable: boolean;
  vectorFingerprint: string;
  lastSyncAt: string | null;
  lastError: string | null;
  workerPid: number | null;
  recentFailures: Array<{ path: string; reason: string }>;
}

export interface RagSearchHit {
  conceptId: string;
  path: string;
  kind: 'file_chunk' | 'directory' | 'workspace_index';
  title: string;
  distance: number | null;
  retrievalMode: RetrievalMode;
  content: string;
  lineStart?: number;
  lineEnd?: number;
}

export interface RagSearchResponse {
  query: string;
  retrievalMode: RetrievalMode;
  indexAvailable: boolean;
  candidateCount: number;
  staleRejectedCount: number;
  notices: string[];
  results: RagSearchHit[];
}

export type ProposalStatus = 'pending' | 'applied' | 'rejected';

export interface EditProposal {
  id: string;
  relativePath: string;
  uriString: string;
  originalContent: string;
  proposedContent: string;
  isNewFile: boolean;
  rationale: string;
  status: ProposalStatus;
  createdAt: string;
}

export interface ToolTrace {
  id: string;
  name: string;
  argsPreview: string;
  resultPreview?: string;
  status: 'running' | 'done' | 'error';
}

export interface ChatMessageItem {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  tools?: ToolTrace[];
  proposalIds?: string[];
  error?: string;
}

export interface ChatStatePayload {
  messages: ChatMessageItem[];
  proposals: Record<string, EditProposal>;
  hasApiKey: boolean;
  isTrusted: boolean;
  isRunning: boolean;
  status: KnowledgeStatusSnapshot | null;
  activeFile: string | null;
  generationModel: string;
}

export type WebviewToExtensionMessage =
  | { type: 'ready' }
  | { type: 'sendPrompt'; text: string; includeActiveFile: boolean }
  | { type: 'cancelRun' }
  | { type: 'clearChat' }
  | { type: 'setApiKey' }
  | { type: 'syncKnowledge' }
  | { type: 'showStatus' }
  | { type: 'preloadModel' }
  | { type: 'openLogs' }
  | { type: 'previewProposal'; id: string }
  | { type: 'applyProposal'; id: string }
  | { type: 'rejectProposal'; id: string };

export type ExtensionToWebviewMessage =
  | { type: 'state'; payload: ChatStatePayload }
  | { type: 'streamDelta'; messageId: string; delta: string }
  | { type: 'toolUpdate'; messageId: string; trace: ToolTrace }
  | { type: 'proposalUpdate'; messageId?: string; proposal: EditProposal }
  | { type: 'statusUpdate'; status: KnowledgeStatusSnapshot }
  | { type: 'runFinished'; messageId: string; error?: string };

