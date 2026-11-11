import fs from 'node:fs';
import path from 'node:path';
import * as vscode from 'vscode';
import { AgentLoop, AgentToolExecutor } from './agent/agentLoop.js';
import { ChatCompletionMessage, GroqHttpClient } from './agent/groqClient.js';
import {
  buildWorkerEnvironment,
  readAgentoConfig,
  resolveNodeRuntime,
  resolveStorageDirectories,
} from './config/settings.js';
import { resolveGroqApiKey } from './config/envLoader.js';
import { DiffProposalManager } from './editor/diffManager.js';
import { McpWorkerClient } from './mcp/workerClient.js';
import {
  AgentoExtensionConfig,
  ChatMessageItem,
  ChatStatePayload,
  KnowledgeStatusSnapshot,
} from './types.js';
import { ChatViewProvider, createChatMessage } from './views/chatViewProvider.js';
import { KnowledgeTreeItem, KnowledgeTreeProvider } from './views/knowledgeTreeProvider.js';
import {
  getActiveEditorSummary,
  getUnsavedBufferIfDirty,
  getWorkspaceDiagnosticsSummary,
} from './workspace/ideContext.js';
import { isRelPathEligible, normalizeRelPath, resolveContainedPath } from './worker/workspace.js';

const GROQ_SECRET_KEY = 'agentocode.groqApiKey';

function formatStatusReport(st: KnowledgeStatusSnapshot): string {
  const lines = [
    '**AgentoCode Local Index & Model Status**',
    `- **Provider / Model:** \`${st.embedProvider}\` / \`${st.modelId}\` (\`${st.dtype}\`)`,
    `- **Model State:** \`${st.modelState}\` (loads: ${st.loadCount}, worker PID: ${st.workerPid ?? 'n/a'})`,
    `- **Embedding Calls:** doc=${st.embedDocCalls}, query=${st.embedQueryCalls}`,
    `- **Concepts / Vectors:** ${st.conceptCount} tracked files, ${st.indexedCount} vectors`,
    `- **Overlength Exclusions / Failures:** overlength=${st.unindexedOverlengthCount}, failures=${st.failureCount}`,
    `- **Semantic Index Available:** ${st.indexAvailable} (fingerprint \`${st.vectorFingerprint}\`)`,
    `- **Sync State:** ${st.syncRunning ? 'running' : 'idle'} (queued: ${st.queuedCount}, lastSync: ${st.lastSyncAt ?? 'never'})`,
  ];
  if (st.lastError) {
    lines.push(`- **Last Error:** ${st.lastError}`);
  }
  if (st.recentFailures.length > 0) {
    lines.push('', '**Recent Issues:**');
    for (const f of st.recentFailures.slice(0, 8)) {
      lines.push(`- \`${f.path}\`: ${f.reason}`);
    }
  }
  return lines.join('\n');
}

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  const outputChannel = vscode.window.createOutputChannel('AgentoCode');
  const statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 50);
  statusBar.command = 'agentocode.showStatus';
  context.subscriptions.push(outputChannel, statusBar);

  const getWorkspaceRoot = (): string | undefined => vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
  const getConfig = (): AgentoExtensionConfig =>
    readAgentoConfig(vscode.workspace.getConfiguration('agentocode'));

  let latestStatus: KnowledgeStatusSnapshot | null = null;
  let chatMessages: ChatMessageItem[] = [];
  let modelHistory: ChatCompletionMessage[] = [];
  let activeAbortController: AbortController | null = null;
  let saveDebounceTimer: NodeJS.Timeout | undefined;

  const proposalManager = new DiffProposalManager(getWorkspaceRoot);
  context.subscriptions.push(proposalManager);

  const getStorageInfo = () => {
    const root = getWorkspaceRoot();
    if (!root) {
      return undefined;
    }
    return resolveStorageDirectories({
      workspaceRoot: root,
      extensionStoragePath: context.storageUri?.fsPath,
      config: getConfig(),
    });
  };

  const treeProvider = new KnowledgeTreeProvider(
    proposalManager,
    () => getStorageInfo()?.knowledgeDir,
  );
  context.subscriptions.push(
    vscode.window.registerTreeDataProvider('agentocode.knowledgeView', treeProvider),
  );

  const workerClient = new McpWorkerClient(
    () => {
      const workspaceRoot = getWorkspaceRoot();
      if (!workspaceRoot) {
        throw new Error('Open a workspace folder before running AgentoCode.');
      }
      if (!vscode.workspace.isTrusted) {
        throw new Error('Workspace is untrusted. Trust the workspace to start AgentoCode.');
      }
      const config = getConfig();
      const { stateDir, knowledgeDir } = resolveStorageDirectories({
        workspaceRoot,
        extensionStoragePath: context.storageUri?.fsPath,
        config,
      });
      const runtime = resolveNodeRuntime(config.nodePath);
      const workerEntry = path.join(context.extensionPath, 'dist', 'indexerWorker.cjs');
      if (!fs.existsSync(workerEntry)) {
        throw new Error(`Compiled worker bundle missing: ${workerEntry}`);
      }

      const env = buildWorkerEnvironment({
        workspaceRoot,
        stateDir,
        knowledgeDir,
        modelCacheDir: context.globalStorageUri?.fsPath
          ? path.join(context.globalStorageUri.fsPath, 'hf-cache')
          : undefined,
        config,
        baseEnv: {},
        extraEnv: runtime.envExtra,
      });

      return {
        command: runtime.command,
        args: [workerEntry],
        cwd: context.extensionPath,
        env,
      };
    },
    (line) => outputChannel.appendLine(`[worker] ${line}`),
    () => updateStatusBar(),
  );

  const getResolvedApiKey = async () =>
    resolveGroqApiKey({
      workspaceRoot: vscode.workspace.isTrusted ? getWorkspaceRoot() : undefined,
      extensionPath: context.extensionPath,
      getSecret: () => context.secrets.get(GROQ_SECRET_KEY),
    });

  const hasGroqApiKey = async (): Promise<boolean> => {
    const resolved = await getResolvedApiKey();
    return Boolean(resolved.apiKey);
  };

  const updateStatusBar = () => {
    if (!vscode.workspace.isTrusted) {
      statusBar.text = '$(shield) AgentoCode: Untrusted';
      statusBar.tooltip = 'Trust this workspace to enable AgentoCode.';
      statusBar.show();
      return;
    }
    if (!getWorkspaceRoot()) {
      statusBar.text = '$(circle-slash) AgentoCode: No Folder';
      statusBar.tooltip = 'Open a folder to use AgentoCode.';
      statusBar.show();
      return;
    }
    if (latestStatus?.syncRunning) {
      statusBar.text = '$(sync~spin) AgentoCode: Syncing...';
      statusBar.tooltip = 'Local AST & embedding synchronization in progress.';
    } else if (latestStatus?.indexAvailable) {
      statusBar.text = `$(database) AgentoCode: ${latestStatus.indexedCount} vectors`;
      statusBar.tooltip = `${latestStatus.embedProvider} (${latestStatus.modelId}) ready`;
    } else if (latestStatus?.modelState === 'error') {
      statusBar.text = '$(warning) AgentoCode: Lexical Fallback';
      statusBar.tooltip = latestStatus.lastError ?? 'Embedding model error; lexical fallback active.';
    } else if (latestStatus) {
      statusBar.text = `$(book) AgentoCode: ${latestStatus.conceptCount} files (Lexical)`;
      statusBar.tooltip = 'Run Sync Knowledge to build local embeddings.';
    } else {
      const conn = workerClient.getConnectionState();
      statusBar.text =
        conn === 'error' ? '$(error) AgentoCode: Worker Error' : '$(sparkle) AgentoCode';
      statusBar.tooltip = 'Click to inspect AgentoCode knowledge & model status.';
    }
    statusBar.show();
  };

  const refreshStatusAndViews = async (pushChat = true): Promise<KnowledgeStatusSnapshot | null> => {
    const apiKeyPresent = await hasGroqApiKey();
    if (vscode.workspace.isTrusted && getWorkspaceRoot()) {
      try {
        latestStatus = await workerClient.getStatus();
      } catch (err: unknown) {
        outputChannel.appendLine(
          `[host] Status fetch failed: ${err instanceof Error ? err.message : String(err)}`,
        );
      }
    }
    treeProvider.updateSnapshot({
      status: latestStatus,
      hasApiKey: apiKeyPresent,
      isTrusted: vscode.workspace.isTrusted,
    });
    updateStatusBar();
    if (pushChat) {
      await chatView.pushFullState();
    }
    return latestStatus;
  };

  const toolExecutor: AgentToolExecutor = {
    listMcpTools: () => workerClient.listTools(),
    callMcpTool: async (name, args) => {
      const workspaceRoot = getWorkspaceRoot();
      if (name === 'read_file' && workspaceRoot && typeof args.path === 'string') {
        const contained = resolveContainedPath(workspaceRoot, args.path);
        const dirtyText = getUnsavedBufferIfDirty(contained.realPath);
        if (dirtyText !== null) {
          const lines = dirtyText.split(/\r?\n/);
          const start = Math.max(1, typeof args.start_line === 'number' ? args.start_line : 1);
          const end = Math.min(
            lines.length,
            typeof args.end_line === 'number' ? args.end_line : Math.min(lines.length, start + 249),
          );
          const slice = lines
            .slice(start - 1, end)
            .map((l, i) => `${start + i}: ${l}`)
            .join('\n');
          return `File (unsaved editor buffer): ${contained.relativePath} (lines ${start}-${end} of ${lines.length})\n\n${slice}`;
        }
      }
      return workerClient.callToolText(name, args);
    },
    getActiveContext: async () => getActiveEditorSummary(getWorkspaceRoot()).formattedContext,
    getDiagnostics: async (relPath) => getWorkspaceDiagnosticsSummary(getWorkspaceRoot(), relPath),
    proposeFileEdit: async ({ path: targetPath, content, rationale }) => {
      const proposal = await proposalManager.createProposal({
        relativePath: targetPath,
        proposedContent: content,
        rationale,
      });
      return {
        proposal,
        summary: `Created pending edit proposal ${proposal.id} for ${proposal.relativePath} (${proposal.isNewFile ? 'new file' : 'update'}). Inform the user to review the diff and click Apply.`,
      };
    },
  };

  const agentLoop = new AgentLoop(new GroqHttpClient(), toolExecutor);

  const runSync = async (force = false): Promise<KnowledgeStatusSnapshot> => {
    if (!vscode.workspace.isTrusted) {
      throw new Error('Cannot sync in an untrusted workspace.');
    }
    if (latestStatus) {
      latestStatus = { ...latestStatus, syncRunning: true };
      updateStatusBar();
      chatView.postStatusUpdate(latestStatus);
    }
    latestStatus = await workerClient.syncKnowledge(force);
    await refreshStatusAndViews(true);
    return latestStatus;
  };

  const setApiKeyFlow = async () => {
    const key = await vscode.window.showInputBox({
      title: 'AgentoCode: Groq API Key',
      prompt: 'Enter your Groq API key (stored securely in VS Code SecretStorage, used only when querying).',
      password: true,
      ignoreFocusOut: true,
      placeHolder: 'gsk_...',
    });
    if (key === undefined) {
      return;
    }
    const trimmed = key.trim();
    if (!trimmed) {
      vscode.window.showWarningMessage('Empty Groq API key was not saved.');
      return;
    }
    await context.secrets.store(GROQ_SECRET_KEY, trimmed);
    vscode.window.showInformationMessage('AgentoCode Groq API key saved to VS Code SecretStorage.');
    await refreshStatusAndViews(true);
  };

  const getChatStateSnapshot = async (): Promise<ChatStatePayload> => {
    const root = getWorkspaceRoot();
    return {
      messages: chatMessages,
      proposals: proposalManager.getAllProposals(),
      hasApiKey: await hasGroqApiKey(),
      isTrusted: vscode.workspace.isTrusted,
      isRunning: activeAbortController !== null,
      status: latestStatus,
      activeFile: getActiveEditorSummary(root).relativePath,
      generationModel: getConfig().generationModel,
    };
  };

  const handleChatPrompt = async (rawText: string, includeActiveFile: boolean): Promise<void> => {
    const text = rawText.trim();
    if (!text) {
      return;
    }

    if (text === '/clear') {
      chatMessages = [];
      modelHistory = [];
      await chatView.pushFullState();
      return;
    }
    if (text === '/status') {
      const st = await refreshStatusAndViews(false);
      chatMessages.push(
        createChatMessage('system', st ? formatStatusReport(st) : 'Worker status unavailable.'),
      );
      await chatView.pushFullState();
      return;
    }
    if (text === '/sync' || text === '/sync --force') {
      chatMessages.push(createChatMessage('user', text));
      const sysMsg = createChatMessage('system', 'Running local incremental indexing (0 Groq calls)...');
      chatMessages.push(sysMsg);
      await chatView.pushFullState();
      try {
        const st = await runSync(text.includes('--force'));
        sysMsg.content = formatStatusReport(st);
      } catch (err: unknown) {
        sysMsg.error = err instanceof Error ? err.message : String(err);
      }
      await chatView.pushFullState();
      return;
    }
    if (text === '/preload') {
      chatMessages.push(createChatMessage('user', text));
      const sysMsg = createChatMessage('system', 'Preloading local Nomic embedding model...');
      chatMessages.push(sysMsg);
      await chatView.pushFullState();
      try {
        latestStatus = await workerClient.preloadModel();
        await refreshStatusAndViews(false);
        sysMsg.content = formatStatusReport(latestStatus);
      } catch (err: unknown) {
        sysMsg.error = err instanceof Error ? err.message : String(err);
      }
      await chatView.pushFullState();
      return;
    }
    if (text === '/help') {
      chatMessages.push(
        createChatMessage(
          'system',
          'Available commands: `/sync`, `/sync --force`, `/status`, `/preload`, `/clear`, `/help`.',
        ),
      );
      await chatView.pushFullState();
      return;
    }

    if (!vscode.workspace.isTrusted) {
      vscode.window.showWarningMessage('Trust this workspace before running AgentoCode.');
      return;
    }
    if (!getWorkspaceRoot()) {
      vscode.window.showWarningMessage('Open a workspace folder first.');
      return;
    }

    const resolvedKey = await getResolvedApiKey();
    const apiKey = resolvedKey.apiKey;
    if (!apiKey) {
      const choice = await vscode.window.showWarningMessage(
        'GROQ_API_KEY is not set in .env, environment variables, or SecretStorage.',
        'Set Groq API Key',
      );
      if (choice === 'Set Groq API Key') {
        await setApiKeyFlow();
      }
      return;
    }

    const userMsg = createChatMessage('user', text);
    const assistantMsg = createChatMessage('assistant', '');
    chatMessages.push(userMsg, assistantMsg);
    activeAbortController = new AbortController();
    await chatView.pushFullState();

    try {
      const config = getConfig();
      const activeSummary = includeActiveFile ? getActiveEditorSummary(getWorkspaceRoot()).formattedContext : undefined;

      const result = await agentLoop.runTurn({
        apiKey,
        model: config.generationModel,
        history: modelHistory,
        userPrompt: text,
        activeContextPrefix: activeSummary
          ? `[Current Editor Context]\n${activeSummary}`
          : undefined,
        maxIterations: config.maxAgentIterations,
        signal: activeAbortController.signal,
        callbacks: {
          onDelta: (delta) => {
            assistantMsg.content += delta;
            chatView.postStreamDelta(assistantMsg.id, delta);
          },
          onToolStart: (trace) => {
            assistantMsg.tools = assistantMsg.tools ?? [];
            assistantMsg.tools.push(trace);
            chatView.postToolUpdate(assistantMsg.id, trace);
          },
          onToolDone: (trace) => {
            assistantMsg.tools = assistantMsg.tools ?? [];
            const idx = assistantMsg.tools.findIndex((t) => t.id === trace.id);
            if (idx >= 0) {
              assistantMsg.tools[idx] = trace;
            }
            chatView.postToolUpdate(assistantMsg.id, trace);
          },
          onProposalCreated: (proposal) => {
            assistantMsg.proposalIds = assistantMsg.proposalIds ?? [];
            assistantMsg.proposalIds.push(proposal.id);
            chatView.postProposalUpdate(proposal, assistantMsg.id);
            treeProvider.refresh();
          },
        },
      });

      modelHistory = result.updatedHistory.slice(-24);
      activeAbortController = null;
      chatView.postRunFinished(assistantMsg.id);
      await refreshStatusAndViews(true);
    } catch (err: unknown) {
      const errMsg = err instanceof Error ? err.message : String(err);
      assistantMsg.error = errMsg;
      activeAbortController = null;
      chatView.postRunFinished(assistantMsg.id, errMsg);
      await chatView.pushFullState();
    }
  };

  const chatView = new ChatViewProvider(context.extensionUri, {
    onSendPrompt: handleChatPrompt,
    onCancelRun: () => {
      activeAbortController?.abort();
      activeAbortController = null;
    },
    onClearChat: () => {
      chatMessages = [];
      modelHistory = [];
    },
    onSyncKnowledge: async () => {
      try {
        await runSync(false);
      } catch (err: unknown) {
        vscode.window.showErrorMessage(`Sync failed: ${err instanceof Error ? err.message : String(err)}`);
      }
    },
    onShowStatus: async () => {
      const st = await refreshStatusAndViews(true);
      if (st) {
        chatMessages.push(createChatMessage('system', formatStatusReport(st)));
        await chatView.pushFullState();
      }
    },
    onPreloadModel: async () => {
      try {
        latestStatus = await workerClient.preloadModel();
        await refreshStatusAndViews(true);
        vscode.window.showInformationMessage(`Local embedding model ready (${latestStatus.modelId}).`);
      } catch (err: unknown) {
        vscode.window.showErrorMessage(`Model load failed: ${err instanceof Error ? err.message : String(err)}`);
      }
    },
    onSetApiKey: setApiKeyFlow,
    onOpenLogs: () => outputChannel.show(true),
    onPreviewProposal: async (id) => proposalManager.previewProposal(id),
    onApplyProposal: async (id) => {
      await proposalManager.applyProposal(id);
      await chatView.pushFullState();
    },
    onRejectProposal: (id) => {
      proposalManager.rejectProposal(id);
      void chatView.pushFullState();
    },
    getStateSnapshot: getChatStateSnapshot,
  });

  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(ChatViewProvider.viewType, chatView, {
      webviewOptions: { retainContextWhenHidden: true },
    }),
  );

  // Register commands.
  const resolveProposalIdArg = (arg?: string | KnowledgeTreeItem): string | undefined => {
    if (typeof arg === 'string') {
      return arg;
    }
    return arg?.proposalId;
  };

  context.subscriptions.push(
    vscode.commands.registerCommand('agentocode.setApiKey', setApiKeyFlow),
    vscode.commands.registerCommand('agentocode.clearApiKey', async () => {
      await context.secrets.delete(GROQ_SECRET_KEY);
      vscode.window.showInformationMessage('Cleared stored AgentoCode Groq API key.');
      await refreshStatusAndViews(true);
    }),
    vscode.commands.registerCommand('agentocode.syncKnowledge', async () => {
      try {
        const st = await runSync(false);
        vscode.window.showInformationMessage(
          `AgentoCode synced ${st.conceptCount} file(s), ${st.indexedCount} vector(s).`,
        );
      } catch (err: unknown) {
        vscode.window.showErrorMessage(`Sync failed: ${err instanceof Error ? err.message : String(err)}`);
      }
    }),
    vscode.commands.registerCommand('agentocode.showStatus', async () => {
      const st = await refreshStatusAndViews(true);
      if (st) {
        outputChannel.appendLine(formatStatusReport(st));
        vscode.window.showInformationMessage(
          `AgentoCode: ${st.modelId} [${st.modelState}] | ${st.indexedCount} vectors | ${st.conceptCount} files | failures=${st.failureCount}`,
        );
      }
    }),
    vscode.commands.registerCommand('agentocode.clearChat', async () => {
      chatMessages = [];
      modelHistory = [];
      await chatView.pushFullState();
    }),
    vscode.commands.registerCommand('agentocode.restartWorker', async () => {
      await workerClient.restart();
      await refreshStatusAndViews(true);
      vscode.window.showInformationMessage('AgentoCode local worker restarted.');
    }),
    vscode.commands.registerCommand('agentocode.openLogs', () => outputChannel.show(true)),
    vscode.commands.registerCommand('agentocode.preloadModel', async () => {
      try {
        latestStatus = await workerClient.preloadModel();
        await refreshStatusAndViews(true);
        vscode.window.showInformationMessage(`Local embedding model loaded (${latestStatus.modelId}).`);
      } catch (err: unknown) {
        vscode.window.showErrorMessage(`Model load failed: ${err instanceof Error ? err.message : String(err)}`);
      }
    }),
    vscode.commands.registerCommand('agentocode.previewProposal', async (arg?: string | KnowledgeTreeItem) => {
      const id = resolveProposalIdArg(arg);
      if (id) {
        await proposalManager.previewProposal(id);
      }
    }),
    vscode.commands.registerCommand('agentocode.applyProposal', async (arg?: string | KnowledgeTreeItem) => {
      const id = resolveProposalIdArg(arg);
      if (id) {
        await proposalManager.applyProposal(id);
        await chatView.pushFullState();
      }
    }),
    vscode.commands.registerCommand('agentocode.rejectProposal', async (arg?: string | KnowledgeTreeItem) => {
      const id = resolveProposalIdArg(arg);
      if (id) {
        proposalManager.rejectProposal(id);
        await chatView.pushFullState();
      }
    }),
  );

  // Sync UI state when active editor, workspace trust, or configuration changes.
  context.subscriptions.push(
    vscode.window.onDidChangeActiveTextEditor(() => {
      void chatView.pushFullState();
    }),
    vscode.workspace.onDidGrantWorkspaceTrust(async () => {
      await refreshStatusAndViews(true);
      if (getConfig().autoSyncOnStartup) {
        await runSync(false).catch(() => undefined);
      }
    }),
    vscode.workspace.onDidChangeConfiguration(async (ev) => {
      if (!ev.affectsConfiguration('agentocode')) {
        return;
      }
      const workerKeys = [
        'embedProvider',
        'embedModelId',
        'embedModelRevision',
        'embedDtype',
        'ollamaEndpoint',
        'localFilesOnly',
        'maxTokens',
        'maxDistance',
        'stateLocation',
        'writeKnowledgeDocs',
        'knowledgeDocsInWorkspace',
        'nodePath',
      ];
      if (workerKeys.some((k) => ev.affectsConfiguration(`agentocode.${k}`))) {
        latestStatus = null;
        await workerClient.dispose();
      }
      await refreshStatusAndViews(true);
    }),
    vscode.workspace.onDidSaveTextDocument((doc) => {
      const root = getWorkspaceRoot();
      if (!root || !vscode.workspace.isTrusted) {
        return;
      }
      if (doc.uri.scheme === 'file' && path.basename(doc.uri.fsPath) === '.env') {
        void refreshStatusAndViews(true);
        return;
      }
      if (!getConfig().autoSyncOnSave) {
        return;
      }
      if (doc.uri.scheme !== 'file') {
        return;
      }
      const rel = normalizeRelPath(path.relative(root, doc.uri.fsPath));
      if (!rel || rel.startsWith('..') || !isRelPathEligible(rel)) {
        return;
      }
      if (saveDebounceTimer) {
        clearTimeout(saveDebounceTimer);
      }
      saveDebounceTimer = setTimeout(() => {
        void runSync(false).catch((err) => {
          outputChannel.appendLine(`[host] Auto-sync on save failed: ${err instanceof Error ? err.message : String(err)}`);
        });
      }, 1500);
    }),
    {
      dispose: () => {
        if (saveDebounceTimer) {
          clearTimeout(saveDebounceTimer);
        }
        void workerClient.dispose();
      },
    },
  );

  await refreshStatusAndViews(true);
  if (vscode.workspace.isTrusted && getWorkspaceRoot() && getConfig().autoSyncOnStartup) {
    void runSync(false).catch((err) => {
      outputChannel.appendLine(`[host] Startup auto-sync failed: ${err instanceof Error ? err.message : String(err)}`);
    });
  }
}

export function deactivate(): void {
  // Subscriptions in ExtensionContext handle worker disposal cleanly.
}

