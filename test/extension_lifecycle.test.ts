import assert from 'node:assert/strict';
import fsp from 'node:fs/promises';
import Module from 'node:module';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

function createLifecycleMockVscode(wsRoot: string, configOverrides: Record<string, unknown> = {}) {
  const registeredCommands = new Map<string, (...args: unknown[]) => unknown>();
  const webviewProviders = new Map<string, { resolveWebviewView(view: unknown): void }>();
  const treeProviders = new Map<string, unknown>();
  const configListeners: Array<(ev: { affectsConfiguration(section: string): boolean }) => Promise<void> | void> = [];
  const saveListeners: Array<(doc: { uri: { scheme: string; fsPath: string } }) => void> = [];
  const executedCommands: Array<{ cmd: string; args: unknown[] }> = [];
  const appliedEditsCount = { value: 0 };
  const infoMessages: string[] = [];
  const warningMessages: string[] = [];
  const errorMessages: string[] = [];
  const outputLines: string[] = [];
  const secretsStore = new Map<string, string>();
  let nextInputBoxValue: string | undefined = 'gsk_lifecycle_key';

  class EventEmitter<T = void> {
    private listeners: Array<(e: T) => void> = [];
    event = (fn: (e: T) => void) => {
      this.listeners.push(fn);
      return { dispose: () => undefined };
    };
    fire(data: T) {
      for (const l of this.listeners) {
        l(data);
      }
    }
    dispose() {
      this.listeners = [];
    }
  }

  class TreeItem {
    constructor(readonly label: string, readonly collapsibleState = 0) {}
  }

  class ThemeIcon {
    constructor(readonly id: string) {}
  }

  class Position {
    constructor(readonly line: number, readonly character: number) {}
  }

  class Range {
    constructor(
      readonly startLine: number,
      readonly startChar: number,
      readonly endLine: number,
      readonly endChar: number,
    ) {}
  }

  class WorkspaceEdit {
    createFile() {}
    insert() {}
    replace() {}
  }

  const makeUri = (fsPathVal: string) => ({
    scheme: 'file',
    fsPath: fsPathVal,
    path: fsPathVal,
    authority: '',
    toString: () => `file://${fsPathVal.replace(/\\/g, '/')}`,
  });

  const vscodeMock = {
    EventEmitter,
    TreeItem,
    TreeItemCollapsibleState: { None: 0, Collapsed: 1, Expanded: 2 },
    ThemeIcon,
    Position,
    Range,
    WorkspaceEdit,
    DiagnosticSeverity: { Error: 0, Warning: 1, Information: 2, Hint: 3 },
    StatusBarAlignment: { Left: 1, Right: 2 },
    Uri: {
      file: (p: string) => makeUri(path.resolve(p)),
      joinPath: (base: { fsPath: string }, ...segs: string[]) => makeUri(path.join(base.fsPath, ...segs)),
      parse: (raw: string) => makeUri(raw),
    },
    workspace: {
      isTrusted: true,
      workspaceFolders: [{ uri: makeUri(wsRoot) }],
      textDocuments: [],
      getConfiguration: () => ({
        get: <T>(key: string, def: T): T =>
          key in configOverrides ? (configOverrides[key] as T) : def,
      }),
      registerTextDocumentContentProvider: () => ({ dispose: () => undefined }),
      onDidGrantWorkspaceTrust: () => ({ dispose: () => undefined }),
      onDidChangeConfiguration: (fn: (ev: { affectsConfiguration(section: string): boolean }) => Promise<void> | void) => {
        configListeners.push(fn);
        return { dispose: () => undefined };
      },
      onDidSaveTextDocument: (fn: (doc: { uri: { scheme: string; fsPath: string } }) => void) => {
        saveListeners.push(fn);
        return { dispose: () => undefined };
      },
      openTextDocument: async (uri: { fsPath: string }) => {
        const fs = await import('node:fs');
        const text = fs.existsSync(uri.fsPath) ? fs.readFileSync(uri.fsPath, 'utf8') : '';
        const lines = text.split(/\r?\n/);
        return {
          uri,
          lineCount: lines.length,
          getText: () => text,
          lineAt: (idx: number) => ({ text: lines[idx] ?? '' }),
        };
      },
      applyEdit: async () => {
        appliedEditsCount.value++;
        return true;
      },
    },
    window: {
      activeTextEditor: undefined,
      createOutputChannel: () => ({
        appendLine: (line: string) => outputLines.push(line),
        show: () => undefined,
        dispose: () => undefined,
      }),
      createStatusBarItem: () => ({
        text: '',
        tooltip: '',
        command: '',
        show: () => undefined,
        dispose: () => undefined,
      }),
      registerTreeDataProvider: (id: string, provider: unknown) => {
        treeProviders.set(id, provider);
        return { dispose: () => undefined };
      },
      registerWebviewViewProvider: (id: string, provider: { resolveWebviewView(view: unknown): void }) => {
        webviewProviders.set(id, provider);
        return { dispose: () => undefined };
      },
      onDidChangeActiveTextEditor: () => ({ dispose: () => undefined }),
      showInformationMessage: (msg: string) => {
        infoMessages.push(msg);
        return Promise.resolve(undefined);
      },
      showWarningMessage: (msg: string) => {
        warningMessages.push(msg);
        return Promise.resolve(undefined);
      },
      showErrorMessage: (msg: string) => {
        errorMessages.push(msg);
        return Promise.resolve(undefined);
      },
      showInputBox: async () => nextInputBoxValue,
      showTextDocument: async () => undefined,
    },
    languages: {
      getDiagnostics: () => [],
    },
    commands: {
      registerCommand: (id: string, fn: (...args: unknown[]) => unknown) => {
        registeredCommands.set(id, fn);
        return { dispose: () => undefined };
      },
      executeCommand: async (cmd: string, ...args: unknown[]) => {
        executedCommands.push({ cmd, args });
      },
    },
  };

  return {
    vscodeMock,
    registeredCommands,
    webviewProviders,
    treeProviders,
    infoMessages,
    errorMessages,
    outputLines,
    secretsStore,
    configListeners,
    saveListeners,
    executedCommands,
    appliedEditsCount,
    setNextInputBoxValue(v: string | undefined) {
      nextInputBoxValue = v;
    },
  };
}

test('extension activate/deactivate wires commands, SecretStorage, ChatView CSP, and slash commands cleanly', async () => {
  const ws = await fsp.mkdtemp(path.join(os.tmpdir(), 'agentocode-ext-life-'));
  const extStorage = path.join(ws, '.ext-store');
  const globalStorage = path.join(ws, '.global-store');

  await fsp.writeFile(path.join(ws, 'calc.ts'), 'export function add(a: number, b: number) { return a + b; }\n', 'utf8');

  const harness = createLifecycleMockVscode(ws, {
    embedProvider: 'lexical_only',
    stateLocation: 'extensionStorage',
  });

  const modAny = Module as unknown as {
    _load: (request: string, parent: unknown, isMain: boolean) => unknown;
  };
  const originalLoad = modAny._load;
  modAny._load = function (request: string, parent: unknown, isMain: boolean) {
    if (request === 'vscode') {
      return harness.vscodeMock;
    }
    return originalLoad.call(this, request, parent, isMain);
  };

  const subscriptions: Array<{ dispose(): void }> = [];
  const ext = await import('../src/extension.js');
  try {
    const mockContext = {
      extensionPath: process.cwd(),
      extensionUri: harness.vscodeMock.Uri.file(process.cwd()),
      storageUri: harness.vscodeMock.Uri.file(extStorage),
      globalStorageUri: harness.vscodeMock.Uri.file(globalStorage),
      subscriptions,
      secrets: {
        get: async (k: string) => harness.secretsStore.get(k),
        store: async (k: string, v: string) => {
          harness.secretsStore.set(k, v);
        },
        delete: async (k: string) => {
          harness.secretsStore.delete(k);
        },
      },
    };

    await ext.activate(mockContext as never);

    // Every expected command & view provider is registered.
    for (const cmd of [
      'agentocode.setApiKey',
      'agentocode.clearApiKey',
      'agentocode.syncKnowledge',
      'agentocode.showStatus',
      'agentocode.clearChat',
      'agentocode.restartWorker',
      'agentocode.openLogs',
      'agentocode.preloadModel',
      'agentocode.previewProposal',
      'agentocode.applyProposal',
      'agentocode.rejectProposal',
    ]) {
      assert.ok(harness.registeredCommands.has(cmd), `missing command ${cmd}`);
    }
    assert.ok(harness.webviewProviders.has('agentocode.chatView'));
    assert.ok(harness.treeProviders.has('agentocode.knowledgeView'));

    // SecretStorage key management
    await harness.registeredCommands.get('agentocode.setApiKey')!();
    assert.equal(harness.secretsStore.get('agentocode.groqApiKey'), 'gsk_lifecycle_key');

    // Webview resolution sets strict CSP + script nonce and handles slash commands.
    let receiveHandler: ((msg: unknown) => Promise<void>) | undefined;
    const postedToWebview: Array<{ type: string; payload?: { messages?: Array<{ content: string }>; hasApiKey?: boolean } }> = [];
    const mockWebviewView = {
      webview: {
        options: {},
        html: '',
        asWebviewUri: (uri: { fsPath: string }) => `vscode-resource://${uri.fsPath}`,
        postMessage: (msg: { type: string }) => {
          postedToWebview.push(msg as never);
          return Promise.resolve(true);
        },
        onDidReceiveMessage: (fn: (msg: unknown) => Promise<void>) => {
          receiveHandler = fn;
          return { dispose: () => undefined };
        },
      },
    };

    harness.webviewProviders.get('agentocode.chatView')!.resolveWebviewView(mockWebviewView);
    assert.ok(mockWebviewView.webview.html.includes("Content-Security-Policy"));
    assert.ok(mockWebviewView.webview.html.includes("default-src 'none'"));
    assert.ok(mockWebviewView.webview.html.includes('webview.css'));
    assert.ok(mockWebviewView.webview.html.includes('webview.js'));
    assert.ok(receiveHandler, 'webview message listener should be attached');

    await receiveHandler!({ type: 'ready' });
    assert.equal(postedToWebview[postedToWebview.length - 1].payload?.hasApiKey, true);

    await receiveHandler!({ type: 'sendPrompt', text: '/help', includeActiveFile: false });
    const afterHelp = postedToWebview[postedToWebview.length - 1].payload?.messages ?? [];
    assert.ok(afterHelp.some((m) => m.content.includes('/sync')));

    await receiveHandler!({ type: 'sendPrompt', text: '/sync', includeActiveFile: false });
    const afterSync = postedToWebview[postedToWebview.length - 1].payload?.messages ?? [];
    assert.ok(afterSync.some((m) => m.content.includes('1 tracked files')));

    await receiveHandler!({ type: 'sendPrompt', text: '/clear', includeActiveFile: false });
    const afterClear = postedToWebview[postedToWebview.length - 1].payload?.messages ?? [];
    assert.equal(afterClear.length, 0);

    // Changing a query-only setting must NOT dispose the worker process.
    const logsBeforeQuerySetting = harness.outputLines.length;
    await harness.configListeners[0]({
      affectsConfiguration: (sec) => sec === 'agentocode' || sec === 'agentocode.generationModel',
    });
    const logsAfterQuerySetting = harness.outputLines.slice(logsBeforeQuerySetting).join('\n');
    assert.equal(logsAfterQuerySetting.includes('MCP local worker process closed'), false);

    // Changing a worker-level setting MUST restart the worker cleanly.
    const logsBeforeWorkerSetting = harness.outputLines.length;
    await harness.configListeners[0]({
      affectsConfiguration: (sec) => sec === 'agentocode' || sec === 'agentocode.embedProvider',
    });
    const logsAfterWorkerSetting = harness.outputLines.slice(logsBeforeWorkerSetting).join('\n');
    assert.ok(logsAfterWorkerSetting.includes('MCP local worker process closed'));

    await harness.registeredCommands.get('agentocode.clearApiKey')!();
    assert.equal(harness.secretsStore.has('agentocode.groqApiKey'), false);

    // Workspace .env key is detected cleanly even with SecretStorage empty.
    await fsp.writeFile(path.join(ws, '.env'), 'GROQ_API_KEY=gsk_from_dotenv\n', 'utf8');
    await receiveHandler!({ type: 'ready' });
    assert.equal(postedToWebview[postedToWebview.length - 1].payload?.hasApiKey, true);

    // Setting SecretStorage while .env defines GROQ_API_KEY warns about .env precedence.
    await harness.registeredCommands.get('agentocode.setApiKey')!();
    assert.ok(
      harness.infoMessages.some((m) => m.includes('.env currently takes precedence')),
    );

    // Saving .env refreshes state cleanly without crashing.
    harness.saveListeners[0]({ uri: { scheme: 'file', fsPath: path.join(ws, '.env') } });
    // Yield briefly so the async refresh kicked off by .env save completes before teardown.
    await new Promise((resolve) => setTimeout(resolve, 100));

    assert.equal(harness.errorMessages.length, 0);
  } finally {
    await ext.deactivate().catch(() => undefined);
    for (const d of subscriptions) {
      d.dispose();
    }
    modAny._load = originalLoad;
    await fsp.rm(ws, { recursive: true, force: true, maxRetries: 15, retryDelay: 50 });
  }
});

