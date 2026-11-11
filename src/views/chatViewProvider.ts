import crypto from 'node:crypto';
import * as vscode from 'vscode';
import {
  ChatMessageItem,
  ChatStatePayload,
  EditProposal,
  ExtensionToWebviewMessage,
  KnowledgeStatusSnapshot,
  ToolTrace,
  WebviewToExtensionMessage,
} from '../types.js';

export interface ChatControllerHandlers {
  onSendPrompt: (text: string, includeActiveFile: boolean) => Promise<void>;
  onCancelRun: () => void;
  onClearChat: () => void;
  onSyncKnowledge: () => Promise<void>;
  onShowStatus: () => Promise<void>;
  onPreloadModel: () => Promise<void>;
  onSetApiKey: () => Promise<void>;
  onOpenLogs: () => void;
  onPreviewProposal: (id: string) => Promise<void>;
  onApplyProposal: (id: string) => Promise<void>;
  onRejectProposal: (id: string) => void;
  getStateSnapshot: () => Promise<ChatStatePayload>;
}

/**
 * Hosts the AgentoCode Sidebar Webview Chat panel with strict CSP and live streaming updates.
 */
export class ChatViewProvider implements vscode.WebviewViewProvider {
  public static readonly viewType = 'agentocode.chatView';
  private view: vscode.WebviewView | undefined;

  constructor(
    private readonly extensionUri: vscode.Uri,
    private readonly handlers: ChatControllerHandlers,
  ) {}

  resolveWebviewView(webviewView: vscode.WebviewView): void {
    this.view = webviewView;
    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [vscode.Uri.joinPath(this.extensionUri, 'dist')],
    };
    webviewView.webview.html = this.buildHtml(webviewView.webview);

    webviewView.webview.onDidReceiveMessage(async (msg: WebviewToExtensionMessage) => {
      try {
        switch (msg.type) {
          case 'ready':
            await this.pushFullState();
            break;
          case 'sendPrompt':
            await this.handlers.onSendPrompt(msg.text, msg.includeActiveFile);
            break;
          case 'cancelRun':
            this.handlers.onCancelRun();
            break;
          case 'clearChat':
            this.handlers.onClearChat();
            await this.pushFullState();
            break;
          case 'setApiKey':
            await this.handlers.onSetApiKey();
            await this.pushFullState();
            break;
          case 'syncKnowledge':
            await this.handlers.onSyncKnowledge();
            break;
          case 'showStatus':
            await this.handlers.onShowStatus();
            break;
          case 'preloadModel':
            await this.handlers.onPreloadModel();
            break;
          case 'openLogs':
            this.handlers.onOpenLogs();
            break;
          case 'previewProposal':
            await this.handlers.onPreviewProposal(msg.id);
            break;
          case 'applyProposal':
            await this.handlers.onApplyProposal(msg.id);
            await this.pushFullState();
            break;
          case 'rejectProposal':
            this.handlers.onRejectProposal(msg.id);
            await this.pushFullState();
            break;
        }
      } catch (err: unknown) {
        vscode.window.showErrorMessage(
          `AgentoCode error: ${err instanceof Error ? err.message : String(err)}`,
        );
      }
    });
  }

  async pushFullState(): Promise<void> {
    if (!this.view) {
      return;
    }
    const payload = await this.handlers.getStateSnapshot();
    this.post({ type: 'state', payload });
  }

  postStreamDelta(messageId: string, delta: string): void {
    this.post({ type: 'streamDelta', messageId, delta });
  }

  postToolUpdate(messageId: string, trace: ToolTrace): void {
    this.post({ type: 'toolUpdate', messageId, trace });
  }

  postProposalUpdate(proposal: EditProposal, messageId?: string): void {
    this.post({ type: 'proposalUpdate', messageId, proposal });
  }

  postStatusUpdate(status: KnowledgeStatusSnapshot): void {
    this.post({ type: 'statusUpdate', status });
  }

  postRunFinished(messageId: string, error?: string): void {
    this.post({ type: 'runFinished', messageId, error });
  }

  private post(message: ExtensionToWebviewMessage): void {
    void this.view?.webview.postMessage(message);
  }

  private buildHtml(webview: vscode.Webview): string {
    const nonce = crypto.randomBytes(16).toString('hex');
    const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(this.extensionUri, 'dist', 'webview.js'));
    const styleUri = webview.asWebviewUri(vscode.Uri.joinPath(this.extensionUri, 'dist', 'webview.css'));

    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource}; script-src 'nonce-${nonce}';" />
  <link rel="stylesheet" href="${styleUri}" />
  <title>AgentoCode Chat</title>
</head>
<body>
  <div id="app"></div>
  <script nonce="${nonce}" src="${scriptUri}"></script>
</body>
</html>`;
  }
}

export function createChatMessage(
  role: ChatMessageItem['role'],
  content: string,
): ChatMessageItem {
  return {
    id: crypto.randomBytes(6).toString('hex'),
    role,
    content,
    timestamp: new Date().toISOString(),
    tools: [],
    proposalIds: [],
  };
}
