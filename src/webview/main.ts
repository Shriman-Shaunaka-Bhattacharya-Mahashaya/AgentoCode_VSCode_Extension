import {
  ChatMessageItem,
  ChatStatePayload,
  EditProposal,
  ExtensionToWebviewMessage,
  KnowledgeStatusSnapshot,
  ToolTrace,
  WebviewToExtensionMessage,
} from '../types.js';
import { escapeHtml, renderSafeMarkdown } from './markdown.js';

interface VsCodeWebviewApi {
  postMessage(message: WebviewToExtensionMessage): void;
  getState(): unknown;
  setState(state: unknown): void;
}

declare function acquireVsCodeApi(): VsCodeWebviewApi;

const vscode = acquireVsCodeApi();

interface UiState {
  messages: ChatMessageItem[];
  proposals: Record<string, EditProposal>;
  hasApiKey: boolean;
  isTrusted: boolean;
  isRunning: boolean;
  status: KnowledgeStatusSnapshot | null;
  activeFile: string | null;
  generationModel: string;
  includeActiveFile: boolean;
  draft: string;
}

const state: UiState = {
  messages: [],
  proposals: {},
  hasApiKey: false,
  isTrusted: true,
  isRunning: false,
  status: null,
  activeFile: null,
  generationModel: 'openai/gpt-oss-120b',
  includeActiveFile: true,
  draft: '',
};

let shellMounted = false;

function formatIndexBadge(status: KnowledgeStatusSnapshot | null): { text: string; cls: string } {
  if (!status) {
    return { text: 'Index: Idle', cls: 'badge-muted' };
  }
  if (status.syncRunning) {
    return { text: 'Syncing...', cls: 'badge-warn' };
  }
  if (status.indexAvailable) {
    return { text: `Semantic Ready (${status.indexedCount})`, cls: 'badge-ok' };
  }
  if (status.modelState === 'error') {
    return { text: 'Model Error (Lexical)', cls: 'badge-err' };
  }
  return { text: `Lexical Fallback (${status.conceptCount} files)`, cls: 'badge-muted' };
}

function renderStatusHeaderHtml(): string {
  const idx = formatIndexBadge(state.status);
  const keyBadge = state.hasApiKey
    ? '<span class="badge badge-ok">Groq Key Set</span>'
    : '<button class="btn-inline btn-warn" data-action="setApiKey">Set Groq Key</button>';

  const trustBanner = !state.isTrusted
    ? '<div class="banner banner-warn">Workspace is untrusted. Trust the workspace in VS Code to enable indexing and chat.</div>'
    : '';

  return `
    <div class="topbar-row">
      <span class="brand">AgentoCode</span>
      <span class="model-tag" title="Query-time Groq model">${escapeHtml(state.generationModel)}</span>
    </div>
    <div class="topbar-row badges">
      ${keyBadge}
      <span class="badge ${idx.cls}">${escapeHtml(idx.text)}</span>
    </div>
    <div class="topbar-actions">
      <button class="btn-small" data-action="syncKnowledge" title="Index workspace locally (0 Groq calls)">Sync</button>
      <button class="btn-small" data-action="showStatus" title="Refresh status (/status)">/status</button>
      <button class="btn-small" data-action="preloadModel" title="Load local Nomic ONNX model">Preload Model</button>
      <button class="btn-small" data-action="openLogs" title="Open worker logs">Logs</button>
    </div>
    ${trustBanner}
  `;
}

function renderToolTrace(trace: ToolTrace): string {
  const icon = trace.status === 'running' ? '...' : trace.status === 'error' ? 'ERR' : 'OK';
  const resultHtml = trace.resultPreview
    ? `<div class="tool-result">${escapeHtml(trace.resultPreview)}</div>`
    : '';
  return `
    <details class="tool-trace tool-${trace.status}">
      <summary>
        <span class="tool-badge">${icon}</span>
        <span class="tool-name">${escapeHtml(trace.name)}</span>
        <span class="tool-args">${escapeHtml(trace.argsPreview)}</span>
      </summary>
      ${resultHtml}
    </details>
  `;
}

function renderProposalCard(prop: EditProposal): string {
  const statusCls = prop.status === 'applied' ? 'badge-ok' : prop.status === 'rejected' ? 'badge-err' : 'badge-warn';
  const actions =
    prop.status === 'pending'
      ? `
      <div class="proposal-actions">
        <button class="btn-small" data-action="previewProposal" data-id="${escapeHtml(prop.id)}">Preview Diff</button>
        <button class="btn-small btn-primary" data-action="applyProposal" data-id="${escapeHtml(prop.id)}">Apply</button>
        <button class="btn-small" data-action="rejectProposal" data-id="${escapeHtml(prop.id)}">Reject</button>
      </div>
    `
      : `
      <div class="proposal-actions">
        <button class="btn-small" data-action="previewProposal" data-id="${escapeHtml(prop.id)}">View Diff</button>
      </div>
    `;

  return `
    <div class="proposal-card">
      <div class="proposal-head">
        <span class="proposal-path">${escapeHtml(prop.relativePath)}</span>
        <span class="badge ${statusCls}">${escapeHtml(prop.status)}</span>
      </div>
      <div class="proposal-rationale">${escapeHtml(prop.rationale)}</div>
      ${actions}
    </div>
  `;
}

function renderMessagesHtml(): string {
  if (state.messages.length === 0) {
    return `
      <div class="empty-state">
        <p><strong>AgentoCode (100% Local Index + Query-Time Groq)</strong></p>
        <p>Indexing runs locally with AST/structural parsing and local embeddings — zero Groq API calls until you ask a question.</p>
        <div class="hint-list">
          <div><code>/sync</code> — Run local incremental indexing</div>
          <div><code>/status</code> — Inspect model, vector &amp; failure counters</div>
          <div><code>/preload</code> — Load local Nomic embedding model</div>
          <div><code>/clear</code> — Clear conversation history</div>
        </div>
      </div>
    `;
  }

  return state.messages
    .map((msg) => {
      const toolsHtml = msg.tools && msg.tools.length > 0 ? `<div class="tools-list">${msg.tools.map(renderToolTrace).join('')}</div>` : '';
      const proposalsHtml =
        msg.proposalIds && msg.proposalIds.length > 0
          ? `<div class="proposals-list">${msg.proposalIds
              .map((id) => state.proposals[id])
              .filter(Boolean)
              .map(renderProposalCard)
              .join('')}</div>`
          : '';
      const errorHtml = msg.error ? `<div class="msg-error">${escapeHtml(msg.error)}</div>` : '';
      const bodyHtml = msg.content ? `<div class="msg-body">${renderSafeMarkdown(msg.content)}</div>` : '';

      return `
        <article class="msg msg-${msg.role}">
          <div class="msg-role">${escapeHtml(msg.role)}</div>
          ${toolsHtml}
          ${bodyHtml}
          ${proposalsHtml}
          ${errorHtml}
        </article>
      `;
    })
    .join('');
}

function wireActionButtons(container: ParentNode): void {
  container.querySelectorAll<HTMLElement>('[data-action]').forEach((el) => {
    el.addEventListener('click', () => {
      const action = el.dataset.action;
      const id = el.dataset.id;
      if (action === 'setApiKey') {
        vscode.postMessage({ type: 'setApiKey' });
      } else if (action === 'syncKnowledge') {
        vscode.postMessage({ type: 'syncKnowledge' });
      } else if (action === 'showStatus') {
        vscode.postMessage({ type: 'showStatus' });
      } else if (action === 'preloadModel') {
        vscode.postMessage({ type: 'preloadModel' });
      } else if (action === 'openLogs') {
        vscode.postMessage({ type: 'openLogs' });
      } else if (action === 'previewProposal' && id) {
        vscode.postMessage({ type: 'previewProposal', id });
      } else if (action === 'applyProposal' && id) {
        vscode.postMessage({ type: 'applyProposal', id });
      } else if (action === 'rejectProposal' && id) {
        vscode.postMessage({ type: 'rejectProposal', id });
      }
    });
  });
}

function ensureShellMounted(): void {
  if (shellMounted) {
    return;
  }
  const root = document.getElementById('app');
  if (!root) {
    return;
  }
  root.innerHTML = `
    <div class="shell">
      <header id="topbar" class="topbar"></header>
      <main id="timeline" class="timeline"></main>
      <footer class="composer">
        <div id="composerMeta" class="composer-meta"></div>
        <textarea id="promptInput" rows="3" placeholder="Ask about your codebase, request an edit, or type /sync, /status, /clear..."></textarea>
        <div class="composer-actions">
          <button id="clearBtn" class="btn-small" type="button">Clear</button>
          <span id="sendWrap"></span>
        </div>
      </footer>
    </div>
  `;

  const input = document.getElementById('promptInput') as HTMLTextAreaElement | null;
  if (input) {
    input.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter' && !ev.shiftKey) {
        ev.preventDefault();
        submitPrompt();
      }
    });
    input.addEventListener('input', () => {
      state.draft = input.value;
    });
  }
  document.getElementById('clearBtn')?.addEventListener('click', () => {
    vscode.postMessage({ type: 'clearChat' });
  });
  shellMounted = true;
}

function updateHeader(): void {
  ensureShellMounted();
  const topbar = document.getElementById('topbar');
  if (!topbar) {
    return;
  }
  topbar.innerHTML = renderStatusHeaderHtml();
  wireActionButtons(topbar);
}

function updateTimeline(): void {
  ensureShellMounted();
  const timeline = document.getElementById('timeline');
  if (!timeline) {
    return;
  }
  const shouldStickBottom =
    timeline.scrollTop + timeline.clientHeight >= timeline.scrollHeight - 40;
  timeline.innerHTML = renderMessagesHtml();
  wireActionButtons(timeline);
  if (shouldStickBottom) {
    timeline.scrollTop = timeline.scrollHeight;
  }
}

function updateComposerControls(): void {
  ensureShellMounted();
  const meta = document.getElementById('composerMeta');
  if (meta) {
    meta.innerHTML = state.activeFile
      ? `<label class="active-file-toggle"><input id="attachActiveFile" type="checkbox" ${state.includeActiveFile ? 'checked' : ''} /> Attach <code>${escapeHtml(state.activeFile)}</code></label>`
      : '<span class="muted">No active workspace file</span>';

    const attachCb = document.getElementById('attachActiveFile') as HTMLInputElement | null;
    attachCb?.addEventListener('change', () => {
      state.includeActiveFile = attachCb.checked;
    });
  }

  const input = document.getElementById('promptInput') as HTMLTextAreaElement | null;
  if (input) {
    input.disabled = !state.isTrusted;
  }

  const sendWrap = document.getElementById('sendWrap');
  if (sendWrap) {
    sendWrap.innerHTML = state.isRunning
      ? '<button id="stopBtn" class="btn-danger" type="button">Stop</button>'
      : `<button id="sendBtn" class="btn-primary" type="button" ${!state.isTrusted ? 'disabled' : ''}>Send</button>`;
    document.getElementById('sendBtn')?.addEventListener('click', submitPrompt);
    document.getElementById('stopBtn')?.addEventListener('click', () => {
      vscode.postMessage({ type: 'cancelRun' });
    });
  }
}

function renderAll(): void {
  updateHeader();
  updateTimeline();
  updateComposerControls();
}

function submitPrompt(): void {
  const input = document.getElementById('promptInput') as HTMLTextAreaElement | null;
  const text = (input?.value ?? state.draft).trim();
  if (!text || state.isRunning) {
    return;
  }
  state.draft = '';
  if (input) {
    input.value = '';
  }
  vscode.postMessage({
    type: 'sendPrompt',
    text,
    includeActiveFile: state.includeActiveFile,
  });
}

function applyFullState(payload: ChatStatePayload): void {
  state.messages = payload.messages;
  state.proposals = payload.proposals;
  state.hasApiKey = payload.hasApiKey;
  state.isTrusted = payload.isTrusted;
  state.isRunning = payload.isRunning;
  state.status = payload.status;
  state.activeFile = payload.activeFile;
  state.generationModel = payload.generationModel;
  renderAll();
}

window.addEventListener('message', (event: MessageEvent<ExtensionToWebviewMessage>) => {
  const msg = event.data;
  if (!msg) {
    return;
  }
  switch (msg.type) {
    case 'state':
      applyFullState(msg.payload);
      break;
    case 'streamDelta': {
      const target = state.messages.find((m) => m.id === msg.messageId);
      if (target) {
        target.content += msg.delta;
        updateTimeline();
      }
      break;
    }
    case 'toolUpdate': {
      const target = state.messages.find((m) => m.id === msg.messageId);
      if (target) {
        target.tools = target.tools ?? [];
        const idx = target.tools.findIndex((t) => t.id === msg.trace.id);
        if (idx >= 0) {
          target.tools[idx] = msg.trace;
        } else {
          target.tools.push(msg.trace);
        }
        updateTimeline();
      }
      break;
    }
    case 'proposalUpdate': {
      state.proposals[msg.proposal.id] = msg.proposal;
      if (msg.messageId) {
        const target = state.messages.find((m) => m.id === msg.messageId);
        if (target) {
          target.proposalIds = target.proposalIds ?? [];
          if (!target.proposalIds.includes(msg.proposal.id)) {
            target.proposalIds.push(msg.proposal.id);
          }
        }
      }
      updateTimeline();
      break;
    }
    case 'statusUpdate':
      state.status = msg.status;
      updateHeader();
      break;
    case 'runFinished': {
      state.isRunning = false;
      const target = state.messages.find((m) => m.id === msg.messageId);
      if (target && msg.error) {
        target.error = msg.error;
      }
      updateTimeline();
      updateComposerControls();
      break;
    }
  }
});

renderAll();
vscode.postMessage({ type: 'ready' });
