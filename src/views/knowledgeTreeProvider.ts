import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import * as vscode from 'vscode';
import { DiffProposalManager } from '../editor/diffManager.js';
import { EditProposal, KnowledgeStatusSnapshot } from '../types.js';

export class KnowledgeTreeItem extends vscode.TreeItem {
  readonly kind: string;
  readonly proposalId?: string;
  readonly children?: KnowledgeTreeItem[];

  constructor(params: {
    label: string;
    kind: string;
    collapsibleState?: vscode.TreeItemCollapsibleState;
    description?: string;
    tooltip?: string;
    iconId?: string;
    command?: vscode.Command;
    contextValue?: string;
    proposalId?: string;
    children?: KnowledgeTreeItem[];
  }) {
    super(params.label, params.collapsibleState ?? vscode.TreeItemCollapsibleState.None);
    this.kind = params.kind;
    this.description = params.description;
    this.tooltip = params.tooltip;
    this.command = params.command;
    this.contextValue = params.contextValue;
    this.proposalId = params.proposalId;
    this.children = params.children;
    if (params.iconId) {
      this.iconPath = new vscode.ThemeIcon(params.iconId);
    }
  }
}

async function listMarkdownDocs(rootDir: string, maxFiles = 150): Promise<Array<{ relPath: string; fullPath: string }>> {
  if (!fs.existsSync(rootDir)) {
    return [];
  }
  const out: Array<{ relPath: string; fullPath: string }> = [];

  async function walk(dir: string, rel: string): Promise<void> {
    if (out.length >= maxFiles) {
      return;
    }
    const entries = await fsp.readdir(dir, { withFileTypes: true });
    entries.sort((a, b) => a.name.localeCompare(b.name));
    for (const entry of entries) {
      if (out.length >= maxFiles) {
        return;
      }
      const nextRel = rel ? `${rel}/${entry.name}` : entry.name;
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        await walk(full, nextRel);
      } else if (entry.isFile() && entry.name.endsWith('.md')) {
        out.push({ relPath: nextRel, fullPath: full });
      }
    }
  }

  await walk(rootDir, '');
  return out;
}

/**
 * VS Code TreeDataProvider surfacing local model state, index metrics, pending edit proposals,
 * and deterministic OKF v0.2 knowledge documents.
 */
export class KnowledgeTreeProvider implements vscode.TreeDataProvider<KnowledgeTreeItem> {
  private readonly onDidChangeTreeDataEmitter = new vscode.EventEmitter<KnowledgeTreeItem | undefined | void>();
  readonly onDidChangeTreeData = this.onDidChangeTreeDataEmitter.event;

  private status: KnowledgeStatusSnapshot | null = null;
  private knowledgeDocs: Array<{ relPath: string; fullPath: string }> = [];
  private hasApiKey = false;
  private isTrusted = true;

  constructor(
    private readonly proposalManager: DiffProposalManager,
    private readonly getKnowledgeDir: () => string | undefined,
  ) {
    this.proposalManager.onDidChange(() => this.refresh());
  }

  updateSnapshot(params: {
    status?: KnowledgeStatusSnapshot | null;
    hasApiKey?: boolean;
    isTrusted?: boolean;
  }): void {
    if (params.status !== undefined) {
      this.status = params.status;
    }
    if (params.hasApiKey !== undefined) {
      this.hasApiKey = params.hasApiKey;
    }
    if (params.isTrusted !== undefined) {
      this.isTrusted = params.isTrusted;
    }
    void this.refreshKnowledgeDocs();
  }

  async refreshKnowledgeDocs(): Promise<void> {
    const dir = this.getKnowledgeDir();
    this.knowledgeDocs = dir ? await listMarkdownDocs(dir) : [];
    this.refresh();
  }

  refresh(): void {
    this.onDidChangeTreeDataEmitter.fire();
  }

  getTreeItem(element: KnowledgeTreeItem): vscode.TreeItem {
    return element;
  }

  async getChildren(element?: KnowledgeTreeItem): Promise<KnowledgeTreeItem[]> {
    if (element) {
      return element.children ?? [];
    }

    const roots: KnowledgeTreeItem[] = [];

    if (!this.isTrusted) {
      roots.push(
        new KnowledgeTreeItem({
          label: 'Untrusted Workspace',
          kind: 'untrusted',
          description: 'Indexing & worker disabled',
          iconId: 'shield',
          tooltip: 'Trust this workspace to enable local indexing and AgentoCode chat.',
        }),
      );
      return roots;
    }

    // 1. Model & Index Status section
    const st = this.status;
    const statusChildren: KnowledgeTreeItem[] = st
      ? [
          new KnowledgeTreeItem({
            label: 'Provider',
            kind: 'stat',
            description: `${st.embedProvider} (${st.dtype})`,
            iconId: 'cpu',
          }),
          new KnowledgeTreeItem({
            label: 'Model',
            kind: 'stat',
            description: `${st.modelId} [${st.modelState}]`,
            iconId: st.modelState === 'ready' ? 'pass-filled' : st.modelState === 'error' ? 'error' : 'circle-outline',
          }),
          new KnowledgeTreeItem({
            label: 'Vectors / Concepts',
            kind: 'stat',
            description: `${st.indexedCount} vectors / ${st.conceptCount} files`,
            iconId: 'database',
          }),
          new KnowledgeTreeItem({
            label: 'Embed Calls',
            kind: 'stat',
            description: `docs=${st.embedDocCalls}, queries=${st.embedQueryCalls}, loads=${st.loadCount}`,
            iconId: 'pulse',
          }),
          new KnowledgeTreeItem({
            label: 'Exclusions & Failures',
            kind: 'stat',
            description: `overlength=${st.unindexedOverlengthCount}, failures=${st.failureCount}`,
            iconId: st.failureCount > 0 || st.unindexedOverlengthCount > 0 ? 'warning' : 'check',
          }),
          new KnowledgeTreeItem({
            label: 'Fingerprint',
            kind: 'stat',
            description: `${st.vectorFingerprint} (${st.indexAvailable ? 'semantic ready' : 'lexical fallback'})`,
            iconId: 'tag',
          }),
        ]
      : [
          new KnowledgeTreeItem({
            label: 'Worker not yet queried',
            kind: 'stat',
            description: 'Click Refresh or run Sync Knowledge',
            iconId: 'info',
          }),
        ];

    if (st?.lastError) {
      statusChildren.push(
        new KnowledgeTreeItem({
          label: 'Last Error',
          kind: 'error',
          description: st.lastError,
          tooltip: st.lastError,
          iconId: 'error',
        }),
      );
    }
    if (st && st.recentFailures.length > 0) {
      for (const f of st.recentFailures.slice(0, 10)) {
        statusChildren.push(
          new KnowledgeTreeItem({
            label: f.path,
            kind: 'failure',
            description: f.reason,
            tooltip: `${f.path}: ${f.reason}`,
            iconId: 'warning',
          }),
        );
      }
    }

    roots.push(
      new KnowledgeTreeItem({
        label: 'Index & Model Status',
        kind: 'section',
        collapsibleState: vscode.TreeItemCollapsibleState.Expanded,
        description: st ? (st.syncRunning ? 'Syncing...' : st.indexAvailable ? 'Ready' : 'Lexical fallback') : 'Idle',
        iconId: 'server-environment',
        children: statusChildren,
      }),
    );

    // 2. Pending Edit Proposals section
    const pending: EditProposal[] = this.proposalManager.getPendingProposals();
    if (pending.length > 0) {
      roots.push(
        new KnowledgeTreeItem({
          label: 'Pending Edit Proposals',
          kind: 'proposalsSection',
          collapsibleState: vscode.TreeItemCollapsibleState.Expanded,
          description: `${pending.length} awaiting approval`,
          iconId: 'git-compare',
          children: pending.map(
            (p) =>
              new KnowledgeTreeItem({
                label: p.relativePath,
                kind: 'proposal',
                proposalId: p.id,
                contextValue: 'proposalPending',
                description: p.isNewFile ? 'new file' : 'modified',
                tooltip: p.rationale,
                iconId: p.isNewFile ? 'new-file' : 'diff',
                command: {
                  command: 'agentocode.previewProposal',
                  title: 'Preview Diff',
                  arguments: [p.id],
                },
              }),
          ),
        }),
      );
    }

    // 3. Generated Knowledge Docs section
    const docItems = this.knowledgeDocs.map(
      (doc) =>
        new KnowledgeTreeItem({
          label: doc.relPath,
          kind: 'knowledgeDoc',
          iconId: 'markdown',
          command: {
            command: 'vscode.open',
            title: 'Open Knowledge Document',
            arguments: [vscode.Uri.file(doc.fullPath)],
          },
        }),
    );

    roots.push(
      new KnowledgeTreeItem({
        label: 'Knowledge Base Docs',
        kind: 'docsSection',
        collapsibleState:
          docItems.length > 0
            ? vscode.TreeItemCollapsibleState.Collapsed
            : vscode.TreeItemCollapsibleState.None,
        description: `${docItems.length} files`,
        iconId: 'book',
        children: docItems,
      }),
    );

    // 4. Actions section
    roots.push(
      new KnowledgeTreeItem({
        label: 'Actions',
        kind: 'actionsSection',
        collapsibleState: vscode.TreeItemCollapsibleState.Expanded,
        iconId: 'tools',
        children: [
          new KnowledgeTreeItem({
            label: this.hasApiKey ? 'Update Groq API Key' : 'Set Groq API Key',
            kind: 'action',
            description: this.hasApiKey ? 'Configured' : 'Required for queries',
            iconId: 'key',
            command: { command: 'agentocode.setApiKey', title: 'Set Groq API Key' },
          }),
          new KnowledgeTreeItem({
            label: 'Sync Knowledge & Vectors',
            kind: 'action',
            description: 'Local only (0 Groq calls)',
            iconId: 'sync',
            command: { command: 'agentocode.syncKnowledge', title: 'Sync Knowledge & Vectors' },
          }),
          new KnowledgeTreeItem({
            label: 'Preload Embedding Model',
            kind: 'action',
            description: 'Warm local Nomic ONNX',
            iconId: 'cloud-download',
            command: { command: 'agentocode.preloadModel', title: 'Preload Local Embedding Model' },
          }),
          new KnowledgeTreeItem({
            label: 'Open Worker Logs',
            kind: 'action',
            iconId: 'output',
            command: { command: 'agentocode.openLogs', title: 'Open Backend & Worker Logs' },
          }),
        ],
      }),
    );

    return roots;
  }
}

