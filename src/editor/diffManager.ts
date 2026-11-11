import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import * as vscode from 'vscode';
import { EditProposal } from '../types.js';
import { resolveContainedPath } from '../worker/workspace.js';

export const PROPOSAL_SCHEME = 'agentocode-proposal';
export const EMPTY_SCHEME = 'agentocode-empty';

/**
 * Manages AI-proposed file edits inside VS Code via native diff previews and undoable WorkspaceEdits.
 */
export class DiffProposalManager implements vscode.Disposable {
  private readonly proposals = new Map<string, EditProposal>();
  private readonly onDidChangeEmitter = new vscode.EventEmitter<void>();
  readonly onDidChange = this.onDidChangeEmitter.event;

  private readonly disposables: vscode.Disposable[] = [];

  constructor(private readonly getWorkspaceRoot: () => string | undefined) {
    const proposalProvider: vscode.TextDocumentContentProvider = {
      provideTextDocumentContent: (uri) => {
        const id = uri.authority || uri.path.split('/').filter(Boolean)[0];
        const proposal = id ? this.proposals.get(id) : undefined;
        return proposal?.proposedContent ?? '';
      },
    };
    const emptyProvider: vscode.TextDocumentContentProvider = {
      provideTextDocumentContent: () => '',
    };

    this.disposables.push(
      vscode.workspace.registerTextDocumentContentProvider(PROPOSAL_SCHEME, proposalProvider),
      vscode.workspace.registerTextDocumentContentProvider(EMPTY_SCHEME, emptyProvider),
      this.onDidChangeEmitter,
    );
  }

  getAllProposals(): Record<string, EditProposal> {
    const out: Record<string, EditProposal> = {};
    for (const [id, prop] of this.proposals.entries()) {
      out[id] = prop;
    }
    return out;
  }

  getPendingProposals(): EditProposal[] {
    return Array.from(this.proposals.values()).filter((p) => p.status === 'pending');
  }

  getProposal(id: string): EditProposal | undefined {
    return this.proposals.get(id);
  }

  /**
   * Read the current workspace buffer (including unsaved editor edits) or disk file safely.
   */
  private async readCurrentText(fileUri: vscode.Uri, absPath: string): Promise<{ exists: boolean; text: string }> {
    const openDoc = vscode.workspace.textDocuments.find(
      (doc) => doc.uri.toString() === fileUri.toString(),
    );
    if (openDoc) {
      return { exists: true, text: openDoc.getText() };
    }
    if (!fs.existsSync(absPath)) {
      return { exists: false, text: '' };
    }
    const text = await fsp.readFile(absPath, 'utf8');
    return { exists: true, text };
  }

  async createProposal(params: {
    relativePath: string;
    proposedContent: string;
    rationale: string;
  }): Promise<EditProposal> {
    const workspaceRoot = this.getWorkspaceRoot();
    if (!workspaceRoot) {
      throw new Error('No open workspace folder.');
    }
    const contained = resolveContainedPath(workspaceRoot, params.relativePath, {
      allowNonExistent: true,
    });
    const fileUri = vscode.Uri.file(contained.realPath);
    const current = await this.readCurrentText(fileUri, contained.realPath);

    const id = crypto.randomBytes(6).toString('hex');
    const proposal: EditProposal = {
      id,
      relativePath: contained.relativePath,
      uriString: fileUri.toString(),
      originalContent: current.text,
      proposedContent: params.proposedContent,
      isNewFile: !current.exists,
      rationale: params.rationale,
      status: 'pending',
      createdAt: new Date().toISOString(),
    };

    this.proposals.set(id, proposal);
    this.onDidChangeEmitter.fire();
    return proposal;
  }

  async previewProposal(id: string): Promise<void> {
    const proposal = this.proposals.get(id);
    if (!proposal) {
      throw new Error(`Unknown proposal: ${id}`);
    }
    const targetUri = vscode.Uri.parse(proposal.uriString);
    const leftUri =
      proposal.isNewFile && !fs.existsSync(targetUri.fsPath)
        ? vscode.Uri.parse(`${EMPTY_SCHEME}://${proposal.id}/${proposal.relativePath}`)
        : targetUri;
    const rightUri = vscode.Uri.parse(`${PROPOSAL_SCHEME}://${proposal.id}/${proposal.relativePath}`);

    const title = `AgentoCode Diff: ${proposal.relativePath}`;
    await vscode.commands.executeCommand('vscode.diff', leftUri, rightUri, title);
  }

  async applyProposal(id: string): Promise<EditProposal> {
    const proposal = this.proposals.get(id);
    if (!proposal) {
      throw new Error(`Unknown proposal: ${id}`);
    }
    if (proposal.status !== 'pending') {
      return proposal;
    }

    const workspaceRoot = this.getWorkspaceRoot();
    if (!workspaceRoot) {
      throw new Error('No open workspace folder.');
    }
    const contained = resolveContainedPath(workspaceRoot, proposal.relativePath, {
      allowNonExistent: true,
    });
    const fileUri = vscode.Uri.file(contained.realPath);
    const current = await this.readCurrentText(fileUri, contained.realPath);

    if (current.text !== proposal.originalContent) {
      const choice = await vscode.window.showWarningMessage(
        `${proposal.relativePath} changed since AgentoCode created this edit proposal. Overwrite anyway?`,
        { modal: true },
        'Apply Anyway',
      );
      if (choice !== 'Apply Anyway') {
        return proposal;
      }
    }

    const edit = new vscode.WorkspaceEdit();
    if (!current.exists) {
      await fsp.mkdir(path.dirname(contained.realPath), { recursive: true });
      edit.createFile(fileUri, { ignoreIfExists: true });
      edit.insert(fileUri, new vscode.Position(0, 0), proposal.proposedContent);
    } else {
      const doc = await vscode.workspace.openTextDocument(fileUri);
      const lastLine = Math.max(0, doc.lineCount - 1);
      const lastChar = doc.lineAt(lastLine).text.length;
      const fullRange = new vscode.Range(0, 0, lastLine, lastChar);
      edit.replace(fileUri, fullRange, proposal.proposedContent);
    }

    const applied = await vscode.workspace.applyEdit(edit);
    if (!applied) {
      throw new Error(`VS Code failed to apply WorkspaceEdit for ${proposal.relativePath}.`);
    }

    const doc = await vscode.workspace.openTextDocument(fileUri);
    await vscode.window.showTextDocument(doc, { preview: false });

    const updated: EditProposal = { ...proposal, status: 'applied' };
    this.proposals.set(id, updated);
    this.onDidChangeEmitter.fire();
    return updated;
  }

  rejectProposal(id: string): EditProposal {
    const proposal = this.proposals.get(id);
    if (!proposal) {
      throw new Error(`Unknown proposal: ${id}`);
    }
    const updated: EditProposal = { ...proposal, status: 'rejected' };
    this.proposals.set(id, updated);
    this.onDidChangeEmitter.fire();
    return updated;
  }

  clearAll(): void {
    this.proposals.clear();
    this.onDidChangeEmitter.fire();
  }

  dispose(): void {
    for (const d of this.disposables) {
      d.dispose();
    }
  }
}

