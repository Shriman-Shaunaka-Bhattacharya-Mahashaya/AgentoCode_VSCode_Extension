import assert from 'node:assert/strict';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import Module from 'node:module';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { renderSafeMarkdown } from '../src/webview/markdown.js';

interface MockTextDoc {
  uri: { scheme: string; fsPath: string; toString(): string };
  languageId: string;
  isDirty: boolean;
  lineCount: number;
  getText(range?: unknown): string;
  lineAt(line: number): { text: string };
}

function createMockVscode(state: {
  textDocuments: MockTextDoc[];
  activeTextEditor?: {
    document: MockTextDoc;
    selection: {
      active: { line: number; character: number };
      start: { line: number; character: number };
      end: { line: number; character: number };
    };
  };
  diagnostics?: Array<[unknown, Array<{ severity: number; range: { start: { line: number; character: number } }; source?: string; message: string }>]>;
  appliedEdits: Array<{ ops: Array<{ type: string; uri: string; text?: string }> }>;
  executedCommands: Array<{ cmd: string; args: unknown[] }>;
  warningChoice?: string;
}) {
  class EventEmitter {
    private listeners: Array<() => void> = [];
    event = (fn: () => void) => {
      this.listeners.push(fn);
      return { dispose: () => undefined };
    };
    fire() {
      for (const l of this.listeners) {
        l();
      }
    }
    dispose() {
      this.listeners = [];
    }
  }

  class TreeItem {
    label: string;
    collapsibleState: number;
    constructor(label: string, collapsibleState = 0) {
      this.label = label;
      this.collapsibleState = collapsibleState;
    }
  }

  class ThemeIcon {
    id: string;
    constructor(id: string) {
      this.id = id;
    }
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
    ops: Array<{ type: string; uri: string; text?: string }> = [];
    createFile(uri: { toString(): string }) {
      this.ops.push({ type: 'createFile', uri: uri.toString() });
    }
    insert(uri: { toString(): string }, _pos: unknown, text: string) {
      this.ops.push({ type: 'insert', uri: uri.toString(), text });
    }
    replace(uri: { toString(): string }, _range: unknown, text: string) {
      this.ops.push({ type: 'replace', uri: uri.toString(), text });
    }
  }

  const makeUri = (scheme: string, fsPathValue: string) => ({
    scheme,
    fsPath: fsPathValue,
    authority: '',
    path: fsPathValue,
    toString: () => (scheme === 'file' ? `file://${fsPathValue.replace(/\\/g, '/')}` : `${scheme}://${fsPathValue}`),
  });

  return {
    EventEmitter,
    TreeItem,
    TreeItemCollapsibleState: { None: 0, Collapsed: 1, Expanded: 2 },
    ThemeIcon,
    Position,
    Range,
    WorkspaceEdit,
    DiagnosticSeverity: { Error: 0, Warning: 1, Information: 2, Hint: 3 },
    Uri: {
      file: (p: string) => makeUri('file', path.resolve(p)),
      parse: (raw: string) => {
        const m = /^([a-z-]+):\/\/(.*)$/i.exec(raw);
        if (m && m[1] === 'file') {
          return makeUri('file', m[2]);
        }
        return makeUri(m?.[1] ?? 'file', m?.[2] ?? raw);
      },
    },
    workspace: {
      get textDocuments() {
        return state.textDocuments;
      },
      registerTextDocumentContentProvider: () => ({ dispose: () => undefined }),
      openTextDocument: async (uri: { scheme?: string; fsPath: string; toString(): string }) => {
        const existing = state.textDocuments.find((d) => d.uri.toString() === uri.toString());
        if (existing) {
          return existing;
        }
        const text = fs.existsSync(uri.fsPath) ? fs.readFileSync(uri.fsPath, 'utf8') : '';
        const lines = text.split(/\r?\n/);
        const doc: MockTextDoc = {
          uri: { scheme: uri.scheme ?? 'file', fsPath: uri.fsPath, toString: uri.toString },
          languageId: 'typescript',
          isDirty: false,
          lineCount: lines.length,
          getText: () => text,
          lineAt: (idx: number) => ({ text: lines[idx] ?? '' }),
        };
        state.textDocuments.push(doc);
        return doc;
      },
      applyEdit: async (edit: WorkspaceEdit) => {
        state.appliedEdits.push({ ops: edit.ops });
        return true;
      },
    },
    window: {
      get activeTextEditor() {
        return state.activeTextEditor;
      },
      showTextDocument: async () => undefined,
      showWarningMessage: async () => state.warningChoice,
    },
    languages: {
      getDiagnostics: () => state.diagnostics ?? [],
    },
    commands: {
      executeCommand: async (cmd: string, ...args: unknown[]) => {
        state.executedCommands.push({ cmd, args });
      },
    },
  };
}

function withMockVscode<T>(mockVscode: unknown, fn: () => Promise<T>): Promise<T> {
  const modAny = Module as unknown as {
    _load: (request: string, parent: unknown, isMain: boolean) => unknown;
  };
  const originalLoad = modAny._load;
  modAny._load = function (request: string, parent: unknown, isMain: boolean) {
    if (request === 'vscode') {
      return mockVscode;
    }
    return originalLoad.call(this, request, parent, isMain);
  };
  return fn().finally(() => {
    modAny._load = originalLoad;
  });
}

test('renderSafeMarkdown sanitizes HTML/XSS payloads while rendering code, headings, and lists', () => {
  const html = renderSafeMarkdown(
    [
      '# Title <script>alert(1)</script>',
      '',
      'Use `x < y` and **bold <img src=x onerror=alert(2)>**.',
      '',
      '- item one',
      '- item two',
      '',
      '```ts',
      'const tag = "<b>hi</b>";',
      '```',
    ].join('\n'),
  );

  assert.ok(!html.includes('<script>'));
  assert.ok(!html.includes('<img '));
  assert.ok(html.includes('&lt;script&gt;alert(1)&lt;/script&gt;'));
  assert.ok(html.includes('<h2>Title &lt;script&gt;alert(1)&lt;/script&gt;</h2>'));
  assert.ok(html.includes('<strong>bold &lt;img src=x onerror=alert(2)&gt;</strong>'));
  assert.ok(html.includes('<ul><li>item one</li><li>item two</li></ul>'));
  assert.ok(html.includes('<div class="code-lang">ts</div>'));
  assert.ok(html.includes('&lt;b&gt;hi&lt;/b&gt;'));
});

test('DiffProposalManager, ideContext, and KnowledgeTreeProvider work end-to-end with VS Code API contracts', async () => {
  const ws = await fsp.mkdtemp(path.join(os.tmpdir(), 'agentocode-ide-'));
  try {
    const filePath = path.join(ws, 'app.ts');
    await fsp.writeFile(filePath, 'export const a = 1;\n', 'utf8');

    const fileUriString = `file://${path.resolve(filePath).replace(/\\/g, '/')}`;
    const dirtyDoc: MockTextDoc = {
      uri: { scheme: 'file', fsPath: path.resolve(filePath), toString: () => fileUriString },
      languageId: 'typescript',
      isDirty: true,
      lineCount: 1,
      getText: () => 'export const a = 999;\n',
      lineAt: () => ({ text: 'export const a = 999;' }),
    };

    const mockState = {
      textDocuments: [dirtyDoc],
      activeTextEditor: {
        document: dirtyDoc,
        selection: {
          active: { line: 0, character: 7 },
          start: { line: 0, character: 0 },
          end: { line: 0, character: 21 },
        },
      },
      diagnostics: [
        [
          dirtyDoc.uri,
          [
            {
              severity: 0,
              range: { start: { line: 0, character: 17 } },
              source: 'ts',
              message: 'Sample error',
            },
          ],
        ],
      ] as Array<[unknown, Array<{ severity: number; range: { start: { line: number; character: number } }; source?: string; message: string }>]>,
      appliedEdits: [] as Array<{ ops: Array<{ type: string; uri: string; text?: string }> }>,
      executedCommands: [] as Array<{ cmd: string; args: unknown[] }>,
      warningChoice: undefined as string | undefined,
    };

    const mockVscode = createMockVscode(mockState);

    await withMockVscode(mockVscode, async () => {
      const { DiffProposalManager } = await import('../src/editor/diffManager.js');
      const { getActiveEditorSummary, getUnsavedBufferIfDirty, getWorkspaceDiagnosticsSummary } =
        await import('../src/workspace/ideContext.js');
      const { KnowledgeTreeProvider } = await import('../src/views/knowledgeTreeProvider.js');

      // Active editor & dirty buffer inspection
      const active = getActiveEditorSummary(ws);
      assert.equal(active.relativePath, 'app.ts');
      assert.ok(active.formattedContext.includes('Dirty (unsaved edits): true'));
      assert.equal(getUnsavedBufferIfDirty(filePath), 'export const a = 999;\n');

      const diagText = getWorkspaceDiagnosticsSummary(ws, 'app.ts');
      assert.ok(diagText.includes('[ERROR] app.ts:1:18 (ts): Sample error'));

      // Proposal creation, diff preview, apply, and reject
      const manager = new DiffProposalManager(() => ws);
      await assert.rejects(() =>
        manager.createProposal({ relativePath: '../outside.ts', proposedContent: 'x', rationale: 'bad' }),
      );
      await assert.rejects(() =>
        manager.createProposal({ relativePath: '.env', proposedContent: 'K=V', rationale: 'secret' }),
      );

      const prop = await manager.createProposal({
        relativePath: 'app.ts',
        proposedContent: 'export const a = 42;\n',
        rationale: 'Update constant',
      });
      assert.equal(prop.originalContent, 'export const a = 999;\n');

      await manager.previewProposal(prop.id);
      assert.equal(mockState.executedCommands[0].cmd, 'vscode.diff');

      // Tree view surfaces pending proposal & untrusted state cleanly
      const knowledgeDir = path.join(ws, '.guardian', 'knowledge_base');
      await fsp.mkdir(knowledgeDir, { recursive: true });
      await fsp.writeFile(path.join(knowledgeDir, 'index.md'), '# Index\n', 'utf8');

      const tree = new KnowledgeTreeProvider(manager, () => knowledgeDir);
      tree.updateSnapshot({ isTrusted: false, hasApiKey: false, status: null });
      const untrustedNodes = await tree.getChildren();
      assert.equal(untrustedNodes[0].label, 'Untrusted Workspace');

      tree.updateSnapshot({ isTrusted: true, hasApiKey: true, status: null });
      await tree.refreshKnowledgeDocs();
      const trustedNodes = await tree.getChildren();
      assert.ok(trustedNodes.some((n) => n.label === 'Pending Edit Proposals'));
      assert.ok(trustedNodes.some((n) => n.label === 'Knowledge Base Docs' && n.description === '1 files'));

      const applied = await manager.applyProposal(prop.id);
      assert.equal(applied.status, 'applied');
      assert.equal(mockState.appliedEdits.length, 1);
      assert.equal(mockState.appliedEdits[0].ops[0].type, 'replace');

      const prop2 = await manager.createProposal({
        relativePath: 'new_dir/added.ts',
        proposedContent: 'export const b = 2;\n',
        rationale: 'New file',
      });
      const rejected = manager.rejectProposal(prop2.id);
      assert.equal(rejected.status, 'rejected');

      manager.dispose();
    });
  } finally {
    await fsp.rm(ws, { recursive: true, force: true });
  }
});
