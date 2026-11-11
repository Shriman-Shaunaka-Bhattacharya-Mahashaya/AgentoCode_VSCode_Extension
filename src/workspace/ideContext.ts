import path from 'node:path';
import * as vscode from 'vscode';
import { normalizeRelPath } from '../worker/workspace.js';

/**
 * Collect active VS Code editor context (file path, unsaved status, selection text).
 */
export function getActiveEditorSummary(workspaceRoot?: string): {
  relativePath: string | null;
  formattedContext: string;
} {
  const editor = vscode.window.activeTextEditor;
  if (!editor || editor.document.uri.scheme !== 'file') {
    return {
      relativePath: null,
      formattedContext: 'No active workspace file editor is currently focused.',
    };
  }

  const fsPath = editor.document.uri.fsPath;
  const relPath = workspaceRoot
    ? normalizeRelPath(path.relative(workspaceRoot, fsPath))
    : fsPath;

  const isInsideWorkspace = workspaceRoot ? !relPath.startsWith('..') && !path.isAbsolute(relPath) : true;
  const displayPath = isInsideWorkspace ? relPath || path.basename(fsPath) : fsPath;

  const selection = editor.selection;
  const selectedText = editor.document.getText(selection);
  const line = selection.active.line + 1;
  const col = selection.active.character + 1;

  const parts = [
    `Active File: ${displayPath}`,
    `Language: ${editor.document.languageId}`,
    `Dirty (unsaved edits): ${editor.document.isDirty}`,
    `Cursor: L${line}:${col}`,
  ];

  if (selectedText.trim()) {
    const startL = selection.start.line + 1;
    const endL = selection.end.line + 1;
    parts.push(`Selected Text (L${startL}-L${endL}):\n\`\`\`${editor.document.languageId}\n${selectedText.slice(0, 6000)}\n\`\`\``);
  }

  return {
    relativePath: isInsideWorkspace ? displayPath : null,
    formattedContext: parts.join('\n'),
  };
}

/**
 * Collect active VS Code Problems/Diagnostics (errors & warnings), optionally filtered to a relative path.
 */
export function getWorkspaceDiagnosticsSummary(
  workspaceRoot?: string,
  filterRelativePath?: string,
): string {
  const entries = vscode.languages.getDiagnostics();
  const normalizedFilter = filterRelativePath ? normalizeRelPath(filterRelativePath) : null;
  const lines: string[] = [];

  for (const [uri, diagnostics] of entries) {
    if (uri.scheme !== 'file' || diagnostics.length === 0) {
      continue;
    }
    const rel = workspaceRoot ? normalizeRelPath(path.relative(workspaceRoot, uri.fsPath)) : uri.fsPath;
    if (workspaceRoot && (rel.startsWith('..') || path.isAbsolute(rel))) {
      continue;
    }
    if (normalizedFilter && rel !== normalizedFilter) {
      continue;
    }

    for (const d of diagnostics) {
      if (d.severity > vscode.DiagnosticSeverity.Warning) {
        continue;
      }
      const sev = d.severity === vscode.DiagnosticSeverity.Error ? 'ERROR' : 'WARN';
      const line = d.range.start.line + 1;
      const col = d.range.start.character + 1;
      lines.push(`[${sev}] ${rel}:${line}:${col} (${d.source ?? 'lsp'}): ${d.message}`);
      if (lines.length >= 80) {
        break;
      }
    }
    if (lines.length >= 80) {
      break;
    }
  }

  return lines.length > 0 ? lines.join('\n') : 'No errors or warnings reported in VS Code diagnostics.';
}

/**
 * Read open dirty editor text if the file is open and modified in VS Code so the agent sees
 * unsaved edits instead of stale disk content.
 */
export function getUnsavedBufferIfDirty(absPath: string): string | null {
  const targetUri = vscode.Uri.file(absPath).toString();
  const doc = vscode.workspace.textDocuments.find((d) => d.uri.toString() === targetUri);
  if (doc && doc.isDirty) {
    return doc.getText();
  }
  return null;
}

