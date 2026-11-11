import path from 'node:path';
import ts from 'typescript';

export interface StructuralUnit {
  kind: string;
  name: string;
  signature: string;
  lineStart: number;
  lineEnd: number;
  summary: string;
}

export interface FileStructuralAnalysis {
  language: string;
  parserUsed: 'typescript_ast' | 'python_structural' | 'markdown_headings' | 'line_text_fallback';
  limitations: string[];
  imports: string[];
  exports: string[];
  units: StructuralUnit[];
  chunks: Array<{
    index: number;
    title: string;
    lineStart: number;
    lineEnd: number;
    content: string;
  }>;
}

const DEFAULT_CHUNK_CHAR_LIMIT = 4500;

/**
 * Analyze a source file locally without any LLM API calls and produce deterministic
 * structural facts and AST-boundary-aware chunks.
 */
export function analyzeSourceFile(
  relativePath: string,
  text: string,
  chunkCharLimit = DEFAULT_CHUNK_CHAR_LIMIT,
): FileStructuralAnalysis {
  const ext = path.posix.extname(relativePath).toLowerCase();

  if (ext === '.ts' || ext === '.tsx' || ext === '.js' || ext === '.jsx' || ext === '.mjs' || ext === '.cjs') {
    return analyzeTypeScriptOrJs(relativePath, text, ext, chunkCharLimit);
  }
  if (ext === '.py' || ext === '.pyi') {
    return analyzePython(relativePath, text, chunkCharLimit);
  }
  if (ext === '.md' || ext === '.mdx') {
    return analyzeMarkdown(relativePath, text, chunkCharLimit);
  }
  return analyzeFallbackText(relativePath, text, ext || 'text', chunkCharLimit);
}

function splitOversizedSpan(
  lines: string[],
  startLine: number,
  endLine: number,
  title: string,
  charLimit: number,
): Array<{ title: string; lineStart: number; lineEnd: number; content: string }> {
  const slice = lines.slice(startLine - 1, endLine);
  const fullText = slice.join('\n');
  if (fullText.length <= charLimit) {
    return [{ title, lineStart: startLine, lineEnd: endLine, content: fullText }];
  }

  const fragments: Array<{ title: string; lineStart: number; lineEnd: number; content: string }> = [];
  let cursor = 0;
  let part = 1;
  while (cursor < slice.length) {
    let charCount = 0;
    let endCursor = cursor;
    while (endCursor < slice.length) {
      const nextLen = slice[endCursor].length + 1;
      if (endCursor > cursor && charCount + nextLen > charLimit) {
        break;
      }
      charCount += nextLen;
      endCursor++;
    }
    fragments.push({
      title: `${title} (part ${part})`,
      lineStart: startLine + cursor,
      lineEnd: startLine + endCursor - 1,
      content: slice.slice(cursor, endCursor).join('\n'),
    });
    cursor = endCursor;
    part++;
  }
  return fragments;
}

function packBoundarySections(
  lines: string[],
  sections: Array<{ title: string; lineStart: number; lineEnd: number }>,
  charLimit: number,
): Array<{ index: number; title: string; lineStart: number; lineEnd: number; content: string }> {
  if (sections.length === 0) {
    const raw = splitOversizedSpan(lines, 1, Math.max(1, lines.length), 'file', charLimit);
    return raw.map((chunk, idx) => ({ ...chunk, index: idx }));
  }

  const rawChunks: Array<{ title: string; lineStart: number; lineEnd: number; content: string }> = [];
  let batchTitles: string[] = [];
  let batchStart = -1;
  let batchEnd = -1;
  let batchChars = 0;

  const flushBatch = () => {
    if (batchStart < 1) {
      return;
    }
    const content = lines.slice(batchStart - 1, batchEnd).join('\n');
    rawChunks.push({
      title: batchTitles.slice(0, 3).join(', ') + (batchTitles.length > 3 ? ` (+${batchTitles.length - 3})` : ''),
      lineStart: batchStart,
      lineEnd: batchEnd,
      content,
    });
    batchTitles = [];
    batchStart = -1;
    batchEnd = -1;
    batchChars = 0;
  };

  for (const sec of sections) {
    const secText = lines.slice(sec.lineStart - 1, sec.lineEnd).join('\n');
    if (secText.length > charLimit) {
      flushBatch();
      rawChunks.push(...splitOversizedSpan(lines, sec.lineStart, sec.lineEnd, sec.title, charLimit));
      continue;
    }

    if (batchStart >= 1 && batchChars + secText.length + 2 > charLimit) {
      flushBatch();
    }
    if (batchStart < 1) {
      batchStart = sec.lineStart;
    }
    batchEnd = sec.lineEnd;
    batchChars += secText.length + 2;
    batchTitles.push(sec.title);
  }
  flushBatch();

  return rawChunks.map((chunk, idx) => ({ ...chunk, index: idx }));
}

function analyzeTypeScriptOrJs(
  relativePath: string,
  text: string,
  ext: string,
  chunkCharLimit: number,
): FileStructuralAnalysis {
  const scriptKind =
    ext === '.tsx' ? ts.ScriptKind.TSX :
    ext === '.jsx' ? ts.ScriptKind.JSX :
    ext === '.js' || ext === '.mjs' || ext === '.cjs' ? ts.ScriptKind.JS :
    ts.ScriptKind.TS;

  const sourceFile = ts.createSourceFile(relativePath, text, ts.ScriptTarget.Latest, true, scriptKind);
  const lines = text.split(/\r?\n/);
  const imports: string[] = [];
  const exports: string[] = [];
  const units: StructuralUnit[] = [];
  const boundarySections: Array<{ title: string; lineStart: number; lineEnd: number }> = [];

  const hasExportModifier = (node: ts.Node): boolean => {
    if (!ts.canHaveModifiers(node)) {
      return false;
    }
    const mods = ts.getModifiers(node);
    return Boolean(mods?.some((m) => m.kind === ts.SyntaxKind.ExportKeyword));
  };

  const lineOf = (pos: number): number => sourceFile.getLineAndCharacterOfPosition(pos).line + 1;

  for (const stmt of sourceFile.statements) {
    const startLine = lineOf(stmt.getStart(sourceFile));
    const endLine = lineOf(stmt.getEnd());

    if (ts.isImportDeclaration(stmt) && ts.isStringLiteral(stmt.moduleSpecifier)) {
      imports.push(stmt.moduleSpecifier.text);
      continue;
    }
    if (ts.isExportDeclaration(stmt) && stmt.moduleSpecifier && ts.isStringLiteral(stmt.moduleSpecifier)) {
      exports.push(`re-export ${stmt.moduleSpecifier.text}`);
      continue;
    }

    if (ts.isFunctionDeclaration(stmt)) {
      const name = stmt.name?.text ?? 'default_function';
      if (hasExportModifier(stmt)) {
        exports.push(name);
      }
      const params = stmt.parameters.map((p) => p.name.getText(sourceFile)).join(', ');
      units.push({
        kind: 'function',
        name,
        signature: `function ${name}(${params})`,
        lineStart: startLine,
        lineEnd: endLine,
        summary: `Function ${name} (L${startLine}-L${endLine})`,
      });
      boundarySections.push({ title: `function ${name}`, lineStart: startLine, lineEnd: endLine });
    } else if (ts.isClassDeclaration(stmt)) {
      const name = stmt.name?.text ?? 'AnonymousClass';
      if (hasExportModifier(stmt)) {
        exports.push(name);
      }
      const methods = stmt.members
        .filter((m): m is ts.MethodDeclaration => ts.isMethodDeclaration(m))
        .map((m) => m.name.getText(sourceFile));
      units.push({
        kind: 'class',
        name,
        signature: `class ${name}`,
        lineStart: startLine,
        lineEnd: endLine,
        summary: `Class ${name}${methods.length ? ` [methods: ${methods.join(', ')}]` : ''} (L${startLine}-L${endLine})`,
      });
      boundarySections.push({ title: `class ${name}`, lineStart: startLine, lineEnd: endLine });
    } else if (ts.isInterfaceDeclaration(stmt) || ts.isTypeAliasDeclaration(stmt) || ts.isEnumDeclaration(stmt)) {
      const name = stmt.name.text;
      if (hasExportModifier(stmt)) {
        exports.push(name);
      }
      const kind = ts.isInterfaceDeclaration(stmt) ? 'interface' : ts.isEnumDeclaration(stmt) ? 'enum' : 'type';
      units.push({
        kind,
        name,
        signature: `${kind} ${name}`,
        lineStart: startLine,
        lineEnd: endLine,
        summary: `${kind} ${name} (L${startLine}-L${endLine})`,
      });
      boundarySections.push({ title: `${kind} ${name}`, lineStart: startLine, lineEnd: endLine });
    } else if (ts.isVariableStatement(stmt)) {
      const isExported = hasExportModifier(stmt);
      const declNames: string[] = [];
      for (const decl of stmt.declarationList.declarations) {
        if (ts.isIdentifier(decl.name)) {
          declNames.push(decl.name.text);
          if (isExported) {
            exports.push(decl.name.text);
          }
        }
      }
      const title = declNames.length ? declNames.join(', ') : `statement@${startLine}`;
      if (declNames.length > 0 && (isExported || endLine - startLine >= 5)) {
        units.push({
          kind: 'binding',
          name: title,
          signature: `const/let ${title}`,
          lineStart: startLine,
          lineEnd: endLine,
          summary: `Top-level binding ${title} (L${startLine}-L${endLine})`,
        });
      }
      boundarySections.push({ title, lineStart: startLine, lineEnd: endLine });
    } else {
      boundarySections.push({ title: `statement L${startLine}`, lineStart: startLine, lineEnd: endLine });
    }
  }

  return {
    language: ext.startsWith('.ts') ? 'typescript' : 'javascript',
    parserUsed: 'typescript_ast',
    limitations: ['Dynamic imports/require paths computed at runtime are unresolved.'],
    imports: Array.from(new Set(imports)),
    exports: Array.from(new Set(exports)),
    units,
    chunks: packBoundarySections(lines, boundarySections, chunkCharLimit),
  };
}

function analyzePython(
  _relativePath: string,
  text: string,
  chunkCharLimit: number,
): FileStructuralAnalysis {
  const lines = text.split(/\r?\n/);
  const imports: string[] = [];
  const exports: string[] = [];
  const units: StructuralUnit[] = [];

  const topBlocks: Array<{ title: string; kind: string; name: string; signature: string; lineStart: number; lineEnd: number }> = [];
  let i = 0;
  let pendingDecoratorStart: number | null = null;

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();
    const indent = line.length - line.trimStart().length;
    const lineNum = i + 1;

    if (indent === 0) {
      if (trimmed.startsWith('@')) {
        if (pendingDecoratorStart === null) {
          pendingDecoratorStart = lineNum;
        }
        i++;
        continue;
      }

      const importMatch = /^import\s+([a-zA-Z0-9_.,\s]+)/.exec(trimmed);
      if (importMatch) {
        for (const mod of importMatch[1].split(',')) {
          const clean = mod.trim().split(/\s+as\s+/)[0].trim();
          if (clean) {
            imports.push(clean);
          }
        }
        pendingDecoratorStart = null;
        i++;
        continue;
      }

      const fromMatch = /^from\s+([a-zA-Z0-9_.]+)\s+import\s+/.exec(trimmed);
      if (fromMatch) {
        imports.push(fromMatch[1]);
        pendingDecoratorStart = null;
        i++;
        continue;
      }

      const defMatch = /^(async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)(.*)$/.exec(trimmed);
      if (defMatch) {
        const startLine = pendingDecoratorStart ?? lineNum;
        pendingDecoratorStart = null;
        const kindWord = defMatch[1].includes('class') ? 'class' : 'function';
        const name = defMatch[2];
        const sigRest = defMatch[3].replace(/:\s*(#.*)?$/, '').trim();
        let endLine = lineNum;
        let j = i + 1;
        while (j < lines.length) {
          const next = lines[j];
          const nextTrim = next.trim();
          if (nextTrim === '' || nextTrim.startsWith('#')) {
            j++;
            continue;
          }
          const nextIndent = next.length - next.trimStart().length;
          if (nextIndent === 0 && !nextTrim.startsWith(')') && !nextTrim.startsWith(']')) {
            break;
          }
          endLine = j + 1;
          j++;
        }
        if (!name.startsWith('_')) {
          exports.push(name);
        }
        const signature = `${defMatch[1]} ${name}${sigRest}`;
        units.push({
          kind: kindWord,
          name,
          signature,
          lineStart: startLine,
          lineEnd: endLine,
          summary: `${kindWord} ${name} (L${startLine}-L${endLine})`,
        });
        topBlocks.push({
          title: `${kindWord} ${name}`,
          kind: kindWord,
          name,
          signature,
          lineStart: startLine,
          lineEnd: endLine,
        });
        i = endLine;
        continue;
      }
      pendingDecoratorStart = null;
    }
    i++;
  }

  const sections = topBlocks.length > 0
    ? topBlocks.map((b) => ({ title: b.title, lineStart: b.lineStart, lineEnd: b.lineEnd }))
    : [{ title: 'module', lineStart: 1, lineEnd: Math.max(1, lines.length) }];

  // Ensure leading module header / imports before the first top-level block aren't dropped.
  if (topBlocks.length > 0 && topBlocks[0].lineStart > 1) {
    sections.unshift({ title: 'module header & imports', lineStart: 1, lineEnd: topBlocks[0].lineStart - 1 });
  }

  return {
    language: 'python',
    parserUsed: 'python_structural',
    limitations: ['Static indentation/signature scan; dynamic bindings and exec/importlib calls remain unresolved.'],
    imports: Array.from(new Set(imports)),
    exports: Array.from(new Set(exports)),
    units,
    chunks: packBoundarySections(lines, sections, chunkCharLimit),
  };
}

function analyzeMarkdown(
  _relativePath: string,
  text: string,
  chunkCharLimit: number,
): FileStructuralAnalysis {
  const lines = text.split(/\r?\n/);
  const units: StructuralUnit[] = [];
  const sections: Array<{ title: string; lineStart: number; lineEnd: number }> = [];
  let currentTitle = 'document';
  let currentStart = 1;

  for (let i = 0; i < lines.length; i++) {
    const m = /^(#{1,4})\s+(.+?)\s*$/.exec(lines[i]);
    if (m) {
      if (i + 1 > currentStart) {
        sections.push({ title: currentTitle, lineStart: currentStart, lineEnd: i });
      }
      currentTitle = m[2];
      currentStart = i + 1;
      units.push({
        kind: `h${m[1].length}`,
        name: m[2],
        signature: `${m[1]} ${m[2]}`,
        lineStart: i + 1,
        lineEnd: i + 1,
        summary: `Heading "${m[2]}" at L${i + 1}`,
      });
    }
  }
  if (lines.length >= currentStart) {
    sections.push({ title: currentTitle, lineStart: currentStart, lineEnd: Math.max(1, lines.length) });
  }

  return {
    language: 'markdown',
    parserUsed: 'markdown_headings',
    limitations: [],
    imports: [],
    exports: [],
    units,
    chunks: packBoundarySections(lines, sections, chunkCharLimit),
  };
}

function analyzeFallbackText(
  _relativePath: string,
  text: string,
  language: string,
  chunkCharLimit: number,
): FileStructuralAnalysis {
  const lines = text.split(/\r?\n/);
  const sections: Array<{ title: string; lineStart: number; lineEnd: number }> = [];
  let start = 1;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].trim() === '' && i + 1 - start >= 20) {
      sections.push({ title: `lines L${start}-L${i + 1}`, lineStart: start, lineEnd: i + 1 });
      start = i + 2;
    }
  }
  if (start <= lines.length) {
    sections.push({ title: `lines L${start}-L${lines.length}`, lineStart: start, lineEnd: lines.length });
  }

  return {
    language,
    parserUsed: 'line_text_fallback',
    limitations: ['Unsupported text format: line-bounded chunks only without language AST symbol extraction.'],
    imports: [],
    exports: [],
    units: [],
    chunks: packBoundarySections(lines, sections, chunkCharLimit),
  };
}

/**
 * Render an OKF v0.2 deterministic knowledge markdown document for a source file.
 * Zero LLM calls are made during generation.
 */
export function renderFileKnowledgeDocument(params: {
  relativePath: string;
  sha256: string;
  sizeBytes: number;
  analysis: FileStructuralAnalysis;
  updatedAt: string;
  resolvedDependencies: string[];
}): string {
  const { relativePath, sha256, sizeBytes, analysis, updatedAt, resolvedDependencies } = params;
  const frontmatter = [
    '---',
    'okf_version: "0.2"',
    `concept_id: "file:${relativePath}"`,
    'kind: "file"',
    `path: "${relativePath}"`,
    `language: "${analysis.language}"`,
    `parser: "${analysis.parserUsed}"`,
    `sha256: "${sha256}"`,
    `size_bytes: ${sizeBytes}`,
    `dependencies: ${JSON.stringify(resolvedDependencies)}`,
    `updated_at: "${updatedAt}"`,
    'generator: "local-deterministic-v1"',
    '---',
  ].join('\n');

  const lines: string[] = [
    frontmatter,
    '',
    `# File: \`${relativePath}\``,
    '',
    `- **Language:** ${analysis.language} (\`${analysis.parserUsed}\`)`,
    `- **SHA-256:** \`${sha256.slice(0, 12)}\` (${sizeBytes} bytes)`,
  ];

  if (analysis.imports.length > 0) {
    lines.push(`- **Imports:** ${analysis.imports.map((x) => `\`${x}\``).join(', ')}`);
  }
  if (analysis.exports.length > 0) {
    lines.push(`- **Exports:** ${analysis.exports.map((x) => `\`${x}\``).join(', ')}`);
  }
  if (analysis.limitations.length > 0) {
    lines.push(`- **Structural notes:** ${analysis.limitations.join(' ')}`);
  }

  lines.push('', '## Structural Units', '');
  if (analysis.units.length === 0) {
    lines.push('_No top-level AST declarations extracted._');
  } else {
    for (const unit of analysis.units) {
      lines.push(`- \`${unit.signature}\` (L${unit.lineStart}-L${unit.lineEnd})`);
    }
  }

  lines.push('', '## Indexed Chunks', '');
  for (const chunk of analysis.chunks) {
    lines.push(
      `### Chunk ${chunk.index + 1}: ${chunk.title} (L${chunk.lineStart}-L${chunk.lineEnd})`,
      '',
      '```' + analysis.language,
      chunk.content,
      '```',
      '',
    );
  }

  return lines.join('\n');
}

/**
 * Render an OKF v0.2 directory aggregation concept document bottom-up.
 */
export function renderDirectoryKnowledgeDocument(params: {
  relDir: string;
  childFiles: Array<{ relativePath: string; sha256: string; unitCount: number; language: string }>;
  childDirs: string[];
  updatedAt: string;
}): string {
  const { relDir, childFiles, childDirs, updatedAt } = params;
  const displayDir = relDir || '.';
  const conceptId = relDir ? `dir:${relDir}` : 'dir:.';
  const frontmatter = [
    '---',
    'okf_version: "0.2"',
    `concept_id: "${conceptId}"`,
    'kind: "directory"',
    `path: "${displayDir}"`,
    `updated_at: "${updatedAt}"`,
    'generator: "local-deterministic-v1"',
    '---',
  ].join('\n');

  const out: string[] = [
    frontmatter,
    '',
    `# Directory: \`${displayDir}\``,
    '',
    `- **Immediate child files:** ${childFiles.length}`,
    `- **Immediate subdirectories:** ${childDirs.length}`,
  ];

  if (childDirs.length > 0) {
    out.push('', '## Subdirectories', '');
    for (const d of childDirs) {
      out.push(`- \`${d}/\``);
    }
  }

  if (childFiles.length > 0) {
    out.push('', '## Files', '');
    for (const f of childFiles) {
      out.push(`- \`${f.relativePath}\` (${f.language}, ${f.unitCount} structural units, sha \`${f.sha256.slice(0, 8)}\`)`);
    }
  }

  return out.join('\n') + '\n';
}

/**
 * Render the text payload passed into the local embedding model for a file chunk.
 */
export function renderChunkEmbeddingText(params: {
  relativePath: string;
  analysis: FileStructuralAnalysis;
  chunkIndex: number;
}): string {
  const chunk = params.analysis.chunks[params.chunkIndex];
  const unitLines = params.analysis.units
    .filter((u) => u.lineStart >= chunk.lineStart && u.lineStart <= chunk.lineEnd)
    .map((u) => u.signature);
  return [
    `File: ${params.relativePath}`,
    `Language: ${params.analysis.language}`,
    `Section: ${chunk.title} (L${chunk.lineStart}-L${chunk.lineEnd})`,
    params.analysis.imports.length > 0 ? `Imports: ${params.analysis.imports.join(', ')}` : '',
    unitLines.length > 0 ? `Symbols: ${unitLines.join('; ')}` : '',
    '',
    chunk.content,
  ]
    .filter(Boolean)
    .join('\n');
}

