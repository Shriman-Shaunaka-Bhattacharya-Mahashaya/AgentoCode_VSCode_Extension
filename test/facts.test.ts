import assert from 'node:assert/strict';
import test from 'node:test';
import {
  analyzeSourceFile,
  renderChunkEmbeddingText,
  renderDirectoryKnowledgeDocument,
  renderFileKnowledgeDocument,
} from '../src/worker/facts.js';

test('TypeScript AST extraction parses imports, exports, classes, and splits oversized files at boundaries', () => {
  const source = [
    "import { helper } from './helper';",
    '',
    'export interface CalcResult {',
    '  value: number;',
    '}',
    '',
    'export class Calculator {',
    '  add(a: number, b: number): number {',
    '    return helper(a + b);',
    '  }',
    '}',
    '',
    'export function multiply(a: number, b: number): number {',
    '  return a * b;',
    '}',
  ].join('\n');

  const analysis = analyzeSourceFile('src/calc.ts', source, 120);
  assert.equal(analysis.parserUsed, 'typescript_ast');
  assert.deepEqual(analysis.imports, ['./helper']);
  assert.ok(analysis.exports.includes('CalcResult'));
  assert.ok(analysis.exports.includes('Calculator'));
  assert.ok(analysis.exports.includes('multiply'));
  assert.ok(analysis.units.some((u) => u.kind === 'class' && u.name === 'Calculator'));
  assert.ok(analysis.units.some((u) => u.kind === 'function' && u.name === 'multiply'));
  assert.ok(analysis.chunks.length >= 2, 'expected boundary splitting with tight char limit');
});

test('Python structural parser extracts imports, classes, functions, and top-level boundaries', () => {
  const source = [
    'import math',
    'from calc_utils import normalize',
    '',
    '@dataclass',
    'class HistoryStore:',
    '    def append(self, value: float) -> None:',
    '        pass',
    '',
    'async def compute_ratio(a: float, b: float) -> float:',
    '    return normalize(a / b)',
  ].join('\n');

  const analysis = analyzeSourceFile('pkg/service.py', source, 140);
  assert.equal(analysis.parserUsed, 'python_structural');
  assert.deepEqual(analysis.imports, ['math', 'calc_utils']);
  assert.ok(analysis.units.some((u) => u.kind === 'class' && u.name === 'HistoryStore'));
  assert.ok(analysis.units.some((u) => u.kind === 'function' && u.name === 'compute_ratio'));
  assert.ok(analysis.chunks.length >= 2);
});

test('unsupported text formats record explicit structural limitations', () => {
  const analysis = analyzeSourceFile('notes.txt', 'alpha\nbeta\ngamma\n', 200);
  assert.equal(analysis.parserUsed, 'line_text_fallback');
  assert.ok(analysis.limitations.some((l) => l.includes('Unsupported text format')));
});

test('OKF v0.2 file and directory documents include required frontmatter and deterministic metadata', () => {
  const analysis = analyzeSourceFile('calculator.py', 'def add(a, b):\n    return a + b\n');
  const fileDoc = renderFileKnowledgeDocument({
    relativePath: 'calculator.py',
    sha256: 'abc123456789',
    sizeBytes: 30,
    analysis,
    updatedAt: '2026-11-11T00:00:00.000Z',
    resolvedDependencies: [],
  });
  assert.ok(fileDoc.includes('okf_version: "0.2"'));
  assert.ok(fileDoc.includes('concept_id: "file:calculator.py"'));
  assert.ok(fileDoc.includes('generator: "local-deterministic-v1"'));

  const dirDoc = renderDirectoryKnowledgeDocument({
    relDir: '',
    childFiles: [{ relativePath: 'calculator.py', sha256: 'abc123456789', unitCount: 1, language: 'python' }],
    childDirs: ['utils'],
    updatedAt: '2026-11-11T00:00:00.000Z',
  });
  assert.ok(dirDoc.includes('concept_id: "dir:."'));
  assert.ok(dirDoc.includes('`calculator.py`'));
  assert.ok(dirDoc.includes('`utils/`'));
});

test('Markdown heading extraction, oversized single-unit fragmentation, and chunk embedding payload work cleanly', () => {
  const md = ['# Heading One', 'Intro text', '## Sub Heading', 'More details here'].join('\n');
  const mdAnalysis = analyzeSourceFile('docs/guide.md', md, 40);
  assert.equal(mdAnalysis.parserUsed, 'markdown_headings');
  assert.ok(mdAnalysis.units.some((u) => u.name === 'Heading One'));
  assert.ok(mdAnalysis.chunks.length >= 2);

  const hugeFn = ['export function giant() {', ...Array.from({ length: 20 }, (_, i) => `  const line${i} = ${i};`), '}'].join('\n');
  const hugeAnalysis = analyzeSourceFile('src/giant.ts', hugeFn, 120);
  assert.ok(hugeAnalysis.chunks.some((c) => c.title.includes('(part 1)')));
  assert.ok(hugeAnalysis.chunks.some((c) => c.title.includes('(part 2)')));

  const payload = renderChunkEmbeddingText({
    relativePath: 'src/giant.ts',
    analysis: hugeAnalysis,
    chunkIndex: 0,
  });
  assert.ok(payload.includes('File: src/giant.ts'));
  assert.ok(payload.includes('Symbols: function giant()'));
});

