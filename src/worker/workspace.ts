import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';

export const MAX_FILE_BYTES = 512_000;

export const EXCLUDED_DIR_NAMES = new Set([
  '.git',
  '.hg',
  '.svn',
  'node_modules',
  '.venv',
  'venv',
  'env',
  '__pycache__',
  '.guardian',
  'knowledge_base',
  'dist',
  'dist-test',
  'build',
  'out',
  '.next',
  '.nuxt',
  '.turbo',
  '.pytest_cache',
  '.mypy_cache',
  '.ruff_cache',
  '.vscode-test',
  'coverage',
  'target',
]);

const BINARY_EXTENSIONS = new Set([
  '.png', '.jpg', '.jpeg', '.gif', '.webp', '.ico', '.bmp', '.tiff',
  '.pdf', '.zip', '.gz', '.tar', '.7z', '.rar', '.jar', '.vsix', '.whl',
  '.exe', '.dll', '.so', '.dylib', '.node', '.bin', '.dat', '.db', '.sqlite',
  '.sqlite3', '.onnx', '.pt', '.pth', '.safetensors', '.gguf', '.woff',
  '.woff2', '.ttf', '.eot', '.mp3', '.mp4', '.wav', '.avi', '.mov', '.pyc',
  '.pyo', '.class', '.o', '.obj', '.lib', '.a',
]);

const CREDENTIAL_EXTENSIONS = new Set([
  '.pem', '.key', '.pfx', '.p12', '.p8', '.jks', '.keystore', '.crt', '.cer',
]);

const ALLOWED_ENV_EXAMPLES = new Set(['.env.example', '.env.template', '.env.sample']);

const WINDOWS_RESERVED_BASENAMES = new Set([
  'con', 'prn', 'aux', 'nul',
  'com1', 'com2', 'com3', 'com4', 'com5', 'com6', 'com7', 'com8', 'com9',
  'lpt1', 'lpt2', 'lpt3', 'lpt4', 'lpt5', 'lpt6', 'lpt7', 'lpt8', 'lpt9',
]);

const ESCAPE_CHAR = '~';

/**
 * Normalize a relative path to forward slashes without leading or trailing slashes.
 */
export function normalizeRelPath(relPath: string): string {
  const cleaned = relPath.replace(/\\/g, '/').replace(/^\.\/+/, '').replace(/^\/+|\/+$/g, '');
  return cleaned === '.' ? '' : cleaned;
}

/**
 * Return true if a file or directory name represents a credential/secret or hidden system entry.
 */
export function isSensitiveOrExcludedName(name: string): boolean {
  const lower = name.toLowerCase();
  if (!lower) {
    return false;
  }
  if (EXCLUDED_DIR_NAMES.has(lower) || EXCLUDED_DIR_NAMES.has(name)) {
    return true;
  }
  if (ALLOWED_ENV_EXAMPLES.has(lower)) {
    return false;
  }
  if (lower === '.env' || lower.startsWith('.env.')) {
    return true;
  }
  if (lower === '.npmrc' || lower === '.pypirc' || lower === '.netrc') {
    return true;
  }
  if (lower.startsWith('id_rsa') || lower.startsWith('id_ed25519')) {
    return true;
  }
  if (lower.includes('credential') || lower.includes('secret')) {
    return true;
  }
  const ext = path.posix.extname(lower);
  if (CREDENTIAL_EXTENSIONS.has(ext) || BINARY_EXTENSIONS.has(ext)) {
    return true;
  }
  return false;
}

/**
 * Check whether every path segment in a workspace-relative path is eligible for indexing/reading.
 */
export function isRelPathEligible(relPath: string): boolean {
  const norm = normalizeRelPath(relPath);
  if (!norm) {
    return true;
  }
  const parts = norm.split('/');
  for (let i = 0; i < parts.length; i++) {
    const part = parts[i];
    if (!part || part === '.' || part === '..') {
      return false;
    }
    if (part.startsWith('.') && !ALLOWED_ENV_EXAMPLES.has(part.toLowerCase()) && part !== '.vscode' && part !== '.github') {
      return false;
    }
    if (isSensitiveOrExcludedName(part)) {
      return false;
    }
  }
  return true;
}

/**
 * Detect binary content from a raw file buffer using null bytes and high replacement/control density.
 */
export function isProbablyBinaryBuffer(buf: Buffer): boolean {
  const sampleLen = Math.min(buf.length, 8192);
  if (sampleLen === 0) {
    return false;
  }
  let suspicious = 0;
  for (let i = 0; i < sampleLen; i++) {
    const b = buf[i];
    if (b === 0) {
      return true;
    }
    if (b < 32 && b !== 9 && b !== 10 && b !== 13) {
      suspicious++;
    }
  }
  return suspicious / sampleLen > 0.15;
}

export interface ContainedPathResult {
  relativePath: string;
  absolutePath: string;
  realPath: string;
}

/**
 * Resolve a user/tool-supplied path strictly inside workspaceRoot, rejecting escapes,
 * symlinks/junctions pointing outside the root, and hidden/credential/generated exclusions.
 */
export function resolveContainedPath(
  workspaceRoot: string,
  candidatePath: string,
  options: { allowNonExistent?: boolean; skipEligibilityCheck?: boolean } = {},
): ContainedPathResult {
  if (!candidatePath || candidatePath.includes('\0')) {
    throw new Error('Invalid empty or null-byte path.');
  }

  const realRoot = fs.realpathSync(workspaceRoot);
  const rawJoined = path.isAbsolute(candidatePath)
    ? path.resolve(candidatePath)
    : path.resolve(realRoot, candidatePath);

  const relFromRoot = normalizeRelPath(path.relative(realRoot, rawJoined));
  if (relFromRoot.startsWith('..') || path.isAbsolute(relFromRoot)) {
    throw new Error(`Path escapes workspace boundary: ${candidatePath}`);
  }

  if (!options.skipEligibilityCheck && relFromRoot && !isRelPathEligible(relFromRoot)) {
    throw new Error(`Path is excluded by workspace security/eligibility rules: ${relFromRoot}`);
  }

  let resolvedReal: string;
  if (fs.existsSync(rawJoined)) {
    resolvedReal = fs.realpathSync(rawJoined);
  } else {
    if (!options.allowNonExistent) {
      throw new Error(`Path does not exist: ${relFromRoot || '.'}`);
    }
    // Walk up until an existing ancestor is found and verify its realpath stays inside root.
    let probe = path.dirname(rawJoined);
    const tail: string[] = [path.basename(rawJoined)];
    while (!fs.existsSync(probe)) {
      const parent = path.dirname(probe);
      if (parent === probe) {
        throw new Error(`No existing ancestor for path: ${candidatePath}`);
      }
      tail.unshift(path.basename(probe));
      probe = parent;
    }
    const realAncestor = fs.realpathSync(probe);
    resolvedReal = path.join(realAncestor, ...tail);
  }

  const realRel = normalizeRelPath(path.relative(realRoot, resolvedReal));
  if (realRel.startsWith('..') || path.isAbsolute(realRel)) {
    throw new Error(`Symlink or junction escapes workspace boundary: ${candidatePath}`);
  }

  return {
    relativePath: realRel,
    absolutePath: rawJoined,
    realPath: resolvedReal,
  };
}

/**
 * Reversibly encode a single path segment for knowledge_base/ paths, handling the escape marker '~'
 * and Windows reserved device names / forbidden filename characters.
 */
export function encodeKnowledgeSegment(segment: string): string {
  let encoded = '';
  for (const ch of segment) {
    const code = ch.codePointAt(0)!;
    if (
      ch === ESCAPE_CHAR ||
      ch === '<' ||
      ch === '>' ||
      ch === ':' ||
      ch === '"' ||
      ch === '/' ||
      ch === '\\' ||
      ch === '|' ||
      ch === '?' ||
      ch === '*' ||
      code < 32
    ) {
      encoded += `${ESCAPE_CHAR}${code.toString(16).padStart(2, '0')}`;
    } else {
      encoded += ch;
    }
  }
  const stem = encoded.split('.')[0].toLowerCase();
  if (WINDOWS_RESERVED_BASENAMES.has(stem)) {
    encoded = `${ESCAPE_CHAR}res_${encoded}`;
  }
  if (encoded.endsWith(' ') || encoded.endsWith('.')) {
    encoded = `${encoded.slice(0, -1)}${ESCAPE_CHAR}${encoded.charCodeAt(encoded.length - 1).toString(16).padStart(2, '0')}`;
  }
  return encoded || `${ESCAPE_CHAR}empty`;
}

/**
 * Decode a segment encoded by encodeKnowledgeSegment.
 */
export function decodeKnowledgeSegment(encoded: string): string {
  let value = encoded.startsWith(`${ESCAPE_CHAR}res_`) ? encoded.slice(5) : encoded;
  if (value === `${ESCAPE_CHAR}empty`) {
    return '';
  }
  value = value.replace(/~([0-9a-fA-F]{2})/g, (_, hex: string) =>
    String.fromCodePoint(Number.parseInt(hex, 16)),
  );
  return value;
}

export function knowledgeRelPathForFile(relFilePath: string): string {
  const parts = normalizeRelPath(relFilePath).split('/').filter(Boolean);
  const encodedParts = parts.map(encodeKnowledgeSegment);
  return path.posix.join('files', ...encodedParts, 'content.md');
}

export function knowledgeRelPathForDir(relDirPath: string): string {
  const norm = normalizeRelPath(relDirPath);
  if (!norm) {
    return path.posix.join('directories', 'content.md');
  }
  const parts = norm.split('/').filter(Boolean).map(encodeKnowledgeSegment);
  return path.posix.join('directories', ...parts, 'content.md');
}

/**
 * Compute SHA-256 hex digest of a buffer or UTF-8 string.
 */
export function sha256Hex(input: Buffer | string): string {
  return crypto.createHash('sha256').update(input).digest('hex');
}

/**
 * Walk a workspace directory yielding eligible UTF-8 text files under MAX_FILE_BYTES.
 */
export async function scanEligibleWorkspaceFiles(
  workspaceRoot: string,
): Promise<Array<{ relativePath: string; realPath: string; sizeBytes: number; sha256: string; text: string }>> {
  const realRoot = await fsp.realpath(workspaceRoot);
  const results: Array<{ relativePath: string; realPath: string; sizeBytes: number; sha256: string; text: string }> = [];

  async function walk(currentDir: string, relDir: string): Promise<void> {
    const entries = await fsp.readdir(currentDir, { withFileTypes: true });
    entries.sort((a, b) => a.name.localeCompare(b.name));

    for (const entry of entries) {
      const rel = relDir ? `${relDir}/${entry.name}` : entry.name;
      if (!isRelPathEligible(rel)) {
        continue;
      }
      const full = path.join(currentDir, entry.name);

      let stat: fs.Stats;
      try {
        stat = await fsp.lstat(full);
      } catch {
        continue;
      }
      if (stat.isSymbolicLink()) {
        // Reject symlinks/junctions pointing outside root or skipping containment.
        try {
          const resolved = await fsp.realpath(full);
          const relCheck = normalizeRelPath(path.relative(realRoot, resolved));
          if (relCheck.startsWith('..') || path.isAbsolute(relCheck)) {
            continue;
          }
          stat = await fsp.stat(resolved);
        } catch {
          continue;
        }
      }

      if (stat.isDirectory()) {
        await walk(full, rel);
        continue;
      }
      if (!stat.isFile() || stat.size > MAX_FILE_BYTES) {
        continue;
      }
      try {
        const buf = await fsp.readFile(full);
        if (isProbablyBinaryBuffer(buf)) {
          continue;
        }
        const text = buf.toString('utf8');
        results.push({
          relativePath: rel,
          realPath: full,
          sizeBytes: buf.length,
          sha256: sha256Hex(buf),
          text,
        });
      } catch {
        // Ignore transient unreadable files during scan.
      }
    }
  }

  await walk(realRoot, '');
  return results;
}

const TRANSIENT_FS_ERRORS = new Set(['EPERM', 'EACCES', 'EBUSY']);

/**
 * Atomically write UTF-8 text or binary data using a temp file and bounded backoff retries
 * for transient Windows sharing/access locks.
 */
export async function atomicWriteFileWithRetry(
  targetPath: string,
  data: string | Buffer,
  maxAttempts = 6,
): Promise<void> {
  await fsp.mkdir(path.dirname(targetPath), { recursive: true });
  const tmpPath = `${targetPath}.tmp.${process.pid}.${crypto.randomBytes(4).toString('hex')}`;
  await fsp.writeFile(tmpPath, data);

  let lastError: unknown;
  try {
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        await fsp.rename(tmpPath, targetPath);
        return;
      } catch (err: unknown) {
        lastError = err;
        const code = (err as NodeJS.ErrnoException | undefined)?.code ?? '';
        if (!TRANSIENT_FS_ERRORS.has(code) || attempt === maxAttempts) {
          throw err;
        }
        await new Promise((resolve) => setTimeout(resolve, 25 * attempt));
      }
    }
  } finally {
    try {
      if (fs.existsSync(tmpPath)) {
        await fsp.unlink(tmpPath);
      }
    } catch {
      // Best-effort cleanup of temp file.
    }
  }
  throw lastError;
}

