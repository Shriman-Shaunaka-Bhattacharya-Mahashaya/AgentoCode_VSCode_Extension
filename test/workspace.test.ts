import assert from 'node:assert/strict';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import {
  atomicWriteFileWithRetry,
  decodeKnowledgeSegment,
  encodeKnowledgeSegment,
  isProbablyBinaryBuffer,
  isRelPathEligible,
  MAX_FILE_BYTES,
  resolveContainedPath,
  scanEligibleWorkspaceFiles,
} from '../src/worker/workspace.js';

async function makeTempDir(prefix: string): Promise<string> {
  return fsp.mkdtemp(path.join(os.tmpdir(), prefix));
}

test('reversible knowledge segment encoding handles Windows reserved names and escape chars', () => {
  const samples = ['con', 'aux.py', 'nul', 'com1.ts', 'foo~bar', 'a:b*c?', 'trail.', 'normal_file.py'];
  for (const s of samples) {
    const encoded = encodeKnowledgeSegment(s);
    assert.equal(decodeKnowledgeSegment(encoded), s, `round-trip failed for ${s}`);
    assert.ok(!/[<>:"/\\|?*]/.test(encoded), `unsafe character in ${encoded}`);
  }
  assert.notEqual(encodeKnowledgeSegment('con'), 'con');
  assert.notEqual(encodeKnowledgeSegment('aux.py'), 'aux.py');
});

test('workspace containment rejects traversal, sibling prefix, hidden/credential files, and oversized files', async () => {
  const parent = await makeTempDir('agentocode-ws-');
  const root = path.join(parent, 'repo');
  const sibling = path.join(parent, 'repo_other');
  await fsp.mkdir(root, { recursive: true });
  await fsp.mkdir(sibling, { recursive: true });

  await fsp.writeFile(path.join(root, 'app.py'), 'print(1)\n', 'utf8');
  await fsp.writeFile(path.join(root, '.env'), 'SECRET=123\n', 'utf8');
  await fsp.writeFile(path.join(root, '.env.example'), 'SECRET=\n', 'utf8');
  await fsp.writeFile(path.join(root, 'private.pem'), 'KEY', 'utf8');
  await fsp.writeFile(path.join(root, 'big.txt'), 'a'.repeat(MAX_FILE_BYTES + 10), 'utf8');
  await fsp.writeFile(path.join(root, 'image.bin'), Buffer.from([0, 1, 2, 3, 0, 4]));
  await fsp.writeFile(path.join(sibling, 'secret.txt'), 'outside', 'utf8');

  try {
    assert.throws(() => resolveContainedPath(root, '../repo_other/secret.txt'));
    assert.throws(() => resolveContainedPath(root, path.join(sibling, 'secret.txt')));
    assert.throws(() => resolveContainedPath(root, '.env'));
    assert.throws(() => resolveContainedPath(root, 'private.pem'));
    assert.throws(() => resolveContainedPath(root, 'bad\0path'));

    const okApp = resolveContainedPath(root, 'app.py');
    assert.equal(okApp.relativePath, 'app.py');

    assert.equal(isRelPathEligible('.env.example'), true);
    assert.equal(isRelPathEligible('.env.local'), false);
    assert.equal(isRelPathEligible('.npmrc'), false);
    assert.equal(isRelPathEligible('id_rsa'), false);
    assert.equal(isRelPathEligible('api_credentials.yaml'), false);
    assert.equal(isRelPathEligible('my_secret.json'), false);
    assert.equal(isRelPathEligible('.git/config'), false);
    assert.equal(isRelPathEligible('knowledge_base/index.md'), false);
    assert.equal(isRelPathEligible('.guardian/vectors/a'), false);
    assert.throws(() => resolveContainedPath(root, '../repo_other/new_file.ts', { allowNonExistent: true }));
    const okNew = resolveContainedPath(root, 'nested/brand_new.ts', { allowNonExistent: true });
    assert.equal(okNew.relativePath, 'nested/brand_new.ts');
    assert.equal(isProbablyBinaryBuffer(Buffer.from([0x61, 0x00, 0x62])), true);
    assert.equal(isProbablyBinaryBuffer(Buffer.from('hello world\n', 'utf8')), false);

    const scanned = await scanEligibleWorkspaceFiles(root);
    const rels = scanned.map((f) => f.relativePath);
    assert.deepEqual(rels, ['.env.example', 'app.py']);
  } finally {
    await fsp.rm(parent, { recursive: true, force: true });
  }
});

test('atomicWriteFileWithRetry creates parent dirs and overwrites atomically', async () => {
  const dir = await makeTempDir('agentocode-atomic-');
  const target = path.join(dir, 'nested', 'doc.md');
  try {
    await atomicWriteFileWithRetry(target, 'first');
    assert.equal(fs.readFileSync(target, 'utf8'), 'first');
    await atomicWriteFileWithRetry(target, 'second');
    assert.equal(fs.readFileSync(target, 'utf8'), 'second');
  } finally {
    await fsp.rm(dir, { recursive: true, force: true });
  }
});

