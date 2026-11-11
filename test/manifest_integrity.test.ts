import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';

test('package.json manifest commands, menus, icons, and security declarations are consistent', () => {
  const pkgPath = path.resolve(process.cwd(), 'package.json');
  const pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf8')) as {
    main: string;
    capabilities: { untrustedWorkspaces?: { supported?: boolean } };
    contributes: {
      viewsContainers: { activitybar: Array<{ id: string; icon: string }> };
      views: Record<string, Array<{ id: string }>>;
      commands: Array<{ command: string }>;
      menus: Record<string, Array<{ command: string }>>;
      configuration: { properties: Record<string, unknown> };
    };
  };

  assert.equal(pkg.main, './dist/extension.js');
  assert.equal(pkg.capabilities.untrustedWorkspaces?.supported, false);

  for (const container of pkg.contributes.viewsContainers.activitybar) {
    assert.ok(fs.existsSync(path.resolve(process.cwd(), container.icon)), `missing icon: ${container.icon}`);
  }

  const commandIds = new Set(pkg.contributes.commands.map((c) => c.command));
  for (const [menuKey, items] of Object.entries(pkg.contributes.menus)) {
    for (const item of items) {
      assert.ok(commandIds.has(item.command), `menu ${menuKey} references unknown command ${item.command}`);
    }
  }

  assert.ok('agentocode.generationModel' in pkg.contributes.configuration.properties);
  assert.ok('agentocode.embedProvider' in pkg.contributes.configuration.properties);
  assert.ok('agentocode.autoSyncOnStartup' in pkg.contributes.configuration.properties);
});

test('.gitignore, README.md, local logs, and GitHub Actions CI workflow exist and enforce expected safety guards', () => {
  const gitignore = fs.readFileSync(path.resolve(process.cwd(), '.gitignore'), 'utf8');
  for (const expected of [
    'node_modules/',
    'dist/',
    'dist-test/',
    '.guardian/',
    'knowledge_base/',
    '.env',
    '*.vsix',
    'Prev_README.md',
    'decision_log.md',
    'problems.md',
  ]) {
    assert.ok(gitignore.includes(expected), `.gitignore missing ${expected}`);
  }

  const vscodeignore = fs.readFileSync(path.resolve(process.cwd(), '.vscodeignore'), 'utf8');
  assert.ok(vscodeignore.includes('.env'), '.vscodeignore must exclude .env');
  assert.ok(vscodeignore.includes('.github/**'), '.vscodeignore must exclude .github/**');

  const readme = fs.readFileSync(path.resolve(process.cwd(), 'README.md'), 'utf8');
  assert.ok(readme.includes('npm test'));
  assert.ok(readme.includes('npm run build'));
  assert.ok(!/prev_readme/i.test(readme), 'README.md must not reference Prev_README.md');
  assert.ok(!/\bindepend/i.test(readme), 'README.md must not contain the word "independent"');
  assert.ok(readme.includes('(#architecture)'), 'README.md should link to #architecture');
  assert.ok(readme.includes('(#user-instructions)'), 'README.md should link to #user-instructions');

  const ciPath = path.resolve(process.cwd(), '.github', 'workflows', 'ci.yml');
  assert.ok(fs.existsSync(ciPath), 'missing .github/workflows/ci.yml');
  const ci = fs.readFileSync(ciPath, 'utf8');
  assert.ok(ci.includes('ubuntu-latest'));
  assert.ok(ci.includes('windows-latest'));
  assert.ok(ci.includes('npm ci'));
  assert.ok(ci.includes('npm test'));
  assert.ok(ci.includes('npm run build'));
  assert.ok(ci.includes('docker build --target test'));
  assert.ok(ci.includes('docker build --target worker'));

  const dockerfile = fs.readFileSync(path.resolve(process.cwd(), 'Dockerfile'), 'utf8');
  assert.ok(dockerfile.includes('node:22-bookworm-slim'), 'Dockerfile must use Debian glibc for onnxruntime-node');
  assert.ok(dockerfile.includes('USER node'), 'worker container should run as non-root node user');

  const dockerignore = fs.readFileSync(path.resolve(process.cwd(), '.dockerignore'), 'utf8');
  for (const ignored of ['.env', 'Prev_README.md', 'decision_log.md', 'problems.md', 'node_modules/', 'dist/']) {
    assert.ok(dockerignore.includes(ignored), `.dockerignore missing ${ignored}`);
  }
  assert.ok(fs.existsSync(path.resolve(process.cwd(), 'docker-compose.yml')), 'missing docker-compose.yml');
});

