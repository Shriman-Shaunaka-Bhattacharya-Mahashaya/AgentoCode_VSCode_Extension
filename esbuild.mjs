import esbuild from 'esbuild';
import fs from 'node:fs';
import path from 'node:path';

const production = process.argv.includes('--production');
const watch = process.argv.includes('--watch');
const testMode = process.argv.includes('--test');

const sharedNodeOptions = {
  bundle: true,
  platform: 'node',
  target: 'node22',
  format: 'cjs',
  sourcemap: !production,
  minify: production,
  logLevel: 'info',
};

async function run() {
  if (testMode) {
    fs.rmSync('dist-test', { recursive: true, force: true });
    const testFiles = fs
      .readdirSync('test')
      .filter((f) => f.endsWith('.test.ts'))
      .map((f) => path.join('test', f));

    // Also build the worker so worker/MCP integration tests run against the real compiled worker.
    await esbuild.build({
      ...sharedNodeOptions,
      entryPoints: ['src/worker/indexerWorker.ts'],
      outfile: 'dist/indexerWorker.cjs',
      external: [
        '@huggingface/transformers',
        'onnxruntime-node',
        'onnxruntime-web',
        'sharp',
      ],
    });

    await esbuild.build({
      ...sharedNodeOptions,
      entryPoints: testFiles,
      outdir: 'dist-test',
      outExtension: { '.js': '.cjs' },
      external: [
        'vscode',
        '@huggingface/transformers',
        'onnxruntime-node',
        'onnxruntime-web',
        'sharp',
      ],
    });
    return;
  }

  fs.rmSync('dist', { recursive: true, force: true });

  const contexts = await Promise.all([
    esbuild.context({
      ...sharedNodeOptions,
      entryPoints: ['src/extension.ts'],
      outfile: 'dist/extension.js',
      external: ['vscode'],
    }),
    esbuild.context({
      ...sharedNodeOptions,
      entryPoints: ['src/worker/indexerWorker.ts'],
      outfile: 'dist/indexerWorker.cjs',
      external: [
        '@huggingface/transformers',
        'onnxruntime-node',
        'onnxruntime-web',
        'sharp',
      ],
    }),
    esbuild.context({
      entryPoints: ['src/webview/main.ts'],
      outfile: 'dist/webview.js',
      bundle: true,
      platform: 'browser',
      target: 'es2022',
      format: 'iife',
      sourcemap: !production,
      minify: production,
      logLevel: 'info',
    }),
  ]);

  if (watch) {
    fs.mkdirSync('dist', { recursive: true });
    fs.copyFileSync('src/webview/styles.css', 'dist/webview.css');
    await Promise.all(contexts.map((ctx) => ctx.watch()));
    console.log('Watching AgentoCode extension, worker, and webview...');
  } else {
    await Promise.all(contexts.map((ctx) => ctx.rebuild()));
    fs.mkdirSync('dist', { recursive: true });
    fs.copyFileSync('src/webview/styles.css', 'dist/webview.css');
    await Promise.all(contexts.map((ctx) => ctx.dispose()));
  }
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
