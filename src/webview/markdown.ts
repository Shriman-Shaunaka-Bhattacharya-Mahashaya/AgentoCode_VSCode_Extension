/**
 * Escape HTML special characters to prevent XSS in the chat webview.
 */
export function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/**
 * Render safe sanitized Markdown (fenced code blocks, inline code, bold, headings, lists).
 */
export function renderSafeMarkdown(raw: string): string {
  if (!raw) {
    return '';
  }
  const codeBlocks: string[] = [];
  let working = raw.replace(/```([a-zA-Z0-9_-]*)\r?\n([\s\S]*?)```/g, (_, lang: string, body: string) => {
    const idx = codeBlocks.length;
    const label = lang ? `<div class="code-lang">${escapeHtml(lang)}</div>` : '';
    codeBlocks.push(
      `<div class="code-wrap">${label}<pre><code>${escapeHtml(body.replace(/\n$/, ''))}</code></pre></div>`,
    );
    return `@@CODEBLOCK_${idx}@@`;
  });

  working = escapeHtml(working);
  working = working.replace(/`([^`\n]+)`/g, '<code>$1</code>');
  working = working.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
  working = working.replace(/^###\s+(.+)$/gm, '<h4>$1</h4>');
  working = working.replace(/^##\s+(.+)$/gm, '<h3>$1</h3>');
  working = working.replace(/^#\s+(.+)$/gm, '<h2>$1</h2>');
  working = working.replace(/^-\s+(.+)$/gm, '<li>$1</li>');
  working = working.replace(/(?:<li>[^\n]*<\/li>\r?\n?)+/g, (m) => `<ul>${m.replace(/\r?\n/g, '')}</ul>`);
  working = working.replace(/\r?\n/g, '<br/>');
  working = working.replace(/(<\/(?:h2|h3|h4|ul)>)<br\/>/g, '$1');

  working = working.replace(/@@CODEBLOCK_(\d+)@@/g, (_, idxStr: string) => {
    return codeBlocks[Number(idxStr)] ?? '';
  });
  return working;
}
