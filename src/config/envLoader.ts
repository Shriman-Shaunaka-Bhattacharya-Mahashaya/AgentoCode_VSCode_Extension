import fs from 'node:fs';
import path from 'node:path';

export type ApiKeySource =
  | 'workspace_env'
  | 'extension_env'
  | 'process_env'
  | 'secret_storage'
  | 'none';

/**
 * Parse a .env file payload into key-value pairs cleanly without third-party dependencies.
 */
export function parseDotEnv(content: string): Record<string, string> {
  const result: Record<string, string> = {};
  const cleaned = content.replace(/^\uFEFF/, '');

  for (const rawLine of cleaned.split(/\r?\n/)) {
    const trimmed = rawLine.trim();
    if (!trimmed || trimmed.startsWith('#')) {
      continue;
    }

    const line = trimmed.startsWith('export ') ? trimmed.slice(7).trim() : trimmed;
    const eqIdx = line.indexOf('=');
    if (eqIdx <= 0) {
      continue;
    }

    const key = line.slice(0, eqIdx).trim();
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(key)) {
      continue;
    }

    let val = line.slice(eqIdx + 1).trim();
    if (
      (val.startsWith('"') && val.endsWith('"') && val.length >= 2) ||
      (val.startsWith("'") && val.endsWith("'") && val.length >= 2)
    ) {
      val = val.slice(1, -1);
    } else {
      const commentIdx = val.indexOf(' #');
      if (commentIdx >= 0) {
        val = val.slice(0, commentIdx).trim();
      }
    }

    result[key] = val.trim();
  }

  return result;
}

export function readEnvFileKey(envFilePath: string, key: string): string | undefined {
  if (!fs.existsSync(envFilePath)) {
    return undefined;
  }
  try {
    const raw = fs.readFileSync(envFilePath, 'utf8');
    const parsed = parseDotEnv(raw);
    const val = parsed[key]?.trim();
    return val ? val : undefined;
  } catch {
    return undefined;
  }
}

/**
 * Resolve GROQ_API_KEY in deterministic order:
 * 1. Workspace root .env
 * 2. Extension root .env
 * 3. Process environment (GROQ_API_KEY)
 * 4. VS Code SecretStorage fallback
 */
export async function resolveGroqApiKey(params: {
  workspaceRoot?: string;
  extensionPath: string;
  processEnv?: Record<string, string | undefined>;
  getSecret?: () => PromiseLike<string | undefined>;
}): Promise<{ apiKey: string | null; source: ApiKeySource }> {
  if (params.workspaceRoot) {
    const wsKey = readEnvFileKey(path.join(params.workspaceRoot, '.env'), 'GROQ_API_KEY');
    if (wsKey) {
      return { apiKey: wsKey, source: 'workspace_env' };
    }
  }

  const extKey = readEnvFileKey(path.join(params.extensionPath, '.env'), 'GROQ_API_KEY');
  if (extKey) {
    return { apiKey: extKey, source: 'extension_env' };
  }

  const envVal = (params.processEnv ?? process.env).GROQ_API_KEY?.trim();
  if (envVal) {
    return { apiKey: envVal, source: 'process_env' };
  }

  if (params.getSecret) {
    const secretVal = (await params.getSecret())?.trim();
    if (secretVal) {
      return { apiKey: secretVal, source: 'secret_storage' };
    }
  }

  return { apiKey: null, source: 'none' };
}
