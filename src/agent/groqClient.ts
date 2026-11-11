export interface OpenAIToolDefinition {
  type: 'function';
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}

export interface ChatToolCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    arguments: string;
  };
}

export type ChatCompletionMessage =
  | { role: 'system' | 'user'; content: string }
  | { role: 'assistant'; content: string | null; tool_calls?: ChatToolCall[] }
  | { role: 'tool'; tool_call_id: string; name: string; content: string };

export interface StreamCompletionResult {
  content: string;
  toolCalls: ChatToolCall[];
  finishReason: string | null;
}

export interface ChatCompletionClient {
  streamChatCompletion(params: {
    apiKey: string;
    model: string;
    messages: ChatCompletionMessage[];
    tools: OpenAIToolDefinition[];
    signal?: AbortSignal;
    onDelta?: (textDelta: string) => void;
  }): Promise<StreamCompletionResult>;
}

const GROQ_CHAT_URL = 'https://api.groq.com/openai/v1/chat/completions';

interface GroqSseChunk {
  choices?: Array<{
    delta?: {
      content?: string | null;
      tool_calls?: Array<{
        index?: number;
        id?: string;
        type?: string;
        function?: {
          name?: string;
          arguments?: string;
        };
      }>;
    };
    finish_reason?: string | null;
  }>;
  error?: { message?: string; type?: string; code?: string };
}

/**
 * Parse a single SSE line from Groq's streaming chat completions endpoint.
 */
export function parseSseDataLine(line: string): GroqSseChunk | 'DONE' | null {
  const trimmed = line.trim();
  if (!trimmed || !trimmed.startsWith('data:')) {
    return null;
  }
  const payload = trimmed.slice(5).trim();
  if (payload === '[DONE]') {
    return 'DONE';
  }
  return JSON.parse(payload) as GroqSseChunk;
}

/**
 * Streaming OpenAI-compatible Groq client built on Node's native fetch.
 */
export class GroqHttpClient implements ChatCompletionClient {
  constructor(private readonly endpointUrl: string = GROQ_CHAT_URL) {}

  async streamChatCompletion(params: {
    apiKey: string;
    model: string;
    messages: ChatCompletionMessage[];
    tools: OpenAIToolDefinition[];
    signal?: AbortSignal;
    onDelta?: (textDelta: string) => void;
  }): Promise<StreamCompletionResult> {
    const body: Record<string, unknown> = {
      model: params.model,
      messages: params.messages,
      stream: true,
      temperature: 0.2,
    };
    if (params.tools.length > 0) {
      body.tools = params.tools;
      body.tool_choice = 'auto';
    }

    const res = await fetch(this.endpointUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${params.apiKey}`,
      },
      body: JSON.stringify(body),
      signal: params.signal,
    });

    if (!res.ok) {
      const errText = await res.text().catch(() => '');
      let message = `Groq API request failed (${res.status} ${res.statusText})`;
      try {
        const parsed = JSON.parse(errText) as { error?: { message?: string } };
        if (parsed.error?.message) {
          message = `Groq API (${res.status}): ${parsed.error.message}`;
        }
      } catch {
        if (errText) {
          message = `${message}: ${errText.slice(0, 400)}`;
        }
      }
      throw new Error(message);
    }

    if (!res.body) {
      throw new Error('Groq API returned an empty response body.');
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    let content = '';
    let finishReason: string | null = null;
    const toolBuilders = new Map<number, { id: string; name: string; arguments: string }>();

    const processLines = (lines: string[]) => {
      for (const line of lines) {
        const parsed = parseSseDataLine(line);
        if (!parsed || parsed === 'DONE') {
          continue;
        }
        if (parsed.error?.message) {
          throw new Error(`Groq stream error: ${parsed.error.message}`);
        }
        const choice = parsed.choices?.[0];
        if (!choice) {
          continue;
        }
        if (choice.finish_reason) {
          finishReason = choice.finish_reason;
        }
        const delta = choice.delta;
        if (!delta) {
          continue;
        }
        if (typeof delta.content === 'string' && delta.content.length > 0) {
          content += delta.content;
          params.onDelta?.(delta.content);
        }
        if (Array.isArray(delta.tool_calls)) {
          for (const tc of delta.tool_calls) {
            const idx = tc.index ?? 0;
            const current = toolBuilders.get(idx) ?? {
              id: tc.id ?? `call_${idx}`,
              name: '',
              arguments: '',
            };
            if (tc.id) {
              current.id = tc.id;
            }
            if (tc.function?.name) {
              current.name += tc.function.name;
            }
            if (tc.function?.arguments) {
              current.arguments += tc.function.arguments;
            }
            toolBuilders.set(idx, current);
          }
        }
      }
    };

    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split(/\r?\n/);
      buffer = parts.pop() ?? '';
      processLines(parts);
    }

    if (buffer.trim()) {
      processLines([buffer]);
    }

    const toolCalls: ChatToolCall[] = Array.from(toolBuilders.entries())
      .sort((a, b) => a[0] - b[0])
      .map(([, builder]) => ({
        id: builder.id,
        type: 'function' as const,
        function: {
          name: builder.name,
          arguments: builder.arguments || '{}',
        },
      }));

    return { content, toolCalls, finishReason };
  }
}

