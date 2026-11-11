import { EditProposal, ToolTrace } from '../types.js';
import {
  ChatCompletionClient,
  ChatCompletionMessage,
  OpenAIToolDefinition,
} from './groqClient.js';

export interface McpToolMetadata {
  name: string;
  description?: string;
  inputSchema?: Record<string, unknown>;
}

export interface AgentToolExecutor {
  listMcpTools(): Promise<McpToolMetadata[]>;
  callMcpTool(name: string, args: Record<string, unknown>): Promise<string>;
  getActiveContext(): Promise<string>;
  getDiagnostics(relativePath?: string): Promise<string>;
  proposeFileEdit(args: {
    path: string;
    content: string;
    rationale: string;
  }): Promise<{ summary: string; proposal: EditProposal }>;
}

export interface AgentRunCallbacks {
  onDelta: (delta: string) => void;
  onToolStart: (trace: ToolTrace) => void;
  onToolDone: (trace: ToolTrace) => void;
  onProposalCreated: (proposal: EditProposal) => void;
}

const BLOCKED_MCP_TOOLS = new Set(['write_file', 'delete_file']);

const IDE_NATIVE_TOOLS: OpenAIToolDefinition[] = [
  {
    type: 'function',
    function: {
      name: 'ide_get_active_context',
      description:
        'Read the user’s currently active VS Code editor path, selection range/text, and dirty buffer indicator.',
      parameters: {
        type: 'object',
        properties: {},
        additionalProperties: false,
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'ide_get_diagnostics',
      description:
        'Read active VS Code compiler/linter warnings and errors (Problems panel), optionally filtered to a relative file path.',
      parameters: {
        type: 'object',
        properties: {
          path: {
            type: 'string',
            description: 'Optional workspace-relative file path filter.',
          },
        },
        additionalProperties: false,
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'propose_file_edit',
      description:
        'Create a pending file edit proposal shown as a VS Code diff preview for user approval. Never overwrites files directly without approval.',
      parameters: {
        type: 'object',
        properties: {
          path: {
            type: 'string',
            description: 'Workspace-relative path of the file to update or create.',
          },
          content: {
            type: 'string',
            description: 'Complete proposed UTF-8 file contents.',
          },
          rationale: {
            type: 'string',
            description: 'Brief explanation of why this edit is proposed.',
          },
        },
        required: ['path', 'content', 'rationale'],
        additionalProperties: false,
      },
    },
  },
];

export const SYSTEM_PROMPT = [
  'You are AgentoCode, an interactive software engineering assistant running inside VS Code.',
  'Follow these strict operating rules:',
  '1. Ground every technical claim in inspected files or tool output. Never guess symbols, paths, or behavior.',
  '2. Prefer `rag_search` first to locate relevant knowledge concepts and source regions, then verify exact lines with `read_file` before answering or proposing edits.',
  '3. Note `retrievalMode` and `indexAvailable` returned by `rag_search`. If the semantic index is unavailable or lexical fallback ran, rely on `read_file` and `list_files` directly.',
  '4. Never mutate files directly. To create or modify a file, read its current contents first and call `propose_file_edit` with the complete updated file content so the user can review the diff and approve it.',
  '5. Do not claim an edit is applied on disk — a proposal remains pending until the user clicks Apply.',
  '6. You do not have a shell execution tool; tell the user clearly which tests or commands they should run to verify changes.',
].join('\n');

/**
 * Filter raw MCP tool schemas so raw filesystem write tools are never exposed directly to the model,
 * then merge IDE-native context, diagnostics, and diff-proposal tools.
 */
export function buildAgentToolDefinitions(mcpTools: McpToolMetadata[]): OpenAIToolDefinition[] {
  const safeMcpDefs: OpenAIToolDefinition[] = mcpTools
    .filter((tool) => !BLOCKED_MCP_TOOLS.has(tool.name))
    .map((tool) => ({
      type: 'function',
      function: {
        name: tool.name,
        description: tool.description ?? `MCP tool ${tool.name}`,
        parameters: tool.inputSchema ?? { type: 'object', properties: {} },
      },
    }));

  return [...safeMcpDefs, ...IDE_NATIVE_TOOLS];
}

function previewText(input: string, maxLen = 220): string {
  const compact = input.replace(/\s+/g, ' ').trim();
  return compact.length <= maxLen ? compact : `${compact.slice(0, maxLen - 3)}...`;
}

/**
 * Multi-turn Groq tool-calling orchestrator running strictly at query time inside the Extension Host.
 */
export class AgentLoop {
  constructor(
    private readonly client: ChatCompletionClient,
    private readonly executor: AgentToolExecutor,
  ) {}

  async runTurn(params: {
    apiKey: string;
    model: string;
    history: ChatCompletionMessage[];
    userPrompt: string;
    activeContextPrefix?: string;
    maxIterations: number;
    signal?: AbortSignal;
    callbacks: AgentRunCallbacks;
  }): Promise<{ assistantText: string; updatedHistory: ChatCompletionMessage[] }> {
    const mcpTools = await this.executor.listMcpTools();
    const tools = buildAgentToolDefinitions(mcpTools);

    const userContent = params.activeContextPrefix
      ? `${params.activeContextPrefix}\n\n${params.userPrompt}`
      : params.userPrompt;

    const workingHistory: ChatCompletionMessage[] = [
      ...params.history,
      { role: 'user', content: userContent },
    ];

    let finalAssistantText = '';
    const maxIters = Math.max(1, params.maxIterations);

    for (let iter = 0; iter < maxIters; iter++) {
      if (params.signal?.aborted) {
        throw new Error('Request canceled by user.');
      }

      const completion = await this.client.streamChatCompletion({
        apiKey: params.apiKey,
        model: params.model,
        messages: [{ role: 'system', content: SYSTEM_PROMPT }, ...workingHistory],
        tools,
        signal: params.signal,
        onDelta: (delta) => {
          params.callbacks.onDelta(delta);
        },
      });

      if (completion.content) {
        finalAssistantText += (finalAssistantText && iter > 0 ? '\n' : '') + completion.content;
      }

      if (completion.toolCalls.length === 0) {
        workingHistory.push({
          role: 'assistant',
          content: completion.content || '',
        });
        return { assistantText: finalAssistantText, updatedHistory: workingHistory };
      }

      workingHistory.push({
        role: 'assistant',
        content: completion.content || null,
        tool_calls: completion.toolCalls,
      });

      for (const tc of completion.toolCalls) {
        if (params.signal?.aborted) {
          throw new Error('Request canceled by user.');
        }
        const traceId = tc.id || `trace_${Date.now()}_${Math.random().toString(16).slice(2, 6)}`;
        let parsedArgs: Record<string, unknown> = {};
        let argsParseError: string | null = null;
        try {
          const parsed = JSON.parse(tc.function.arguments || '{}');
          if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
            parsedArgs = parsed as Record<string, unknown>;
          } else {
            argsParseError = 'Tool arguments must be a JSON object.';
          }
        } catch (err: unknown) {
          argsParseError = `Invalid JSON tool arguments: ${err instanceof Error ? err.message : String(err)}`;
        }

        const runningTrace: ToolTrace = {
          id: traceId,
          name: tc.function.name,
          argsPreview: previewText(tc.function.arguments || '{}', 140),
          status: 'running',
        };
        params.callbacks.onToolStart(runningTrace);

        let toolResultText = '';
        let status: 'done' | 'error' = 'done';

        try {
          if (argsParseError) {
            throw new Error(argsParseError);
          }
          if (BLOCKED_MCP_TOOLS.has(tc.function.name)) {
            throw new Error(
              `Direct disk mutation tool "${tc.function.name}" is disabled in VS Code mode. Use propose_file_edit instead.`,
            );
          }

          if (tc.function.name === 'ide_get_active_context') {
            toolResultText = await this.executor.getActiveContext();
          } else if (tc.function.name === 'ide_get_diagnostics') {
            const p = typeof parsedArgs.path === 'string' ? parsedArgs.path : undefined;
            toolResultText = await this.executor.getDiagnostics(p);
          } else if (tc.function.name === 'propose_file_edit') {
            const targetPath = typeof parsedArgs.path === 'string' ? parsedArgs.path : '';
            const content = typeof parsedArgs.content === 'string' ? parsedArgs.content : '';
            const rationale = typeof parsedArgs.rationale === 'string' ? parsedArgs.rationale : 'Proposed edit';
            if (!targetPath) {
              throw new Error('propose_file_edit requires non-empty "path".');
            }
            const res = await this.executor.proposeFileEdit({ path: targetPath, content, rationale });
            params.callbacks.onProposalCreated(res.proposal);
            toolResultText = res.summary;
          } else {
            toolResultText = await this.executor.callMcpTool(tc.function.name, parsedArgs);
          }
        } catch (err: unknown) {
          status = 'error';
          toolResultText = `Tool error (${tc.function.name}): ${err instanceof Error ? err.message : String(err)}`;
        }

        params.callbacks.onToolDone({
          ...runningTrace,
          status,
          resultPreview: previewText(toolResultText, 240),
        });

        workingHistory.push({
          role: 'tool',
          tool_call_id: tc.id,
          name: tc.function.name,
          content: toolResultText,
        });
      }
    }

    const limitMsg = `\n\n[Stopped after reaching maxAgentIterations=${maxIters}.]`;
    params.callbacks.onDelta(limitMsg);
    finalAssistantText += limitMsg;
    workingHistory.push({ role: 'assistant', content: finalAssistantText });
    return { assistantText: finalAssistantText, updatedHistory: workingHistory };
  }
}

