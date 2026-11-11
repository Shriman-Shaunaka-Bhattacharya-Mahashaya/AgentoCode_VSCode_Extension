import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { getDefaultEnvironment, StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import { McpToolMetadata } from '../agent/agentLoop.js';
import { KnowledgeStatusSnapshot, RagSearchResponse } from '../types.js';

export type WorkerConnectionState = 'disconnected' | 'connecting' | 'connected' | 'error';

export interface McpWorkerLaunchSpec {
  command: string;
  args: string[];
  cwd: string;
  env: Record<string, string>;
}

/**
 * Supervises the persistent local Node MCP worker over stdio JSON-RPC.
 */
export class McpWorkerClient {
  private client: Client | null = null;
  private transport: StdioClientTransport | null = null;
  private state: WorkerConnectionState = 'disconnected';
  private connectPromise: Promise<void> | null = null;
  private cachedTools: McpToolMetadata[] = [];

  constructor(
    private readonly launchSpecFactory: () => McpWorkerLaunchSpec,
    private readonly onLog: (line: string) => void = () => undefined,
    private readonly onStateChange: (state: WorkerConnectionState) => void = () => undefined,
  ) {}

  getConnectionState(): WorkerConnectionState {
    return this.state;
  }

  private setState(next: WorkerConnectionState): void {
    this.state = next;
    this.onStateChange(next);
  }

  async ensureConnected(): Promise<void> {
    if (this.client && this.state === 'connected') {
      return;
    }
    if (this.connectPromise) {
      return this.connectPromise;
    }

    this.connectPromise = (async () => {
      this.setState('connecting');
      try {
        const spec = this.launchSpecFactory();
        const env: Record<string, string> = {
          ...getDefaultEnvironment(),
          ...spec.env,
        };

        const transport = new StdioClientTransport({
          command: spec.command,
          args: spec.args,
          cwd: spec.cwd,
          env,
          stderr: 'pipe',
        });

        if (transport.stderr) {
          transport.stderr.on('data', (chunk: Buffer | string) => {
            const text = chunk.toString();
            for (const line of text.split(/\r?\n/)) {
              if (line.trim()) {
                this.onLog(line);
              }
            }
          });
        }

        transport.onclose = () => {
          this.onLog('MCP local worker process closed.');
          this.client = null;
          this.transport = null;
          this.cachedTools = [];
          if (this.state !== 'error') {
            this.setState('disconnected');
          }
        };

        transport.onerror = (err: Error) => {
          this.onLog(`MCP transport error: ${err.message}`);
          this.setState('error');
        };

        const client = new Client(
          { name: 'agentocode-vscode-host', version: '0.1.0' },
          { capabilities: {} },
        );

        await client.connect(transport);
        this.transport = transport;
        this.client = client;

        const toolsRes = await client.listTools();
        this.cachedTools = toolsRes.tools.map((t) => ({
          name: t.name,
          description: t.description,
          inputSchema: t.inputSchema as Record<string, unknown> | undefined,
        }));

        this.setState('connected');
      } catch (err: unknown) {
        this.setState('error');
        const msg = err instanceof Error ? err.message : String(err);
        this.onLog(`Failed to connect MCP worker: ${msg}`);
        await this.dispose().catch(() => undefined);
        throw err;
      } finally {
        this.connectPromise = null;
      }
    })();

    return this.connectPromise;
  }

  async listTools(): Promise<McpToolMetadata[]> {
    await this.ensureConnected();
    if (this.cachedTools.length === 0 && this.client) {
      const res = await this.client.listTools();
      this.cachedTools = res.tools.map((t) => ({
        name: t.name,
        description: t.description,
        inputSchema: t.inputSchema as Record<string, unknown> | undefined,
      }));
    }
    return this.cachedTools;
  }

  async callToolText(name: string, args: Record<string, unknown> = {}): Promise<string> {
    await this.ensureConnected();
    if (!this.client) {
      throw new Error('MCP client is not connected.');
    }
    const res = await this.client.callTool({ name, arguments: args });
    const parts: string[] = [];
    if (Array.isArray(res.content)) {
      for (const item of res.content) {
        if (item && typeof item === 'object' && 'text' in item && typeof item.text === 'string') {
          parts.push(item.text);
        } else {
          parts.push(JSON.stringify(item));
        }
      }
    }
    const joined = parts.join('\n').trim();
    if (res.isError) {
      throw new Error(joined || `MCP tool "${name}" reported an error.`);
    }
    return joined;
  }

  async getStatus(): Promise<KnowledgeStatusSnapshot> {
    const raw = await this.callToolText('knowledge_status', {});
    return JSON.parse(raw) as KnowledgeStatusSnapshot;
  }

  async syncKnowledge(force = false): Promise<KnowledgeStatusSnapshot> {
    const raw = await this.callToolText('sync_knowledge', { force });
    return JSON.parse(raw) as KnowledgeStatusSnapshot;
  }

  async preloadModel(): Promise<KnowledgeStatusSnapshot> {
    const raw = await this.callToolText('preload_model', {});
    return JSON.parse(raw) as KnowledgeStatusSnapshot;
  }

  async ragSearch(query: string, topK = 6): Promise<RagSearchResponse> {
    const raw = await this.callToolText('rag_search', { query, top_k: topK });
    return JSON.parse(raw) as RagSearchResponse;
  }

  async restart(): Promise<void> {
    await this.dispose();
    await this.ensureConnected();
  }

  async dispose(): Promise<void> {
    this.cachedTools = [];
    const client = this.client;
    const transport = this.transport;
    this.client = null;
    this.transport = null;

    if (client) {
      await client.close().catch(() => undefined);
    }
    if (transport) {
      await transport.close().catch(() => undefined);
    }
    this.setState('disconnected');
  }
}

