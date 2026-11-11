# AgentoCode for VS Code

AgentoCode is a local-indexed AI coding assistant for Visual Studio Code. It combines local static code analysis, local ONNX embeddings, and local vector search with query-time Groq reasoning and diff-previewed file edits. *(A standalone CLI version of AgentoCode also exists.)*

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [User Instructions](#user-instructions)
- [Knowledge, Indexing, and Retrieval](#knowledge-indexing-and-retrieval)
- [Configuration](#configuration)
- [Security and Workspace Boundaries](#security-and-workspace-boundaries)
- [Development, Testing, and CI](#development-testing-and-ci)
- [Known Scope and Limitations](#known-scope-and-limitations)

---

## Overview

AgentoCode keeps codebase indexing strictly local and reserves remote LLM calls strictly for interactive user queries:

- **Zero-LLM Local Indexing:** Workspace indexing extracts AST/structural facts, generates OKF v0.2 knowledge documents, and computes local embeddings without making Groq API calls.
- **Query-Time Reasoning:** Groq (`openai/gpt-oss-120b` by default) runs only when you submit a question or request in chat, reading `GROQ_API_KEY` from `.env` (gitignored) with fallback to environment variables or VS Code `SecretStorage`.
- **Local Semantic + Lexical Retrieval:** Runs `nomic-ai/nomic-embed-text-v1.5` locally via `@huggingface/transformers` ONNX (or optional local Ollama) with live-hash freshness validation and automatic lexical fallback when no vector index is built.
- **IDE-Native Diff Workflow:** File edits are never written directly to disk by background tools; every proposed change opens a VS Code diff preview (`vscode.diff`) that you approve or reject via undoable `WorkspaceEdit`s.
- **Editor Context Integration:** Attaches active file path, cursor position, selected code, unsaved dirty editor buffers, and VS Code `Problems` diagnostics during query-time reasoning.

---

## Architecture

AgentoCode splits IDE interaction from CPU-heavy embedding and vector storage across an explicit MCP stdio boundary:

```text
+-------------------------------------------------------------------+
| VS Code Extension Host (dist/extension.js)                        |
|  - Sidebar Webview Chat (ChatViewProvider)                        |
|  - Knowledge & Index Explorer (KnowledgeTreeProvider)             |
|  - Query-Time Groq Streaming Loop (GroqHttpClient + AgentLoop)    |
|  - Diff Preview & WorkspaceEdit Engine (DiffProposalManager)      |
|  - SecretStorage, Status Bar, Workspace Trust, Dirty Buffer Reads |
+---------------------------------+---------------------------------+
                                  | MCP JSON-RPC over stdio
                                  | (stdout reserved for protocol;
                                  |  logs on stderr & mcp-worker.log)
                                  v
+-------------------------------------------------------------------+
| Local MCP Indexer & Embedding Worker (dist/indexerWorker.cjs)     |
|  - Workspace Containment & Exclusions (workspace.ts)              |
|  - AST & Structural Fact Extraction (facts.ts)                    |
|  - Local Embedders: ONNX Nomic / Ollama / Lexical (embeddings.ts) |
|  - Fingerprinted Float32 Vector Store (vectorStore.ts)            |
|  - Incremental Sync & Freshness-Checked RAG (rag.ts)              |
+-------------------------------------------------------------------+
```

### Key Components

| Module | Responsibility |
| --- | --- |
| `src/extension.ts` | Activation, command registration, Status Bar updates, `SecretStorage`, Workspace Trust gating, and debounced save sync. |
| `src/views/chatViewProvider.ts`, `src/webview/main.ts` | CSP-hardened Sidebar Chat Webview with streaming responses, collapsible tool traces, slash commands, and inline diff approval cards. |
| `src/views/knowledgeTreeProvider.ts` | Sidebar TreeView showing model state, vector/concept counts, failures, pending edit proposals, OKF knowledge documents, and quick actions. |
| `src/agent/groqClient.ts`, `src/agent/agentLoop.ts` | Streaming SSE client for Groq Chat Completions and multi-turn tool orchestrator. |
| `src/editor/diffManager.ts`, `src/workspace/ideContext.ts` | Virtual diff documents, conflict checking, undoable `WorkspaceEdit` application, active selection extraction, and VS Code diagnostics lookup. |
| `src/mcp/workerClient.ts` | Manages the persistent Node worker subprocess (`node` on `PATH` or `ELECTRON_RUN_AS_NODE=1` fallback). |
| `src/worker/indexerWorker.ts` | Exposes `list_files`, `read_file`, `rag_search`, `read_concept`, `knowledge_status`, `sync_knowledge`, and `preload_model` over MCP stdio (raw disk `write_file` is omitted). |

---

## Getting Started

### Prerequisites

- **Node.js:** `v22+` (verified with `v22.14.0`)
- **VS Code:** `^1.95.0`
- **Groq API Key:** Required only when sending chat queries (not needed for local indexing or lexical search).

### Build & Launch Locally

```powershell
npm install
npm run build
```

1. Open this repository in VS Code.
2. Press `F5` (uses `.vscode/launch.json`) to launch an **Extension Development Host** window.
3. Open a trusted workspace folder in the Extension Development Host window.

---

## User Instructions

### Initial Setup in VS Code

1. Set `GROQ_API_KEY` in a local `.env` file (see `.env.example`; `.env` is excluded by `.gitignore` and `.vscodeignore`, and blocked from indexing/reading inside workspaces). You can place `.env` in your open workspace root or extension directory, or use `AgentoCode: Set Groq API Key` (`SecretStorage`) as a fallback.
2. Click the **AgentoCode** icon in the Activity Bar to open the **Chat** and **Knowledge & Index** views.
3. *(Optional)* Click **Preload Model** (or send `/preload` in chat) to download and warm `nomic-ai/nomic-embed-text-v1.5` (`q8` ONNX weights, cached once in VS Code global extension storage).
   - To run strictly offline without cached weights (`agentocode.localFilesOnly: true`) or without downloading ONNX weights, set `agentocode.embedProvider` to `"lexical_only"` or `"ollama"`.
4. Click **Sync** (or send `/sync` in chat) to parse workspace files and populate the local vector index.

### Reviewing and Applying Code Changes

When AgentoCode proposes a code change via `propose_file_edit`:

1. A **Pending Edit Proposal** card appears inline in chat and under **Pending Edit Proposals** in the **Knowledge & Index** view.
2. Click **Preview Diff** to inspect side-by-side changes in VS Code (`vscode.diff`).
3. Click **Apply** to apply the change via `WorkspaceEdit` (fully reversible with `Ctrl+Z`, with conflict detection if the file changed after proposal creation) or **Reject** to discard it.

### Chat Slash Commands

| Command | Behavior |
| --- | --- |
| `/sync` | Queue incremental local AST/structural indexing and vector updates (zero Groq calls). |
| `/sync --force` | Recompute all eligible files even when content hashes match. |
| `/status` | Display provider/model state, vector and concept counts, overlength exclusions, and failures. |
| `/preload` | Load and warm the configured local embedding model inside the persistent worker. |
| `/clear` | Clear chat UI messages and conversation history. |
| `/help` | List available chat slash commands. |

### Command Palette Commands

- `AgentoCode: Set Groq API Key` (`agentocode.setApiKey`)
- `AgentoCode: Clear Groq API Key` (`agentocode.clearApiKey`)
- `AgentoCode: Sync Knowledge & Vectors` (`agentocode.syncKnowledge`)
- `AgentoCode: Show Index & Model Status` (`agentocode.showStatus`)
- `AgentoCode: Clear Chat History` (`agentocode.clearChat`)
- `AgentoCode: Preload Local Embedding Model` (`agentocode.preloadModel`)
- `AgentoCode: Restart Local Indexer Worker` (`agentocode.restartWorker`)
- `AgentoCode: Open Backend & Worker Logs` (`agentocode.openLogs`)
- `AgentoCode: Preview Proposed Edit Diff` / `Apply Proposed Edit` / `Reject Proposed Edit`

---

## Knowledge, Indexing, and Retrieval

### Structural Analysis & Chunking (`src/worker/facts.ts`)

- **TypeScript / JavaScript** (`.ts`, `.tsx`, `.js`, `.jsx`, `.mjs`, `.cjs`): Parsed with the TypeScript Compiler API (`ts.createSourceFile`) to extract imports, exports, classes, interfaces, types, enums, and functions, splitting oversized files along top-level AST boundaries.
- **Python** (`.py`, `.pyi`): Scanned for module imports, classes, decorated definitions, and functions, splitting at top-level definition boundaries.
- **Markdown** (`.md`, `.mdx`): Split along heading boundaries (`#`–`####`).
- **Other Text Formats**: Split into line-bounded chunks with explicit structural limitation notices.
- **OKF v0.2 Documents**: Deterministic YAML-frontmatter Markdown files (`index.md`, `files/<encoded>/content.md`, `directories/<encoded>/content.md`) with reversible segment encoding for Windows reserved filenames (`con`, `prn`, `aux`, `nul`, `com1`–`com9`, `lpt1`–`lpt9`) and special characters.

### Incremental Lifecycle & Vector Store (`src/worker/rag.ts`, `src/worker/vectorStore.ts`)

- **Hash-Based Updates:** File modifications, additions, deletions, and local import-dependent files are tracked by SHA-256. No-op syncs perform zero embedding calls; file and directory deletions clean up knowledge documents and vectors immediately.
- **Operational Token Cap:** Preflight tokenization runs with truncation disabled against `agentocode.maxTokens` (default `2048`). Overlength chunks remain intact and visibly excluded (`unindexedOverlengthCount`) rather than silently truncated.
- **Fingerprinted Storage & Corruption Recovery:** Normalized `Float32Array` vectors are stored in `<stateDir>/vectors/<fingerprint>/` (`meta.json`, `records.json`, `vectors.bin`) with atomic writes and Windows file-lock retry backoff. Corrupted stores are quarantined under `.corrupt.<timestamp>` and rebuilt cleanly.
- **Freshness Checking & Fallback:** `rag_search` verifies stored vector SHA-256 hashes against live disk files and rejects stale hits. When vectors or embedding models are unavailable, search falls back transparently to lexical search with `indexAvailable: false` and `retrievalMode: "lexical_fallback"`.

---

## Configuration

All settings live under the `agentocode` namespace in VS Code Settings:

| Setting | Default | Description |
| --- | --- | --- |
| `agentocode.generationModel` | `"openai/gpt-oss-120b"` | Groq chat model used strictly at query time. |
| `agentocode.embedProvider` | `"transformers_onnx"` | Local embedding provider: `"transformers_onnx"`, `"ollama"`, or `"lexical_only"`. |
| `agentocode.embedModelId` | `"nomic-ai/nomic-embed-text-v1.5"` | Hugging Face ONNX model ID (or Ollama model name when using `"ollama"`). |
| `agentocode.embedModelRevision` | `"e9b6763023c676ca8431644204f50c2b100d9aab"` | Pinned Hugging Face revision/commit for `transformers_onnx`. |
| `agentocode.embedDtype` | `"q8"` | ONNX quantization dtype (`"q8"`, `"fp32"`, `"fp16"`, `"q4"`). |
| `agentocode.ollamaEndpoint` | `"http://127.0.0.1:11434"` | Local Ollama endpoint when `embedProvider` is `"ollama"`. |
| `agentocode.localFilesOnly` | `false` | Disable remote Hugging Face Hub downloads when loading ONNX weights. |
| `agentocode.maxTokens` | `2048` | Preflight token cap enforced without silent truncation (`256`–`8192`). |
| `agentocode.maxDistance` | `0.65` | Cosine-distance threshold (`1 - cosineSimilarity`) for semantic filtering. |
| `agentocode.stateLocation` | `"extensionStorage"` | Store vectors/manifest/logs in VS Code internal storage (`"extensionStorage"`) or workspace `.guardian/` (`"workspace"`). |
| `agentocode.writeKnowledgeDocs` | `true` | Materialize OKF v0.2 `knowledge_base/` Markdown documents during indexing. |
| `agentocode.knowledgeDocsInWorkspace` | `false` | Place `knowledge_base/` in workspace root instead of extension storage when `writeKnowledgeDocs` is true. |
| `agentocode.nodePath` | `""` | Custom Node.js executable path for the local worker (defaults to `node` on `PATH`, or Electron fallback). |
| `agentocode.autoSyncOnStartup` | `false` | Automatically run local indexing when opening a trusted workspace (zero Groq calls). |
| `agentocode.autoSyncOnSave` | `false` | Debounced local re-indexing (`1500 ms`) when workspace files are saved (zero Groq calls). |
| `agentocode.maxAgentIterations` | `12` | Maximum tool-calling iterations per user query (`1`–`30`). |

---

## Security and Workspace Boundaries

- **Workspace Trust Required:** `capabilities.untrustedWorkspaces.supported` is `false`. Indexing, worker spawning, `.env` reading, and chat are disabled until you trust the workspace.
- **Containment & Exclusions:** All workspace reads and edit proposals verify realpaths against the workspace root (`fs.realpathSync`), rejecting `..` traversals, outside-root symlinks/junctions, hidden directories (`.git`, `.venv`, `node_modules`, `.guardian`, `knowledge_base`), credential patterns (`.env`, `.env.*` excluding templates, `*.pem`, `*.key`, `id_rsa*`, `*secret*`, `*credentials*`), binary files, and files over `512,000` bytes. Thus `.env` containing `GROQ_API_KEY` cannot be indexed into vectors or read by model tools.
- **No Arbitrary Shell Tool:** The agent has no terminal/shell execution tool.
- **Dirty-Buffer Protection:** Tool reads check open unsaved VS Code documents first, and edit application warns before overwriting if a file changed after proposal generation.

---

## Development, Testing, and CI

### Scripts

| Command | Purpose |
| --- | --- |
| `npm run typecheck` | Run strict TypeScript checking (`tsc --noEmit`). |
| `npm test` | Typecheck, bundle tests and worker with `esbuild`, and run `node --test dist-test/*.test.cjs`. |
| `npm run build` | Typecheck and produce minified production bundles in `dist/`. |
| `npm run watch` | Watch and rebuild extension, worker, and webview bundles continuously. |

### Test Coverage (`test/`)

- `test/workspace.test.ts` — Path containment, exclusions, binary/size checks, reversible encoding, atomic retries.
- `test/facts.test.ts` — TypeScript AST, Python structural parsing, Markdown splitting, oversized fragmenting, OKF rendering.
- `test/vector_store_and_embeddings.test.ts` — Fingerprinting, normalization, vector persistence/deletion compaction, corrupt-store quarantine, model prefixes, local HTTP `OllamaEmbedder`, `TransformersOnnxEmbedder`, and `LexicalOnlyEmbedder`.
- `test/rag_and_vectors.test.ts` — Incremental sync, no-op caching, import-dependent refresh, staleness rejection, overlength tracking, and lexical fallback.
- `test/groq_and_agent_edges.test.ts` — Local HTTP SSE `GroqHttpClient` streaming/errors/abort, `AgentLoop` error/iteration bounds, and settings resolution.
- `test/agent_and_mcp.test.ts` — Tool filtering (`write_file` blocked), multi-turn `AgentLoop`, compiled worker MCP stdio round-trip (`lexical_only`), and `@huggingface/transformers` module loading under `localFilesOnly`.
- `test/mcp_worker_edges.test.ts` — Compiled worker containment errors, subdir deletion pruning, force sync, and raw stdout/stderr wire isolation.
- `test/ide_and_ui.test.ts` — Markdown XSS sanitization, `DiffProposalManager`, `ideContext`, and `KnowledgeTreeProvider` against a mocked `vscode` API shim.
- `test/extension_lifecycle.test.ts` — `activate`/`deactivate`, command registration, `SecretStorage`, Webview CSP/nonces, config-reload scoping, debounced save sync, and chat slash commands.
- `test/manifest_integrity.test.ts` — Manifest consistency, `.gitignore` safety checks, and CI workflow verification.

### GitHub Actions CI

`.github/workflows/ci.yml` runs on every `push` and `pull_request` across `ubuntu-latest` and `windows-latest` (Node 22):

1. `npm ci`
2. `npm run typecheck`
3. `npm test`
4. `npm run build`
5. `node --test dist-test/*.test.cjs` against the production worker bundle
6. Verification that all `dist/` runtime artifacts exist

---

## Known Scope and Limitations

- **Offline / Headless CI Scope:** Automated tests verify IDE contracts against a mocked `vscode` shim and validate HTTP/embedding protocols locally; live Hugging Face ONNX weight downloads, real VS Code Electron Extension Host runs, and live Groq network calls require user runtime network access and credentials.
- **Single-Root Workspace Focus:** Commands and indexing currently target the first open workspace folder (`workspaceFolders[0]`).
- **Manual Verification in Terminal:** Because no shell execution tool is exposed to the agent, run project builds and test suites manually in the VS Code integrated terminal.
