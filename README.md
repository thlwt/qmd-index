# QMD Index

**Query Markdown Documents** — On-device hybrid search + knowledge graph extraction for personal knowledge bases.

Built on top of [@tobilu/qmd](https://github.com/tobi/qmd), QMD Index adds a Web UI, Hyper-Extract knowledge graph, MCP server, Docker support, and full Hermes Agent integration — fully local, no data leaves your machine.

> **Canonical documentation for this machine's memory/knowledge system:**
> [`D:\agent-os-registry\README.md`](../agent-os-registry/README.md) (human) and
> [`D:\agent-os-registry\AGENTS.md`](../agent-os-registry/AGENTS.md) (agent).
> This project owns `index.sqlite` and is its **sole writer**. See also
> [`AGENTS.md`](AGENTS.md) for the agent-facing project guide.

## Features

- **Hybrid Search** — BM25 keyword (FTS5) + semantic vector search + LLM reranking; Chinese compound words handled via jieba segmentation + multi-variant FTS5 queries + LIKE fallback
- **Hyper-Extract Knowledge Graph** — LLM automatically extracts entities, relationships, and document links from your markdown files; browse and edit the graph visually
- **Unified Graph View** — QMD document graph merged with Hyper entity graph in a single vis-network visualization (diamond-shaped entity nodes, type-colored, purple relationship edges)
- **Entity-Aware Search** — When you search, results show matched entity tags; documents linked to matching entities are appended even if BM25 misses them
- **Hermes Agent API** — Dedicated `POST /api/hyper/agent/lookup` and `POST /api/hyper/agent/search` endpoints return structured JSON with entities + relationships + linked documents, optimized for AI agent consumption
- **Manual Editing** — Add/edit/delete entities and relationships from the Web UI with inline forms
- **SSE Real-Time Progress** — Batch extraction sends per-document progress via Server-Sent Events with an animated progress bar and elapsed timer
- **Three Agent Integration Modes:**
  - **MCP stdio** — Local `qmd_mcp_server.py` exposes 7 tools (search, get, collections, priority-search, hyper-lookup, hyper-search, hyper-doc-entities)
  - **MCP SSE** — Docker-based `qmd-mcp` service on `:8010/sse` for remote MCP connections
  - **REST API** — `POST /api/agent/search` returns unified QMD + Hyper results as clean JSON
- **MCP Server** — 7 tools: `query`, `get`, `list_collections`, `search_priority`, `hyper_lookup`, `hyper_search`, `hyper_doc_entities`
- **Web UI** — Flask-based interface with document browser, split-pane comparison, inline editing with Markdown preview, tag cloud, and graph visualization
- **Multi-collection** — Organize documents into separate collections
- **Deployment** — QMD-Index itself runs as **host-native python** on `:8090` (the sole writer of `index.sqlite`); the two Docker containers are `qmd-mcp` (MCP SSE on `:8010`) and `qmd-wiki-hermes` (integration layer on `:8091`). The old `qmd-webui` container is retired — it cannot load the Windows `sqlite-vec` vec0 DLL and cannot use SQLite WAL on a 9p bind mount.
- **Fully Local** — All processing uses local LLM, embedding, and reranker models

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Web UI (port 8090)                        │
│  Flask server (server.py)  ───  HTML/JS (index.html)             │
│    · Hybrid Search    · Hyper-Extract UI    · Graph View         │
│    · SSE Progress     · Entity/Relation CRUD · Tag Cloud         │
└──────────────────────┬───────────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────────┐
│                  qmd.py (Python CLI)                             │
│    Search · Index · List · Stats · Remove                        │
└──────────────────────┬───────────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────────┐
│              @tobilu/qmd (Node.js Core Engine)                   │
│   FTS5 · Vector Search · Embeddings · Reranking                 │
└──────────────────────┬───────────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────────┐
│                    SQLite (index.sqlite)                         │
│   documents · content · vectors · hyper_entities                │
│   hyper_relationships · hyper_doc_entities · FTS5 indexes       │
└───────┬─────────────────────────────────────────────────┬───────┘
        │                                                 │
┌───────▼──────────────┐                  ┌──────────────▼────────┐
│  qmd-mcp (port 8010) │                  │  llama.cpp / LM Studio│
│  MCP SSE Server      │                  │  LLM · Embedding      │
│  7 MCP tools         │                  │  Reranker · QE (opt)  │
└──────────────────────┘                  └───────────────────────┘
```

## Quick Start

### Prerequisites

- **Python 3.10+**
- **Bun** (for `@tobilu/qmd` Node.js engine)
- **LM Studio** or **llama.cpp** server with:
  - Embedding model (e.g., `gemma-300M`)
  - Reranker model (e.g., `qwen3-reranker-0.6b`)
  - (Optional) QE model for query expansion (e.g., `qmd-query-expansion-1.7B` on port 2782)
  - LLM for hyper-extract (e.g., `qwen/qwen3.5-9b`)

### Local Installation

```bash
# 1. Clone
git clone https://github.com/thlwt/qmd-index.git
cd qmd-index

# 2. Python deps
pip install flask flask-cors requests mcp jieba

# 3. Node.js deps
bun install

# 4. Configure
cp qmd.example.yml qmd.yml
# Edit qmd.yml with your doc paths and model endpoints
# Edit webui/settings.json with your model URLs

# 5. Index documents
python qmd.py index --all

# 6. Start Web UI
cd webui && python server.py --port 8090
```

Open http://localhost:8090.

### Docker (recommended for production)

```bash
# Make sure the LLM (:1235) and embedding/reranker models are running on the host
# Docker reaches the host via host.docker.internal

# 1. QMD-Index itself: host-native python on :8090 (NOT a container)
powershell -ExecutionPolicy Bypass -File D:\QMD-Index\webui\start_server.ps1

# 2. The two containers
cd D:\QMD-Index
docker compose up -d
```

Three services run:
- **QMD-Index** — http://localhost:8090 (Flask UI + REST API, host-native python, sole `index.sqlite` writer)
- **qmd-wiki-hermes** — http://localhost:8091 (integration layer; reads a read-only `/tmp` snapshot, delegates writes to :8090)
- **qmd-mcp** — http://localhost:8010/sse (MCP SSE endpoint)

## Configuration

### `webui/settings.json`

```json
{
  "embedding_url": "http://127.0.0.1:1278/v1/embeddings",
  "embedding_model": "embeddinggemma-300M-Q8_0.gguf",
  "embedding_dim": 768,
  "reranker_url": "http://127.0.0.1:1245/v1/rerank",
  "reranker_model": "qwen3-reranker-0.6b-q8_0.gguf",
  "llm_url": "http://127.0.0.1:1239/v1",
  "llm_model": "Qwen3VL-4B-Instruct-Q4_K_M.gguf",
  "llm_key": "",
  "llm_ctx": 32768,
  "query_expansion_url": "http://127.0.0.1:2782/v1",
  "query_expansion_model": "qmd-query-expansion-1.7B"
}
```

> **Query Expansion**: Port 2782 is **optional** — search falls back gracefully to the original query if QE is not running. To enable: start `llama-server.exe -m models\qmd-query-expansion-1.7B-q4_k_m.gguf --port 2782`.

Ports are read from `qmd.yml` by default. For Docker CPU mode override with env vars: `QMD_EMBEDDING_URL=http://127.0.0.1:2780/v1/embeddings`, `QMD_RERANKER_URL=http://127.0.0.1:2781/v1/rerank`.

> **Long documents** (>9,000 chars) are automatically split into multiple FTS5 chunks at paragraph/sentence boundaries. All chunks are joined transparently during search. Full content is preserved in the `content` table for vector embedding and LIKE fallback.

### `qmd.yml`

```yaml
collections:
  my-docs:
    path: /path/to/your/markdown/files
    pattern: "**/*.md"
    ignore:
      - "node_modules/**"

models:
  embedding_model_url: "http://127.0.0.1:1278/v1"
  embedding_model_name: "embeddinggemma-300M-Q8_0.gguf"
  embedding_dim: 768
  reranker_model_url: "http://127.0.0.1:1245/v1"
  reranker_model_name: "qwen3-reranker-0.6b-q8_0.gguf"
  query_expansion_model: "/path/to/query-expansion.gguf"

defaults:
  limit: 10
  min_score: 0.3
  rerank: true
```

## Web UI Guide

### Document Browser
- Browse documents by collection
- Filter by name
- View document details in modal
- Inline editing with Markdown preview
- Split-pane comparison

### Search
- **Keyword search** — Type a query, press Enter or click Search
- **Semantic toggle** — Enable for vector-based semantic search
- **Entity-aware results** — Results show colored entity badges; entity-linked docs appended at bottom with label
- **Collection filter** — Limit search to a specific collection

### Graph View
- Select "QMD Wiki-Graph" mode for document backlinks
- Select "Hyper Entities" mode for knowledge graph entities
- Hyper entities are diamond-shaped, type-colored
- Click a hyper entity node to see its detail panel with relationships and linked documents

### Hyper-Extract

**Extract entities from a collection:**
1. Go to "Hyper Extract" section
2. Click "Hyper Extract All" to process all documents
3. Watch real-time SSE progress bar with elapsed timer
4. Entities, relationships, and doc-links are stored in `hyper_*` tables

**Manual editing:**
- **Add Entity** — Click ➕ button, fill name/type/description
- **Edit Entity** — Click ✏️ on an entity, edit inline
- **Add Relationship** — Click ➕ in entity detail panel
- **Delete Relationship** — Click ✕ on a relationship row

## REST API Reference

### Search

```
GET /api/search?q=<query>&collection=<name>&limit=20&semantic=true
```

Returns BM25 results with entity enrichment (entity tags, matched entities, entity-matched docs).

### Agent Search (unified)

```
POST /api/agent/search
Content-Type: application/json

{
  "query": "golf course design",
  "collection": "skills",
  "limit": 5,
  "include_web": false
}
```

Response includes `sources.qmd` (BM25 docs) and `sources.entities` (hyper entities matching query, with relationships).

### Hyper-Extract

```
POST /api/hyper/status                          # Stats (entities, relationships, doc_links)
POST /api/hyper/extract/batch                   # SSE streaming batch extraction
POST /api/hyper/entities       {"name":"...", "type":"...", "description":"..."}
PUT  /api/hyper/entities/<id>  {"name":"...", ...}
DELETE /api/hyper/entities/<id>
POST /api/hyper/relationships  {"source_id":..., "target_id":..., "rel_type":"..."}
DELETE /api/hyper/relationships/<id>
GET  /api/hyper/entities                        # List all entities
GET  /api/hyper/entities/<id>                   # Entity detail + relationships + docs
GET  /api/graph?mode=hyper                      # vis-network graph data
```

### Agent Endpoints

```
POST /api/hyper/agent/lookup  {"entity_name":"..."}   # Fuzzy entity lookup + neighbors + docs
POST /api/hyper/agent/search  {"query":"..."}          # Keyword entity search
POST /api/agent/search        {"query":"...", ...}     # Unified QMD + Hyper search
```

## MCP Server: Three Modes

### Mode 1: Local stdio

Run on your machine and connect Hermes Agent via stdio:

```bash
python qmd_mcp_server.py
```

Hermes config:

```yaml
mcp_servers:
  qmd:
    command: python
    args: ["D:\\QMD-Index\\qmd_mcp_server.py"]
```

Available tools: `query`, `get`, `list_collections`, `search_priority`, `hyper_lookup`, `hyper_search`, `hyper_doc_entities`.

### Mode 2: Docker SSE (remote)

The `qmd-mcp` container runs on port 8010. Hermes connects via URL:

```yaml
mcp_servers:
  qmd:
    url: http://localhost:8010/sse
```

No Python environment needed on the client machine.

### Mode 3: REST API (HTTP)

No MCP setup needed — just POST to `http://localhost:8090/api/agent/search`:

```bash
curl -X POST http://localhost:8090/api/agent/search \
  -H "Content-Type: application/json" \
  -d '{"query": "golf", "limit": 5}'
```

## Docker Deployment

### Start

```bash
docker compose up -d
```

### Stop

```bash
docker compose down
```

### Rebuild after code changes

```bash
docker compose build --no-cache
docker compose up -d
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `LLM_URL` | (from settings.json) | Override LLM API URL (e.g., `http://host.docker.internal:5000/v1`) |
| `LLM_MODEL` | (from settings.json) | Override LLM model name |
| `QMD_EMBEDDING_URL` / `EMBEDDING_URL` | `http://127.0.0.1:1278/v1/embeddings` | Embedding service URL (used by MCP server); override to `2780` for Docker CPU mode |
| `QMD_DB_PATH` | `models/qmd/index.sqlite` | Database path override |
| `TAVILY_API_KEY` | (unset) | Tavily API key for web search fallback |

### Volumes

| Host path | Container path | Purpose |
|---|---|---|
| `./models` | `/app/models` | SQLite database + vector data |
| `./webui/settings.json` | `/app/webui/settings.json` | LLM/model configuration |

## Hyper-Extract Knowledge Graph Details

### Database Schema

The knowledge graph is stored in `index.sqlite` under three tables:

- **`hyper_entities`** — Nodes (id, name, type, description, metadata)
- **`hyper_relationships`** — Edges (source_id, target_id, rel_type, weight, context)
- **`hyper_doc_entities`** — Document-to-entity links (doc_id, entity_id, mentions, contexts)

### Extraction Pipeline

1. Documents are chunked and sent to the LLM with a structured extraction prompt
2. LLM returns JSON with `entities[]` and `relationships[]`
3. Extractor handles reasoning models (combines `content` + `reasoning_content`)
4. Balanced-brace JSON parser (`_extract_json_block`) extracts valid JSON even with surrounding text
5. Entities are upserted (deduplicated by name); any missing source/target entities are auto-created before relationship insertion
6. Doc-entity links are recorded with mention contexts
7. SSE events stream progress per document back to the client

### Reasoning Model Support

The extractor detects reasoning models (those that output thinking in `reasoning_content`) and:
1. Attempts to disable reasoning via `"reasoning": {"enabled": false}`
2. Falls back to concatenating `content` + `reasoning_content`
3. Uses `_extract_json_block` (balanced-brace counting) to find valid JSON anywhere in the response

## MCP Tool Reference

| Tool | Description |
|---|---|
| `query(query, collection, limit)` | BM25 FTS5 keyword search across documents |
| `get(path, lines)` | Retrieve full document content by path or ID |
| `list_collections()` | List all collections with document counts |
| `search_priority(query, collection, limit)` | Priority search: QMD BM25 → embedding → web fallback |
| `hyper_lookup(entity_name)` | Look up an entity with relationships + linked documents |
| `hyper_search(query, limit)` | Search entities by keyword with relationship/doc counts |
| `hyper_doc_entities(doc_id)` | Get all entities linked to a document |

## CLI Usage

```bash
# List all collections
python qmd.py list

# Search (keyword BM25 — Chinese compound words auto-segmented by jieba)
python qmd.py search "高尔夫球场灌溉系统设计"

# Search with semantic reranking
python qmd.py search "your query" --semantic

# Show document details
python qmd.py show <doc_id>

# Index a collection
python qmd.py index --collection my-docs

# Index all collections
python qmd.py index --all

# Remove a document
python qmd.py remove <doc_id>

# Index statistics
python qmd.py stats
```

## License

MIT
