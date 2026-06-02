# QMD Index

**Query Markup Documents** — On-device hybrid search for markdown files with BM25, vector search, and LLM reranking.

Built on top of [@tobilu/qmd](https://github.com/tobi/qmd), QMD Index provides a user-friendly Web UI and Python CLI for managing and searching your personal knowledge base — fully local, no data leaves your machine.

## Features

- **Hybrid Search** — BM25 keyword search (FTS5) combined with semantic vector search
- **LLM Reranking** — Re-rank search results using a local reranker model for higher accuracy
- **Query Expansion** — Automatically expand queries with a local LLM to improve recall
- **Web UI** — Flask-based web interface with:
  - 📂 Browse & filter documents by collection
  - 🔍 Full-text and semantic search
  - 🏷️ Tag cloud navigation
  - 🔗 Document graph visualization
  - ↔️ Split-pane document comparison
  - ✏️ Inline editing with Markdown preview
- **Python CLI** — Search, list, and manage your index from the terminal
- **MCP Server** — Expose search as MCP tools for AI agent integration
- **Multi-collection** — Organize documents into separate collections with independent indexing
- **Fully Local** — All processing happens on-device using local LLM models

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Web UI (port 8090)                │
│  Flask server (server.py)  ───  HTML/JS (index.html) │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              qmd.py (Python CLI)                     │
│    Search · Index · List · Stats · Remove            │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│         @tobilu/qmd (Node.js Core Engine)            │
│   FTS5 · Vector Search · Embeddings · Reranking     │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              llama.cpp (local API servers)           │
│   Embedding API (:1278) · Reranker API (:1245)       │
└─────────────────────────────────────────────────────┘
```

## Prerequisites

- **Python 3.10+**
- **Bun** (for the `@tobilu/qmd` Node.js engine)
- **llama.cpp** server running with embedding and reranker models (or compatible OpenAI-compatible API)
- A GGUF **query expansion model** (optional, for query expansion feature)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/qmd-index.git
cd qmd-index
```

### 2. Install Python dependencies

```bash
pip install flask flask-cors requests
```

### 3. Install Node.js dependencies

```bash
bun install
```

### 4. Configure

Copy the example config and customize:

```bash
cp qmd.example.yml qmd.yml
```

Edit `qmd.yml` to point to your document collections and model endpoints. See [Configuration](#configuration) below.

### 5. Index your documents

```bash
python qmd.py index --all
```

### 6. Start the Web UI

```bash
cd webui
python server.py --port 8090
```

Open http://localhost:8090 in your browser.

## Configuration

### qmd.yml

```yaml
collections:
  my-docs:
    path: /path/to/your/markdown/files
    pattern: "**/*.md"
    ignore:
      - "node_modules/**"

models:
  embedding_model_url: "http://127.0.0.1:1278/v1"
  embedding_model_name: "your-embedding-model.gguf"
  embedding_dim: 1024
  reranker_model_url: "http://127.0.0.1:1245/v1"
  reranker_model_name: "your-reranker-model.gguf"
  query_expansion_model: "/path/to/query-expansion.gguf"

defaults:
  limit: 10
  min_score: 0.3
  rerank: true
```

### webui/settings.json

```json
{
  "embedding_url": "http://127.0.0.1:1278/v1/embeddings",
  "embedding_model": "your-embedding-model",
  "embedding_dim": 1024,
  "reranker_url": "http://127.0.0.1:1245/v1/rerank",
  "reranker_model": "your-reranker-model",
  "llm_url": "http://localhost:1234/v1",
  "llm_model": "your-llm-model",
  "llm_key": "",
  "llm_ctx": 32768
}
```

## CLI Usage

```bash
# List all collections with document counts
python qmd.py list

# Search (keyword BM25)
python qmd.py search "your query"

# Search with semantic reranking
python qmd.py search "your query" --semantic

# Show document details
python qmd.py show <doc_id>

# Index a specific collection
python qmd.py index --collection my-docs

# Index all collections
python qmd.py index --all

# Remove a document
python qmd.py remove <doc_id>

# Index statistics
python qmd.py stats
```

## Web UI Features

| Feature | Description |
|---|---|
| **Browse** | Navigate documents by collection, filter by name |
| **Search** | Full-text keyword search (FTS5) |
| **Semantic Search** | Vector-based semantic search with reranking |
| **Graph View** | Visualize document connections and backlinks |
| **Tag Cloud** | Browse and filter by tags |
| **Document Modal** | View, edit, and manage document content |
| **Split View** | Compare two documents side-by-side |
| **MCP Tools** | Expose search capabilities to AI agents |

## MCP Server

QMD Index includes an MCP (Model Context Protocol) server that allows AI agents to search your knowledge base:

```bash
python qmd_mcp_server.py
```

This exposes search, list, stats, and file operations as MCP tools.

## License

MIT
