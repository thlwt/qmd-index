#!/usr/bin/env python3
"""
QMD Index WebUI Backend Server
Flask-based API server that wraps qmd.py commands and exposes JSON REST endpoints.

Usage:
    python server.py              # Start on port 8090
    python server.py --port 9000  # Custom port
"""

import os
import sys
import json
import sqlite3
import subprocess
import hashlib
import struct
import requests
from datetime import datetime, timezone
from pathlib import Path

# Ensure webui directory is in sys.path for hyper_extract import
_webui_dir = Path(__file__).resolve().parent
if str(_webui_dir) not in sys.path:
    sys.path.insert(0, str(_webui_dir))

from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS

from hyper_extract.api import hyper_api

# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
QMD_PY = BASE_DIR / "qmd.py"
INDEX_PATH = None  # auto-detected by qmd.py
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder="static", static_url_path="/")
CORS(app)

app.register_blueprint(hyper_api)

PORT = int(os.environ.get("QMD_PORT", 8090))

# Global job registry for async tasks (embed, batch add, etc.)
_jobs = {}
import uuid as _uuid
import threading as _threading


# ============================================================
# Helper: run qmd.py and parse JSON output
# ============================================================

def _run_qmd(*args, timeout=60):
    """Run a qmd.py command and return parsed JSON."""
    cmd = [sys.executable, str(QMD_PY)] + list(args)
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            cmd, capture_output=True, encoding='utf-8', timeout=timeout, cwd=str(BASE_DIR), env=env
        )
        if result.returncode != 0:
            return {"error": result.stderr.strip() or f"exit code {result.returncode}"}
        # Try to parse JSON output; if not, wrap stdout in a safe response
        try:
            return json.loads(result.stdout.strip())
        except (json.JSONDecodeError, ValueError):
            return {"output": result.stdout.strip()}
    except subprocess.TimeoutExpired:
        return {"error": "Command timed out"}
    except Exception as e:
        return {"error": str(e)}


def _run_qmd_raw(*args, timeout=60):
    """Run qmd.py and return raw stdout."""
    cmd = [sys.executable, str(QMD_PY)] + list(args)
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            cmd, capture_output=True, encoding='utf-8', timeout=timeout, cwd=str(BASE_DIR), env=env
        )
        if result.returncode != 0:
            return {"error": result.stderr.strip() or f"exit code {result.returncode}"}
        return {"output": result.stdout.strip()}
    except subprocess.TimeoutExpired:
        return {"error": "Command timed out"}
    except Exception as e:
        return {"error": str(e)}


def _find_db_path():
    """Find the QMD index database path."""
    r = _run_qmd_raw("stats")
    for line in (r.get("output", "") or "").split("\n"):
        if "Database:" in line:
            return line.split(":", 1)[1].strip().split("(")[0].strip()
    return None


def _chunk_text(text, max_chars=512):
    """Split text into chunks of max_chars. Returns list of (pos, end, text)."""
    chunks = []
    pos = 0
    while pos < len(text):
        end = min(pos + max_chars, len(text))
        chunks.append((pos, end, text[pos:end]))
        pos = end
    return chunks


# ============================================================
# WikiLink / Backlink / Graph helpers
# ============================================================

import re as _re

def _ensure_links_table(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS doc_links (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        from_doc_id INTEGER NOT NULL,
        from_title TEXT,
        to_doc_id INTEGER,
        to_title TEXT NOT NULL,
        resolved INTEGER DEFAULT 0
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_links_from ON doc_links(from_doc_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_links_to ON doc_links(to_doc_id)")

def _parse_wikilinks(text):
    return _re.findall(r'\[\[([^\]]+)\]\]', text)

def _group_docs_by_tag(doc_tags):
    """Given {doc_id: [tag, ...]}, return {tag: [doc_id, ...]}."""
    tag_docs = {}
    for tid, tags in doc_tags.items():
        for t in tags:
            tag_docs.setdefault(t, []).append(tid)
    return tag_docs

def _resolve_link_target(target, conn):
    target = target.strip()
    row = conn.execute("SELECT id FROM documents WHERE title=? AND active=1", (target,)).fetchone()
    if row: return row[0]
    row = conn.execute("SELECT id FROM documents WHERE path=? AND active=1", (target,)).fetchone()
    if row: return row[0]
    row = conn.execute("SELECT id FROM documents WHERE title LIKE ? AND active=1 LIMIT 1", (f"%{target}%",)).fetchone()
    if row: return row[0]
    return None


# ============================================================
# Frontmatter / Tag helpers
# ============================================================

def _ensure_tags_column(conn):
    """Add tags column to documents table if it doesn't exist."""
    try:
        conn.execute("ALTER TABLE documents ADD COLUMN tags TEXT DEFAULT ''")
    except Exception:
        pass  # column already exists

def _parse_frontmatter(text):
    """Extract YAML frontmatter tags from markdown text. Returns list of tag strings."""
    text = str(text or "")
    m = _re.match(r'^---\s*\n(.*?)\n---', text, _re.DOTALL)
    if not m:
        return []
    fm = m.group(1)
    tags = []
    # Format 1: tags: [a, b, c]
    m2 = _re.search(r'tags\s*:\s*\[([^\]]*)\]', fm)
    if m2:
        tags = [t.strip().strip('"\'') for t in m2.group(1).split(',') if t.strip()]
    else:
        # Format 2: tags:\n  - a\n  - b
        m3 = _re.search(r'tags\s*:\s*\n((?:\s+-\s+.+\n?)+)', fm)
        if m3:
            tags = _re.findall(r'-\s*(.+)', m3.group(1))
            tags = [t.strip().strip('"\'') for t in tags]
    return tags

def _extract_and_store_tags(doc_id, conn):
    """Extract tags from document content and store in documents.tags column."""
    _ensure_tags_column(conn)
    hash_row = conn.execute("SELECT hash FROM documents WHERE id=?", (doc_id,)).fetchone()
    if not hash_row:
        return []
    content_row = conn.execute("SELECT doc FROM content WHERE hash=?", (hash_row[0],)).fetchone()
    if not content_row or not content_row[0]:
        return []
    tags = _parse_frontmatter(str(content_row[0]))
    if tags:
        conn.execute("UPDATE documents SET tags=? WHERE id=?", (json.dumps(tags, ensure_ascii=False), doc_id))
    else:
        conn.execute("UPDATE documents SET tags='' WHERE id=?", (doc_id,))
    return tags


# ============================================================
# API Endpoints — covering ALL tobi/qmd features + our CRUD
# ============================================================

@app.route("/")
def index():
    resp = app.make_response(send_from_directory("static", "index.html"))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


# --- Core Search (maps to qmd.py) ---

@app.route("/api/search", methods=["GET"])
def api_search():
    """Keyword search (BM25). Maps to: qmd search"""
    query = request.args.get("q", "")
    if not query:
        return jsonify({"error": "Missing 'q' parameter"}), 400

    collection = request.args.get("collection") or None
    limit = int(request.args.get("limit", 20))
    use_semantic = request.args.get("semantic", "false").lower() == "true"

    # Build args
    args = ["search", query, "-n", str(limit)]
    if collection:
        args += ["-c", collection]
    if use_semantic:
        args.append("--semantic")

    result = _run_qmd_raw(*args)
    # Enrich with structured results (doc IDs) for clickable search results
    structured = _enrich_search_with_ids(result.get("output", ""), collection)
    if structured:
        result["structured"] = structured
    # Entity-aware enrichment: find docs linked to entities matching query
    conn = None
    try:
        db_path = _find_db_path()
        if db_path and os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            words = [w.strip().lower() for w in query.split() if len(w.strip()) > 1]
            if words:
                like_params = [f"%{w}%" for w in words]
                like_clauses = " OR ".join("e.name LIKE ?" for _ in words)
                matched_entities = conn.execute(
                    f"SELECT e.id, e.name, e.type FROM hyper_entities e WHERE {like_clauses} LIMIT 10",
                    like_params
                ).fetchall()
                if matched_entities:
                    result["matched_entities"] = [{"id": r[0], "name": r[1], "type": r[2]} for r in matched_entities]
                    entity_ids = [r[0] for r in matched_entities]
                    ph = ",".join("?" for _ in entity_ids)
                    linked = conn.execute(
                        f"SELECT de.doc_id, e.name, e.id FROM hyper_doc_entities de JOIN hyper_entities e ON e.id=de.entity_id WHERE de.entity_id IN ({ph})",
                        entity_ids
                    ).fetchall()
                    entity_doc_map = {}
                    for doc_id, ename, eid in linked:
                        entity_doc_map.setdefault(doc_id, []).append({"id": eid, "name": ename})
                    if entity_doc_map and structured is not None:
                        existing_ids = {r.get("id") for r in structured}
                        for r in structured:
                            if r.get("id") in entity_doc_map:
                                r["entity_matched"] = entity_doc_map[r["id"]]
                        for doc_id, ents in entity_doc_map.items():
                            if doc_id not in existing_ids:
                                doc_row = conn.execute(
                                    "SELECT id, title, path, collection FROM documents WHERE id=? AND active=1",
                                    (doc_id,)
                                ).fetchone()
                                if doc_row:
                                    structured.append({
                                        "id": doc_row[0],
                                        "title": doc_row[1] or f"Doc#{doc_row[0]}",
                                        "path": doc_row[2] or "",
                                        "collection": doc_row[3] or "",
                                        "score": 0.5,
                                        "entity_matched": ents,
                                    })
    except Exception:
        pass
    finally:
        if conn:
            conn.close()
    return jsonify(result)


def _enrich_search_with_ids(output_text, collection):
    """Parse qmd search text output and add doc IDs by matching paths in DB."""
    if not output_text:
        return None
    db_path = _find_db_path()
    if not db_path or not os.path.exists(db_path):
        return None
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        results = []
        current = {}
        for line in output_text.split("\n"):
            lm = line.strip()
            hm = _re.match(r'^\[(\d+)/(\d+)\]\s+(.+)$', line)
            if hm:
                if current.get("title"):
                    results.append(current)
                current = {"title": hm.group(3).strip(), "index": int(hm.group(1))}
            elif lm.startswith("File:") and current:
                current["path"] = lm[5:].strip()
            elif lm.startswith("Collection:") and current:
                current["collection"] = lm[11:].strip()
            elif lm.startswith("Score:") and current:
                try:
                    current["score"] = float(lm[6:].strip())
                except ValueError:
                    pass
        if current.get("title"):
            results.append(current)

        # Resolve IDs for each result
        for r in results:
            col = r.get("collection") or collection or ""
            p = r.get("path", "")
            if col and p:
                # DB uses forward slashes; search output also uses forward slashes
                basename = p.split("/")[-1].split("\\")[-1]
                # Exact match on full path
                row = cur.execute(
                    "SELECT id FROM documents WHERE collection=? AND active=1 AND path=?",
                    (col, p)
                ).fetchone()
                # Try LIKE with full directory prefix
                if not row:
                    dir_part = p[:p.rfind("/")] if "/" in p else ""
                    row = cur.execute(
                        "SELECT id FROM documents WHERE collection=? AND active=1 AND path LIKE ?",
                        (col, f"{dir_part}/{basename}" if dir_part else f"%{basename}")
                    ).fetchone()
                # Last resort: basename match (may return first match if basename not unique)
                if not row:
                    rows = cur.execute(
                        "SELECT id, path FROM documents WHERE collection=? AND active=1 AND path LIKE ?",
                        (col, f"%/{basename}")
                    ).fetchall()
                    if rows:
                        # Pick the one with shortest path (most likely the exact basename match)
                        rows.sort(key=lambda x: len(x[1]))
                        row = rows[0]
                if row:
                    r["id"] = row[0]
        # Enrich with hyper-entities
        for r in results:
            if r.get("id"):
                try:
                    ents = cur.execute("""
                        SELECT e.id, e.name, e.type
                        FROM hyper_entities e
                        JOIN hyper_doc_entities de ON de.entity_id = e.id
                        WHERE de.doc_id = ?
                        ORDER BY de.mentions DESC
                        LIMIT 8
                    """, (r["id"],)).fetchall()
                    if ents:
                        r["entities"] = [{"id": e[0], "name": e[1], "type": e[2]} for e in ents]
                except Exception:
                    pass
        return results
    except Exception:
        return None
    finally:
        if conn:
            conn.close()


@app.route("/api/search/json", methods=["GET"])
def api_search_json():
    """Structured JSON search — queries FTS5 directly and returns doc IDs."""
    query = request.args.get("q", "")
    if not query:
        return jsonify({"error": "Missing 'q' parameter"}), 400

    collection = request.args.get("collection") or None
    limit = int(request.args.get("limit", 20))

    db_path = _find_db_path()
    if not db_path or not os.path.exists(db_path):
        return jsonify({"error": "Cannot find index database"}), 500

    try:
        conn = sqlite3.connect(db_path)
        _ensure_tags_column(conn)
        cur = conn.cursor()

        # Check for FTS5 table
        tables = [r[0] for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        fts_table = next((t for t in tables if t == "documents_fts"), None)
        if not fts_table:
            conn.close()
            return jsonify({"error": "FTS5 table not found", "results": [], "count": 0})

        # Clean query for FTS5
        fts_query = " OR ".join(f'"{w}"' for w in query.strip().split() if w) or query

        params = [fts_query]
        coll_where = ""
        if collection:
            coll_where = "AND d.collection = ?"
            params.append(collection)

        # FTS5 MATCH query (must use table name directly, not alias)
        rows = cur.execute(f"""
            SELECT d.id, d.title, d.path, d.collection, d.tags
            FROM documents d
            WHERE d.id IN (
                SELECT rowid FROM [{fts_table}]
                WHERE [{fts_table}] MATCH ?
            )
            AND d.active = 1 {coll_where}
            LIMIT ?
        """, params + [limit]).fetchall()

        conn.close()

        results = []
        for r in rows:
            tags_list = []
            if r[4]:
                try:
                    tags_list = json.loads(r[4]) if isinstance(r[4], str) else r[4]
                except Exception:
                    pass
            results.append({
                "id": r[0],
                "title": r[1] or "",
                "path": r[2] or "",
                "collection": r[3] or "",
                "tags": tags_list,
                "score": round(1.0 - 0.1 * len(results), 4),
            })

        return jsonify({"query": query, "results": results, "count": len(results)})
    except Exception as e:
        return jsonify({"error": f"Search failed: {e}", "results": [], "count": 0})


# --- Semantic Search (vec type from tobi/qmd) ---

@app.route("/api/search/semantic", methods=["GET"])
def api_search_semantic():
    """Semantic search using embeddings. Maps to: qmd search --semantic"""
    query = request.args.get("q", "")
    if not query:
        return jsonify({"error": "Missing 'q' parameter"}), 400

    collection = request.args.get("collection") or None
    limit = int(request.args.get("limit", 15))

    args = ["search", query, "-n", str(limit), "--semantic"]
    if collection:
        args += ["-c", collection]

    result = _run_qmd_raw(*args)
    # Enrich with structured results (doc IDs) for clickable search results
    structured = _enrich_search_with_ids(result.get("output", ""), collection)
    if structured:
        result["structured"] = structured
    return jsonify(result)


@app.route("/api/search/vector", methods=["GET"])
def api_search_vector():
    """Fast vector ANN search using vec0 extension (two-phase, no JOIN)."""
    query = request.args.get("q", "")
    if not query:
        return jsonify({"error": "Missing 'q' parameter"}), 400

    collection = request.args.get("collection") or None
    limit = int(request.args.get("limit", 15))

    import sqlite3, urllib.request, urllib.error, struct

    # Find DB path
    db_path = None
    r = _run_qmd_raw("stats")
    for line in (r.get("output", "") or "").split("\n"):
        if "Database:" in line:
            db_path = line.split(":", 1)[1].strip().split("(")[0].strip()

    if not db_path or not os.path.exists(db_path):
        return jsonify({"error": "Cannot find index database"}), 500

    try:
        embed_url = SETTINGS.get("embedding_url", "http://127.0.0.1:8025/v1/embeddings")
        embed_model = SETTINGS.get("embedding_model", "embeddinggemma-300M-Q8_0")
        payload = json.dumps({"input": query, "model": embed_model}).encode("utf-8")
        req = urllib.request.Request(
            embed_url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            emb_data = json.loads(resp.read())
        if "data" not in emb_data or not emb_data["data"]:
            return jsonify({"error": "Embedding API returned no data", "raw": str(emb_data)[:500]}), 502
        vector = emb_data["data"][0]["embedding"]
    except Exception as e:
        return jsonify({"error": f"Embedding failed: {e}"}), 502
    
    # Two-phase vec0 search (no JOIN to avoid hang)
    try:
        conn = sqlite3.connect(db_path)
        conn.enable_load_extension(True)
        conn.execute("PRAGMA journal_mode=WAL")
        vec0_path = os.path.join(str(BASE_DIR), "node_modules", "sqlite-vec-windows-x64", "vec0.dll")
        if os.path.exists(vec0_path):
            conn.load_extension(vec0_path)

        # Phase 1: vec0 ANN match (schema: hash_seq TEXT PK, embedding float[768])
        vec_bytes = struct.pack(f"{len(vector)}f", *vector)
        vrows = conn.execute(
            "SELECT hash_seq, distance FROM vectors_vec "
            "WHERE embedding MATCH ? AND k=? ORDER BY distance ASC",
            (vec_bytes, limit * 5),
        ).fetchall()

        # Phase 2: resolve document details by hash (one row per document)
        results = []
        seen = set()
        for hash_seq, dist in vrows:
            doc_hash, _, seq_s = str(hash_seq).rpartition("_")
            if not doc_hash or doc_hash in seen:
                continue
            seen.add(doc_hash)
            drow = conn.execute(
                "SELECT id, path, collection FROM documents WHERE hash=? AND active=1 LIMIT 1",
                (doc_hash,),
            ).fetchone()
            crow = conn.execute("SELECT doc FROM content WHERE hash=?", (doc_hash,)).fetchone()
            try:
                seq = int(seq_s)
            except (ValueError, TypeError):
                seq = 0
            preview = (crow[0] or "")[seq * 512: seq * 512 + 512] if crow else ""
            results.append({
                "chunk_id": hash_seq,
                "chunk_index": seq,
                "doc_id": drow[0] if drow else None,
                "path": drow[1] if drow else "",
                "collection": drow[2] if drow else "",
                "content_preview": preview[:500],
                "score": round(1.0 - (dist or 0.0), 4),
            })
            if len(results) >= limit:
                break
        conn.close()

        return jsonify({"query": query, "results": results, "count": len(results)})
    except sqlite3.OperationalError as e:
        if "no such module" in str(e):
            return jsonify({"error": "vec0 extension not available", "detail": str(e)}), 501
        # Fallback: run qmd.py semantic search
        fallback = _run_qmd_raw("search", query, "-n", str(limit), "--semantic",
                                *(["-c", collection] if collection else []))
        return jsonify({"query": query, "fallback": True, "results": fallback.get("output", "")[:4000]})
    except Exception as e:
        return jsonify({"error": f"Vector search error: {e}"}), 500


# --- Hybrid Search (query type from tobi/qmd — lex + vec + hyde combined) ---

@app.route("/api/search/hybrid", methods=["POST"])
def api_search_hybrid():
    """Hybrid search: BM25 + semantic rerank. Maps to: qmd query"""
    data = request.get_json() or {}
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Missing 'query' parameter"}), 400

    collection = data.get("collection") or None
    limit = int(data.get("limit", 15))
    intent = data.get("intent", "")  # for query expansion (hyde)

    # Run BM25 first, then semantic rerank if available
    bm25_result = _run_qmd_raw("search", query, "-n", str(limit * 3),
                                *(["-c", collection] if collection else []))

    semantic_result = None
    if intent:
        # Query expansion with user-provided intent (hyde mode)
        expanded_query = f"{query} {intent}"
        semantic_result = _run_qmd_raw("search", expanded_query, "-n", str(limit * 3),
                                       *(["-c", collection] if collection else []))

    return jsonify({
        "query": query,
        "intent": intent,
        "bm25_results": bm25_result.get("output", "")[:4000],
        "semantic_results": semantic_result.get("output", "")[:4000] if semantic_result else None,
    })


# --- Document Retrieval (get / multi-get from tobi/qmd) ---

@app.route("/api/document/<int:doc_id>", methods=["GET"])
def api_get_document(doc_id):
    """Get a single document by ID. Maps to: qmd show"""
    result = _run_qmd_raw("show", str(doc_id))
    return jsonify(result)


@app.route("/api/search/resolve-id", methods=["POST"])
def api_search_resolve_id():
    """Resolve search result (collection + filename) to database doc ID.

    Used by the frontend after parsing search results that don't include IDs.
    Body: { collection, path }
    Returns: { id: <doc_id> }
    """
    import sqlite3
    data = request.get_json() or {}
    collection = data.get("collection", "")
    filename = data.get("path", "")

    if not collection or not filename:
        return jsonify({"error": "Missing 'collection' and/or 'path'"}), 400

    # Find DB path via stats output
    db_path = None
    r = _run_qmd_raw("stats")
    for line in (r.get("output", "") or "").split("\n"):
        if "Database:" in line:
            db_path = line.split(":", 1)[1].strip().split("(")[0].strip()

    if not db_path or not os.path.exists(db_path):
        return jsonify({"error": "Cannot find index database"}), 500

    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        # DB uses forward slashes; normalize input to match
        db_path_style = filename.replace("\\", "/")
        basename = db_path_style.split("/")[-1]

        # Strategy 1: exact match on full path
        row = cur.execute(
            "SELECT id FROM documents WHERE collection=? AND active=1 AND path=?",
            (collection, db_path_style)
        ).fetchone()

        # Strategy 2: LIKE match with basename + directory prefix
        if not row:
            dir_part = db_path_style[:db_path_style.rfind("/")] if "/" in db_path_style else ""
            like_pattern = f"{dir_part}/{basename}" if dir_part else basename
            row = cur.execute(
                "SELECT id FROM documents WHERE collection=? AND active=1 AND path LIKE ?",
                (collection, like_pattern)
            ).fetchone()

        # Strategy 3: match by basename only (handle leading path mismatch)
        if not row:
            rows = cur.execute(
                "SELECT id, path FROM documents WHERE collection=? AND active=1 AND path LIKE ? ORDER BY LENGTH(path) ASC LIMIT 3",
                (collection, f"%{basename}")
            ).fetchall()
            if rows:
                row = rows[0]

        conn.close()

        if row:
            return jsonify({"id": row[0]})
        else:
            return jsonify({"error": f"No document found for '{basename}' in collection '{collection}'"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/documents/multi-get", methods=["POST"])
def api_multi_get():
    """Batch get multiple documents. Maps to: qmd multi-get (simulated via multiple show calls)."""
    data = request.get_json() or {}
    doc_ids = data.get("doc_ids", [])

    if not doc_ids and "glob" in data:
        # Support glob pattern: find docs matching pattern by listing all collections
        return jsonify({"error": "Glob-based multi-get requires implementing collection iteration"})

    results = []
    for did in doc_ids[:50]:  # limit batch size
        r = _run_qmd_raw("show", str(did))
        if "error" not in r:
            results.append(r)

    return jsonify({"results": results, "errors": len(doc_ids) - len(results)})


# --- Collections (collection add/list from tobi/qmd) ---

@app.route("/api/collections", methods=["GET"])
def api_collections():
    """List all collections. Maps to: qmd list"""
    result = _run_qmd_raw("list")
    return jsonify(result)


@app.route("/api/collections/add", methods=["POST"])
def api_add_collection():
    """Add a directory as a new collection. Maps to: qmd collection add + qmd add."""
    data = request.get_json() or {}
    path = data.get("path", "")
    name = data.get("name", "")

    if not path or not name:
        return jsonify({"error": "Missing 'path' and/or 'name'"}), 400

    if not os.path.exists(path):
        return jsonify({"error": f"Path not found: {path}"}), 400

    # Add files from directory to collection
    result = _run_qmd_raw("add", path, "-c", name, "-r")
    return jsonify(result)


# --- Context Management (context add from tobi/qmd) ---

@app.route("/api/context", methods=["POST"])
def api_add_context():
    """Add context metadata for a collection. Maps to: qmd context add."""
    data = request.get_json() or {}
    collection_name = data.get("collection", "")
    description = data.get("description", "")

    if not collection_name or not description:
        return jsonify({"error": "Missing 'collection' and/or 'description'"}), 400

    # Store context in our store_collections table via direct SQL
    import sqlite3
    db_path = None
    try:
        r = _run_qmd_raw("stats")
        if "error" not in r:
            for line in r.get("output", "").split("\n"):
                if "Database:" in line:
                    db_path = line.split(":", 1)[1].strip().split("(")[0].strip()
    except Exception:
        pass

    # Fallback: use qmd.py's internal path resolution via a small Python call
    import subprocess as sp
    r2 = sp.run([sys.executable, str(QMD_PY), "stats"], capture_output=True, text=True, timeout=10)
    for line in (r2.stdout or "").split("\n"):
        if "Database:" in line:
            db_path = line.split(":", 1)[1].strip().split("(")[0].strip()

    if not db_path or not os.path.exists(db_path):
        return jsonify({"error": f"Cannot find index database at: {db_path}"}), 500

    try:
        conn = sqlite3.connect(db_path)
        # Check if context column exists in store_collections
        cur = conn.cursor()
        cur.execute("SELECT * FROM store_collections WHERE name=?", (collection_name,))
        row = cur.fetchone()
        now = datetime.now().isoformat() + "Z"

        if row:
            # Update existing collection's context
            cur.execute(
                "UPDATE store_collections SET context=?, updated_at=? WHERE name=?",
                (description, now, collection_name)
            )
        else:
            # Insert new context entry
            try:
                cur.execute(
                    "INSERT INTO store_collections (name, path, pattern, include_by_default, context) VALUES (?, ?, ?, 1, ?)",
                    (collection_name, "", "**/*.md", description)
                )
            except Exception:
                # Table might have different schema — try alternative
                cur.execute(
                    "INSERT OR REPLACE INTO store_collections (name, value) VALUES (?, ?)",
                    (f"context:{collection_name}", description)
                )

        conn.commit()
        conn.close()
        return jsonify({"success": True, "message": f"Context set for '{collection_name}'"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/context/<name>", methods=["GET"])
def api_get_context(name):
    """Get context metadata for a collection."""
    import sqlite3

    # Find DB path
    db_path = None
    r = _run_qmd_raw("stats")
    for line in (r.get("output", "") or "").split("\n"):
        if "Database:" in line:
            db_path = line.split(":", 1)[1].strip().split("(")[0].strip()

    if not db_path or not os.path.exists(db_path):
        return jsonify({"error": f"Cannot find index database"}), 500

    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        # Try standard column first
        row = cur.execute(
            "SELECT context FROM store_collections WHERE name=?", (name,)
        ).fetchone()
        if not row:
            row = cur.execute(
                "SELECT value FROM store_collections WHERE name=?",
                (f"context:{name}",)
            ).fetchone()

        conn.close()
        return jsonify({"collection": name, "context": row[0] if row else ""})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Explain / Score Trace (explain from tobi/qmd) ---

@app.route("/api/search/explain", methods=["GET"])
def api_search_explain():
    """Search with score trace / explain mode. Shows retrieval breakdown."""
    query = request.args.get("q", "")
    if not query:
        return jsonify({"error": "Missing 'q' parameter"}), 400

    collection = request.args.get("collection") or None
    limit = int(request.args.get("limit", 10))

    # Run both BM25 and semantic, then combine results with explanation
    bm25_result = _run_qmd_raw("search", query, "-n", str(limit * 3),
                                *(["-c", collection] if collection else []))

    explain_data = {
        "query": query,
        "collection": collection or "all",
        "bm25_count": 0,
        "semantic_available": False,
        "results": [],
    }

    # Parse BM25 results for explanation
    output = bm25_result.get("output", "")
    lines = output.split("\n")
    doc_lines = []
    for line in lines:
        if line.startswith("[") and "/" in line:
            doc_lines.append(line)

    explain_data["bm25_count"] = len(doc_lines)

    # Try semantic search too
    try:
        sem_result = _run_qmd_raw("search", query, "-n", str(limit), "--semantic",
                                   *(["-c", collection] if collection else []))
        if "error" not in sem_result and "Generating embedding" not in sem_result.get("output", ""):
            explain_data["semantic_available"] = True
    except Exception:
        pass

    return jsonify(explain_data)


@app.route("/api/agent/search", methods=["POST"])
def api_agent_search():
    """Unified agent-friendly search: QMD BM25 + hyper-extract entities + web fallback.
    Returns clean structured JSON optimized for AI agent consumption.
    Request JSON: {"query": "...", "collection": "...", "limit": 10, "include_web": false}
    """
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or request.args.get("q", "")).strip()
    if not query:
        return jsonify({"error": "Missing 'query'"}), 400
    collection = data.get("collection") or request.args.get("collection") or None
    limit = int(data.get("limit", request.args.get("limit", 10)))
    include_web = data.get("include_web", False)

    result = {"query": query, "sources": {}}

    # 1. QMD BM25 search (via existing endpoint logic)
    try:
        args_list = ["search", query, "-n", str(limit)]
        if collection:
            args_list += ["-c", collection]
        raw = _run_qmd_raw(*args_list)
        qmd_docs = _enrich_search_with_ids(raw.get("output", ""), collection)
        if qmd_docs:
            result["sources"]["qmd"] = qmd_docs
    except Exception:
        pass

    # 2. Hyper-extract entities matched by query
    conn = None
    try:
        db_path = _find_db_path()
        if db_path and os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            words = [w.strip().lower() for w in query.split() if len(w.strip()) > 1]
            if words:
                like_params = [f"%{w}%" for w in words]
                like_clauses = " OR ".join("e.name LIKE ?" for _ in words)
                entities = conn.execute(
                    f"SELECT e.id, e.name, e.type, e.description FROM hyper_entities e WHERE {like_clauses} LIMIT 10",
                    like_params
                ).fetchall()
                if entities:
                    result["sources"]["entities"] = [
                        {"id": r[0], "name": r[1], "type": r[2], "description": r[3]}
                        for r in entities
                    ]
                    # For each entity, fetch top relationships
                    for ent in result["sources"]["entities"]:
                        rels = conn.execute(
                            """SELECT r.rel_type, s.name AS src, t.name AS tgt
                               FROM hyper_relationships r
                               JOIN hyper_entities s ON s.id = r.source_id
                               JOIN hyper_entities t ON t.id = r.target_id
                               WHERE r.source_id=? OR r.target_id=?
                               LIMIT 5""",
                            (ent["id"], ent["id"])
                        ).fetchall()
                        if rels:
                            ent["relationships"] = [
                                {"type": r[0], "source": r[1], "target": r[2]} for r in rels
                            ]
    except Exception:
        pass
    finally:
        if conn:
            conn.close()

    # 3. Web fallback via search_priority
    if include_web:
        try:
            tavily_key = os.environ.get("TAVILY_API_KEY", "").strip()
            if tavily_key:
                resp = requests.post(
                    "https://api.tavily.com/search",
                    json={"api_key": tavily_key, "query": query,
                           "search_depth": "basic", "max_results": limit,
                           "include_answer": True},
                    timeout=15)
                web_data = resp.json()
                web_results = [{"title": item.get("title", ""),
                                "url": item.get("url", ""),
                                "content": item.get("content", "")[:500],
                                "source": "tavily"}
                               for item in web_data.get("results", [])]
                if web_data.get("answer"):
                    web_results.insert(0, {"title": "AI Answer", "url": "",
                                            "content": web_data["answer"][:1000],
                                            "source": "tavily"})
                if web_results:
                    result["sources"]["web"] = web_results
        except Exception:
            pass

    return jsonify(result)

@app.route("/api/status", methods=["GET"])
def api_status():
    """Index health check. Maps to: qmd stats"""
    result = _run_qmd_raw("stats")
    if "error" in result:
        return jsonify(result), 500

   # Parse stats into structured JSON
    output = result.get("output", "")
    stats = {}

    for line in output.split("\n"):
        if "Documents (active):" in line:
            try:
                stats["documents"] = int(line.split(":")[1].strip())
            except ValueError:
                pass
        elif "Content entries:" in line:
            try:
                stats["content_entries"] = int(line.split(":")[1].strip())
            except ValueError:
                pass
        elif "Vector chunks:" in line:
            try:
                stats["vector_chunks"] = int(line.split(":")[1].strip())
            except ValueError:
                pass
        elif "Database:" in line:
            size_part = line.split("(")[1].rstrip(")") if "(" in line else ""
            stats["db_size_mb"] = float(size_part.replace("MB", "").strip()) if size_part else 0

    return jsonify(stats)


# --- CRUD Operations (add, remove, update, optimize, import) ---

@app.route("/api/add", methods=["POST"])
def api_add():
    """Add file(s) to index. Maps to: qmd add"""
    data = request.get_json() or {}
    paths = data.get("paths", [])  # batch mode: list of file/directory paths
    collection = data.get("collection") or None
    recursive = data.get("recursive", False)

    if not paths:
        return jsonify({"error": "Missing 'paths' (array of paths)"}), 400

    results = []
    errors = []
    for path in paths:
        args = ["add", path]
        if collection:
            args += ["-c", collection]
        if recursive:
            args.append("-r")
        result = _run_qmd_raw(*args)
        results.append({"path": path, **result})
        if "error" in result:
            errors.append({"path": path, "error": result["error"]})

    return jsonify({
        "success": len(errors) == 0,
        "total": len(paths),
        "added": len(paths) - len(errors),
        "errors": errors,
        "results": results,
    })


@app.route("/api/add/batch", methods=["POST"])
def api_add_batch():
    """Batch add files with progress tracking. Returns job_id for async polling."""
    data = request.get_json() or {}
    paths = data.get("paths", [])
    collection = data.get("collection") or None
    recursive = data.get("recursive", False)

    if not paths:
        return jsonify({"error": "Missing 'paths'"}), 400

    import time
    job_id = str(_uuid.uuid4())[:8]
    state = {
        "job_id": job_id,
        "total": len(paths),
        "current": 0,
        "status": "running",
        "results": [],
        "errors": [],
    }
    _jobs[job_id] = state

    def _process():
        for i, path in enumerate(paths):
            args = ["add", path]
            if collection:
                args += ["-c", collection]
            if recursive:
                args.append("-r")
            result = _run_qmd_raw(*args)
            s = _jobs.get(job_id)
            if s:
                s["current"] = i + 1
                s["results"].append({"path": path, **result})
                if "error" in result:
                    s["errors"].append({"path": path, "error": result["error"]})
            time.sleep(0.1)

        s = _jobs.get(job_id)
        if s:
            s["status"] = "partial" if s.get("errors") else "done"

    _threading.Thread(target=_process, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/progress/<job_id>", methods=["GET"])
def api_progress(job_id):
    """Poll job progress (embed, batch add, etc.)."""
    state = _jobs.get(job_id)
    if state is None:
        return jsonify({"job_id": job_id, "status": "unknown"}), 404
    return jsonify(state)


@app.route("/api/remove/<int:doc_id>", methods=["DELETE"])
def api_remove(doc_id):
    """Remove document. Maps to: qmd remove"""
    result = _run_qmd_raw("remove", str(doc_id))
    return jsonify(result)


@app.route("/api/document/<int:doc_id>/reindex", methods=["POST"])
def api_reindex_document(doc_id):
    """Remove document and re-add from its original file path, then optionally re-embed.

    Body: { "embed": true/false } (default: true)
    """
    data = request.get_json() or {}
    do_embed = data.get("embed", True)

    db_path = _find_db_path()
    if not db_path or not os.path.exists(db_path):
        return jsonify({"success": False, "error": "Cannot find index database"}), 500

    # Open connection for read + delete, then close before qmd.py runs (separate process)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        doc = conn.execute(
            "SELECT path, collection FROM documents WHERE id=? AND active=1", (doc_id,)
        ).fetchone()
    except Exception as e:
        conn.close()
        return jsonify({"success": False, "error": f"DB query failed: {e}"}), 500

    if not doc:
        conn.close()
        return jsonify({"success": False, "error": f"Document #{doc_id} not found"}), 404

    orig_path, collection = doc

    # Hard-delete from documents table (qmd remove is soft-delete only)
    conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
    conn.commit()
    conn.close()

    steps = [{"step": "remove", "status": "ok"}]

    # Resolve file path
    resolved = None
    for candidate in [orig_path, str(UPLOAD_DIR / orig_path), str(UPLOAD_DIR / os.path.basename(orig_path))]:
        if os.path.exists(candidate):
            resolved = candidate
            break
    if not resolved:
        return jsonify({"success": False, "error": f"Original file not found: {orig_path}"}), 404
    orig_path = resolved

    # Step 2: Re-add via qmd.py (separate process, owns its connection)
    add_args = ["add", orig_path]
    if collection:
        add_args += ["-c", collection]
    add_result = _run_qmd_raw(*add_args, timeout=120)
    if "error" in add_result:
        return jsonify({"success": False, "error": f"Re-add failed: {add_result['error']}", "steps": steps}), 500
    steps.append({"step": "add", "status": "ok"})

    # Step 3: Re-embed — open a fresh connection
    embed_info = None
    if do_embed:
        try:
            conn2 = sqlite3.connect(db_path)
            conn2.execute("PRAGMA journal_mode=WAL")
            basename = os.path.basename(orig_path)
            new_doc = conn2.execute(
                "SELECT id, hash FROM documents WHERE path=? AND active=1 ORDER BY id DESC LIMIT 1",
                (basename,)
            ).fetchone()
            if not new_doc:
                conn2.close()
                embed_info = {"error": f"could not find re-added doc (path={basename})"}
            else:
                new_id, doc_hash = new_doc
                content_row = conn2.execute(
                    "SELECT doc FROM content WHERE hash=?", (doc_hash,)
                ).fetchone()
                if not content_row:
                    conn2.close()
                    embed_info = {"error": "no content found for hash"}
                else:
                    doc_text = content_row[0]
                    chunks = _chunk_text(doc_text, 512)
                    batch_texts = [c[2] for c in chunks]

                    embed_url = SETTINGS.get("embedding_url", "http://127.0.0.1:8025/v1/embeddings")
                    embed_model = SETTINGS.get("embedding_model", "embeddinggemma-300M-Q8_0")
                    now = datetime.now(timezone.utc).isoformat()

                    conn2.enable_load_extension(True)
                    vec0_dll = os.path.join(str(BASE_DIR), "node_modules", "sqlite-vec-windows-x64", "vec0.dll")
                    if os.path.exists(vec0_dll):
                        conn2.load_extension(vec0_dll)

                    emb_resp = requests.post(embed_url, json={
                        "input": [t[:8000] for t in batch_texts],
                        "model": embed_model
                    }, timeout=300)
                    emb_resp.raise_for_status()
                    emb_data = emb_resp.json()
                    emb_list = sorted(emb_data["data"], key=lambda x: x["index"])

                    # Clean up old vector entries for this hash (re-index case)
                    conn2.execute("DELETE FROM vectors_vec WHERE hash_seq LIKE ?", (f"{doc_hash}_%",))
                    conn2.execute("DELETE FROM content_vectors WHERE hash=?", (doc_hash,))
                    conn2.commit()

                    done = 0
                    for seq, ((pos, end, text), emb_item) in enumerate(zip(chunks, emb_list)):
                        hs = f"{doc_hash}_{seq}"
                        embedding = emb_item["embedding"]
                        conn2.execute(
                            "INSERT INTO vectors_vec (hash_seq, embedding) VALUES (?, ?)",
                            (hs, json.dumps(embedding))
                        )
                        conn2.execute(
                            "INSERT INTO content_vectors (hash, seq, pos, model, embedded_at) VALUES (?, ?, ?, ?, ?)",
                            (doc_hash, seq, pos, embed_model, now)
                        )
                        done += 1

                    conn2.commit()
                    embed_info = {"done": done, "doc_id": new_id}
                    conn2.close()
        except Exception as e:
            embed_info = {"error": str(e)}

        steps.append({"step": "embed", "status": "ok" if embed_info and "done" in embed_info else "failed",
                      "detail": embed_info})

    return jsonify({
        "success": True,
        "doc_id": doc_id,
        "path": orig_path,
        "collection": collection,
        "steps": steps,
        "embed": embed_info,
    })


# ============================================================
# Graph / WikiLink / Backlinks API
# ============================================================

@app.route("/api/scan-links", methods=["POST"])
def api_scan_links():
    """Scan documents for [[wiki links]] and populate doc_links table."""
    data = request.get_json() or {}
    doc_id = data.get("doc_id")  # optional: scan only one doc

    db_path = _find_db_path()
    if not db_path or not os.path.exists(db_path):
        return jsonify({"error": "Cannot find index database"}), 500

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    _ensure_links_table(conn)

    if doc_id:
        docs = conn.execute(
            "SELECT id, title, path FROM documents WHERE id=? AND active=1", (doc_id,)
        ).fetchall()
    else:
        docs = conn.execute(
            "SELECT id, title, path FROM documents WHERE active=1 ORDER BY id"
        ).fetchall()

    # Clean up links from inactive documents
    conn.execute(
        "DELETE FROM doc_links WHERE from_doc_id IN (SELECT id FROM documents WHERE active=0)"
    )

    total_links = 0
    for doc_row in docs:
        did, dtitle, dpath = doc_row
        # Clear old links for this doc before re-scanning
        conn.execute("DELETE FROM doc_links WHERE from_doc_id=?", (did,))

        # Get content to scan for links
        hash_row = conn.execute(
            "SELECT hash FROM documents WHERE id=?", (did,)
        ).fetchone()
        if not hash_row:
            continue
        content_row = conn.execute(
            "SELECT doc FROM content WHERE hash=?", (hash_row[0],)
        ).fetchone()
        if not content_row or not content_row[0]:
            continue

        links = _parse_wikilinks(str(content_row[0]))
        if not links:
            continue

        for target in links:
            to_id = _resolve_link_target(target, conn)
            conn.execute(
                "INSERT INTO doc_links (from_doc_id, from_title, to_doc_id, to_title, resolved) VALUES (?, ?, ?, ?, ?)",
                (did, dtitle or dpath, to_id, target, 1 if to_id else 0)
            )
            total_links += 1

    conn.commit()
    conn.close()
    return jsonify({"success": True, "scanned": len(docs), "links_found": total_links})


@app.route("/api/graph", methods=["GET"])
def api_graph():
    """Return graph data (nodes + edges) for visualization.

    Query params:
      mode: 'all' (default) | 'wiki' | 'collection'
        - wiki: wiki link edges + tag-based edges only
        - collection: virtual collection nodes + edges from each doc to its collection
        - all: wiki + tag + collection edges combined
      detail: 'high' (default) | 'low'
        - high: all individual doc nodes (heavy for large datasets)
        - low: aggregated collection clusters only (fast, useful for 500+ docs)
    """
    mode = request.args.get("mode", "all")
    detail = request.args.get("detail", "high")

    db_path = _find_db_path()
    if not db_path or not os.path.exists(db_path):
        return jsonify({"error": "Cannot find index database"}), 500

    conn = sqlite3.connect(db_path)
    _ensure_links_table(conn)

    # All active documents as nodes
    nodes = conn.execute(
        "SELECT id, title, collection, path, tags FROM documents WHERE active=1 ORDER BY id"
    ).fetchall()
    conn.close()

    node_list = []
    edge_list = []
    existing_pairs = set()
    low_detail = (detail == "low")
    is_collection_mode = mode in ("collection", "all")

    # ── Build per-collection doc lists and tag info ──
    collection_docs = {}
    doc_collection = {}
    doc_tags = {}
    for n in nodes:
        nid, title, col, path, raw_tags = n
        col = col or "_uncategorized"
        doc_collection[nid] = col
        collection_docs.setdefault(col, []).append(nid)

        if raw_tags:
            try:
                parsed = json.loads(raw_tags)
                if isinstance(parsed, list) and len(parsed) > 0:
                    doc_tags[nid] = parsed
            except (json.JSONDecodeError, TypeError):
                pass

    if low_detail and is_collection_mode:
        # ── Aggregated view: only collection cluster nodes ──
        # Virtual collection nodes
        col_node_id_base = 10_000_000
        col_idx = {}
        for ci, (col_name, doc_ids) in enumerate(sorted(collection_docs.items())):
            if len(doc_ids) < 2:
                continue
            virtual_id = col_node_id_base + ci
            col_idx[col_name] = virtual_id
            node_list.append({
                "id": virtual_id,
                "title": f"📁 {col_name}",
                "collection": col_name,
                "path": "",
                "virtual": True,
                "doc_count": len(doc_ids),
            })

        # Inter-collection edges via shared tags
        # Map each tag → list of collection names that have docs with that tag
        tag_collections = {}
        for nid, tags in doc_tags.items():
            col = doc_collection[nid]
            for t in tags:
                tag_collections.setdefault(t, set()).add(col)

        for tag, cols in tag_collections.items():
            cols = sorted(cols)
            if len(cols) < 2:
                continue
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    a = col_idx.get(cols[i])
                    b = col_idx.get(cols[j])
                    if a and b:
                        pair = (a, b)
                        if pair not in existing_pairs:
                            edge_list.append({"from": a, "to": b, "type": "tag", "tag": tag})
                            existing_pairs.add(pair)
                            existing_pairs.add((b, a))

        # Include wiki edges aggregated at collection level
        conn2 = sqlite3.connect(db_path)
        wiki_edges = conn2.execute(
            """SELECT dl.from_doc_id, dl.to_doc_id
               FROM doc_links dl
               INNER JOIN documents df ON df.id = dl.from_doc_id AND df.active = 1
               LEFT JOIN documents dt ON dt.id = dl.to_doc_id
               WHERE dl.resolved = 1
                 AND (dt.id IS NULL OR dt.active = 1)"""
        ).fetchall()
        conn2.close()

        for from_id, to_id in wiki_edges:
            from_col = doc_collection.get(from_id)
            to_col = doc_collection.get(to_id)
            a = col_idx.get(from_col)
            b = col_idx.get(to_col)
            if a and b and a != b:
                pair = (a, b)
                if pair not in existing_pairs:
                    edge_list.append({"from": a, "to": b, "type": "wiki"})
                    existing_pairs.add(pair)
                    existing_pairs.add((b, a))

        wiki_edge_count = sum(1 for e in edge_list if e["type"] == "wiki")
        tag_edge_count = sum(1 for e in edge_list if e["type"] == "tag")

        return jsonify({
            "nodes": node_list,
            "edges": edge_list,
            "node_count": len(node_list),
            "edge_count": len(edge_list),
            "wiki_edge_count": wiki_edge_count,
            "tag_edge_count": tag_edge_count,
            "collection_edge_count": 0,
            "mode": mode,
            "detail": "low",
            "aggregated": True,
        })

    # ── High-detail view: all individual nodes ──
    node_list = [{"id": n[0], "title": n[1] or f"Doc#{n[0]}", "collection": n[2] or "", "path": n[3] or ""} for n in nodes]

    if mode in ("wiki", "all"):
        conn2 = sqlite3.connect(db_path)
        wiki_edges = conn2.execute(
            """SELECT dl.from_doc_id, dl.from_title, dl.to_doc_id, dl.to_title
               FROM doc_links dl
               INNER JOIN documents df ON df.id = dl.from_doc_id AND df.active = 1
               LEFT JOIN documents dt ON dt.id = dl.to_doc_id
               WHERE dl.resolved = 1
                 AND (dt.id IS NULL OR dt.active = 1)"""
        ).fetchall()
        conn2.close()

        for e in wiki_edges:
            edge_list.append({"from": e[0], "from_title": e[1], "to": e[2], "to_title": e[3], "type": "wiki"})
            existing_pairs.add((e[0], e[2]))
            existing_pairs.add((e[2], e[0]))

        # Tag-based edges
        for tag, tids in _group_docs_by_tag(doc_tags).items():
            if len(tids) < 2 or len(tids) > 50:
                continue
            for i in range(len(tids)):
                for j in range(i + 1, len(tids)):
                    a, b = tids[i], tids[j]
                    pair = (a, b)
                    if pair not in existing_pairs:
                        edge_list.append({"from": a, "to": b, "type": "tag", "tag": tag})
                        existing_pairs.add(pair)
                        existing_pairs.add((b, a))

    collection_edge_count = 0
    if is_collection_mode:
        col_node_id_base = 10_000_000
        for ci, (col_name, doc_ids) in enumerate(sorted(collection_docs.items())):
            if len(doc_ids) < 2:
                continue
            virtual_id = col_node_id_base + ci
            node_list.append({
                "id": virtual_id,
                "title": f"📁 {col_name}",
                "collection": col_name,
                "path": "",
                "virtual": True,
                "doc_count": len(doc_ids),
            })
            for did in doc_ids:
                pair = (did, virtual_id)
                if pair not in existing_pairs:
                    edge_list.append({"from": did, "to": virtual_id, "type": "collection", "collection": col_name})
                    existing_pairs.add(pair)
                    existing_pairs.add((virtual_id, did))
                    collection_edge_count += 1

    wiki_edge_count = sum(1 for e in edge_list if e["type"] == "wiki")
    tag_edge_count = sum(1 for e in edge_list if e["type"] == "tag")

    # ── Hyper-Extract entities & relationships (mode: hyper / all) ──
    hyper_entity_count = 0
    hyper_rel_count = 0
    if mode in ("hyper", "all"):
        try:
            from hyper_extract import HyperDB
            hyper_db = HyperDB(db_path)
            hyper_entities = hyper_db.conn.execute(
                "SELECT id, name, type, description FROM hyper_entities ORDER BY id"
            ).fetchall()
            hyper_rels = hyper_db.conn.execute(
                "SELECT source_id, target_id, rel_type, weight FROM hyper_relationships"
            ).fetchall()
            hyper_doc_links = hyper_db.conn.execute(
                "SELECT doc_id, entity_id FROM hyper_doc_entities"
            ).fetchall()
            hyper_db.conn.close()

            entity_id_base = 20_000_000
            added_entity_ids = set()
            for he in hyper_entities:
                eid, name, etype, desc = he
                node_id = entity_id_base + eid
                node_list.append({
                    "id": node_id,
                    "title": name,
                    "collection": f"hyper:{etype}",
                    "path": "",
                    "virtual": False,
                    "hyper_entity": True,
                    "entity_type": etype,
                    "entity_id": eid,
                })
                added_entity_ids.add(node_id)
                hyper_entity_count += 1

            for hr in hyper_rels:
                source_id, target_id, rel_type, weight = hr
                from_node = entity_id_base + source_id
                to_node = entity_id_base + target_id
                if from_node in added_entity_ids and to_node in added_entity_ids:
                    pair = (from_node, to_node)
                    if pair not in existing_pairs:
                        edge_list.append({
                            "from": from_node,
                            "to": to_node,
                            "type": "hyper",
                            "label": rel_type,
                            "weight": weight or 1.0,
                        })
                        existing_pairs.add(pair)
                        existing_pairs.add((to_node, from_node))
                        hyper_rel_count += 1

            for hdl in hyper_doc_links:
                doc_id, entity_id = hdl
                from_node = entity_id_base + entity_id
                to_node = doc_id
                if from_node in added_entity_ids:
                    pair = (from_node, to_node)
                    if pair not in existing_pairs:
                        edge_list.append({
                            "from": from_node,
                            "to": to_node,
                            "type": "hyper_doc",
                        })
                        existing_pairs.add(pair)
                        existing_pairs.add((to_node, from_node))
        except Exception:
            pass

    return jsonify({
        "nodes": node_list,
        "edges": edge_list,
        "node_count": len([n for n in node_list if not n.get("virtual")]),
        "edge_count": len(edge_list),
        "wiki_edge_count": wiki_edge_count,
        "tag_edge_count": tag_edge_count,
        "collection_edge_count": collection_edge_count,
        "hyper_entity_count": hyper_entity_count,
        "hyper_rel_count": hyper_rel_count,
        "mode": mode,
        "detail": "high",
        "aggregated": False,
    })


@app.route("/api/document/<int:doc_id>/backlinks", methods=["GET"])
def api_backlinks(doc_id):
    """Return documents that link to this document (incoming references)."""
    db_path = _find_db_path()
    if not db_path or not os.path.exists(db_path):
        return jsonify({"error": "Cannot find index database"}), 500

    conn = sqlite3.connect(db_path)
    _ensure_links_table(conn)

    incoming = conn.execute(
        """SELECT dl.from_doc_id, dl.from_title, dl.to_title, d.title, d.path
           FROM doc_links dl
           LEFT JOIN documents d ON d.id = dl.from_doc_id
           WHERE dl.to_doc_id=? AND dl.resolved=1
           ORDER BY dl.id""", (doc_id,)
    ).fetchall()

    outgoing = conn.execute(
        """SELECT dl.to_doc_id, dl.to_title, dl.resolved, d.title, d.path
           FROM doc_links dl
           LEFT JOIN documents d ON d.id = dl.to_doc_id
           WHERE dl.from_doc_id=?
           ORDER BY dl.id""", (doc_id,)
    ).fetchall()

    conn.close()
    return jsonify({
        "doc_id": doc_id,
        "incoming": [{"from_id": r[0], "from_title": r[1] or r[3] or "", "alias": r[2], "path": r[4] or ""} for r in incoming],
        "outgoing": [{"to_id": r[0], "to_title": r[1], "resolved": bool(r[2]), "doc_title": r[3] or "", "path": r[4] or ""} for r in outgoing],
    })


# ============================================================
# Tags API
# ============================================================

@app.route("/api/scan-tags", methods=["POST"])
def api_scan_tags():
    """Scan documents for YAML frontmatter tags and store them."""
    data = request.get_json() or {}
    doc_id = data.get("doc_id")

    db_path = _find_db_path()
    if not db_path or not os.path.exists(db_path):
        return jsonify({"error": "Cannot find index database"}), 500

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    _ensure_tags_column(conn)

    if doc_id:
        docs = conn.execute("SELECT id FROM documents WHERE id=? AND active=1", (doc_id,)).fetchall()
    else:
        docs = conn.execute("SELECT id FROM documents WHERE active=1 ORDER BY id").fetchall()

    total_tagged = 0
    for (did,) in docs:
        tags = _extract_and_store_tags(did, conn)
        if tags:
            total_tagged += 1
    conn.commit()
    conn.close()
    return jsonify({"success": True, "scanned": len(docs), "tagged": total_tagged})


@app.route("/api/tags", methods=["GET"])
def api_tags():
    """Return all tags with document counts."""
    db_path = _find_db_path()
    if not db_path or not os.path.exists(db_path):
        return jsonify({"error": "Cannot find index database"}), 500

    conn = sqlite3.connect(db_path)
    _ensure_tags_column(conn)
    rows = conn.execute(
        "SELECT tags FROM documents WHERE active=1 AND tags IS NOT NULL AND tags != ''"
    ).fetchall()
    conn.close()

    tag_counts = {}
    for (tags_json,) in rows:
        try:
            for tag in json.loads(tags_json):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        except Exception:
            pass

    sorted_tags = sorted(tag_counts.items(), key=lambda x: -x[1])
    return jsonify({"tags": [{"name": t, "count": c} for t, c in sorted_tags], "total": len(sorted_tags)})


@app.route("/api/documents-by-tag", methods=["GET"])
def api_documents_by_tag():
    """Return documents matching a specific tag."""
    tag = request.args.get("tag", "")
    if not tag:
        return jsonify({"error": "Missing 'tag' parameter"}), 400

    db_path = _find_db_path()
    if not db_path or not os.path.exists(db_path):
        return jsonify({"error": "Cannot find index database"}), 500

    conn = sqlite3.connect(db_path)
    _ensure_tags_column(conn)
    rows = conn.execute(
        "SELECT id, title, path, collection, tags FROM documents WHERE active=1 AND tags IS NOT NULL AND tags != ''"
    ).fetchall()
    conn.close()

    results = []
    for doc_id, title, path, collection, tags_json in rows:
        try:
            doc_tags = json.loads(tags_json)
            if tag in doc_tags:
                results.append({"id": doc_id, "title": title or "", "path": path or "", "collection": collection or "", "tags": doc_tags})
        except Exception:
            pass

    return jsonify({"tag": tag, "documents": results, "count": len(results)})


@app.route("/api/collection/docs", methods=["GET"])
def api_collection_docs():
    """List documents within a collection — returns doc IDs, titles, paths.

    Query params: collection (required), limit (default 50), page (default 1)
    """
    import sqlite3

    collection = request.args.get("collection", "")
    if not collection:
        return jsonify({"error": "Missing 'collection' parameter"}), 400

    limit = int(request.args.get("limit", 50))
    page = int(request.args.get("page", 1))
    offset = (page - 1) * limit

    # Find DB path via stats output
    db_path = None
    r = _run_qmd_raw("stats")
    for line in (r.get("output", "") or "").split("\n"):
        if "Database:" in line:
            db_path = line.split(":", 1)[1].strip().split("(")[0].strip()

    if not db_path or not os.path.exists(db_path):
        return jsonify({"error": "Cannot find index database"}), 500

    try:
        conn = sqlite3.connect(db_path)
        _ensure_tags_column(conn)
        cur = conn.cursor()
        # Get total count for pagination
        total = cur.execute(
            "SELECT COUNT(*) FROM documents WHERE collection=? AND active=1", (collection,)
        ).fetchone()[0]
        # Get documents with metadata
        docs = cur.execute(
            """SELECT id, title, path, created_at, modified_at, active, tags
               FROM documents
               WHERE collection=? AND active=1
               ORDER BY id DESC LIMIT ? OFFSET ?""", (collection, limit, offset)
        ).fetchall()

        result_docs = []
        for row in docs:
            tags_list = []
            if row[6]:
                try: tags_list = json.loads(row[6])
                except Exception: pass
            result_docs.append({
                "id": row[0],
                "title": row[1] or "",
                "path": row[2] or "",
                "created_at": row[3] or "",
                "modified_at": row[4] or "",
                "active": bool(row[5]),
                "tags": tags_list,
            })

        conn.close()
        return jsonify({
            "collection": collection,
            "total": total,
            "page": page,
            "limit": limit,
            "docs": result_docs,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/document/<int:doc_id>/content", methods=["GET"])
def api_get_document_content(doc_id):
    """Get full content of a document by ID.

    Returns the complete body text (not just metadata).
    Content is stored in the 'doc' column of the content table.
    """
    import sqlite3

    db_path = None
    r = _run_qmd_raw("stats")
    for line in (r.get("output", "") or "").split("\n"):
        if "Database:" in line:
            db_path = line.split(":", 1)[1].strip().split("(")[0].strip()

    if not db_path or not os.path.exists(db_path):
        return jsonify({"error": "Cannot find index database"}), 500

    try:
        conn = sqlite3.connect(db_path)
        _ensure_tags_column(conn)
        cur = conn.cursor()
        # Get document metadata
        doc_row = cur.execute(
            "SELECT id, title, path, collection, created_at, modified_at, tags FROM documents WHERE id=?", (doc_id,)
        ).fetchone()

        if not doc_row:
            conn.close()
            return jsonify({"error": f"Document #{doc_id} not found"}), 404

        # Get content from content table — note: content is in 'doc' column, not 'content'
        full_content = ""
        file_hash = None
        try:
            cr = cur.execute("SELECT hash FROM documents WHERE id=?", (doc_id,)).fetchone()
            if cr:
                file_hash = cr[0]
                cr2 = cur.execute("SELECT doc FROM content WHERE hash=?", (file_hash,)).fetchone()
                if cr2:
                    full_content = str(cr2[0])
        except Exception:
            pass

        conn.close()
        return jsonify({
            "id": doc_row[0],
            "title": doc_row[1] or "",
            "path": doc_row[2] or "",
            "collection": doc_row[3] or "",
            "created_at": doc_row[4] or "",
            "modified_at": doc_row[5] or "",
            "tags": json.loads(doc_row[6]) if doc_row[6] else [],
            "content": full_content,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/document/<int:doc_id>/content", methods=["PUT"])
def api_update_document_content(doc_id):
    """Update document content (MD editing). Body: { content: "..." }"""
    data = request.get_json() or {}
    new_content = data.get("content", "")
    if not new_content:
        return jsonify({"error": "Missing 'content' in request body"}), 400

    db_path = None
    r = _run_qmd_raw("stats")
    for line in (r.get("output", "") or "").split("\n"):
        if "Database:" in line:
            db_path = line.split(":", 1)[1].strip().split("(")[0].strip()

    if not db_path or not os.path.exists(db_path):
        return jsonify({"error": "Cannot find index database"}), 500

    import sqlite3
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        doc = cur.execute("SELECT hash FROM documents WHERE id=?", (doc_id,)).fetchone()
        if not doc:
            conn.close()
            return jsonify({"error": f"Document #{doc_id} not found"}), 404
        old_hash = doc[0]

        # Update content in the content table
        cur.execute("UPDATE content SET doc=? WHERE hash=?", (new_content, old_hash))
        if cur.rowcount == 0:
            cur.execute("INSERT INTO content (hash, doc, created_at) VALUES (?, ?, ?)",
                        (old_hash, new_content, datetime.now().isoformat() + "Z"))

        now = datetime.now().isoformat() + "Z"
        cur.execute("UPDATE documents SET modified_at=? WHERE id=?", (now, doc_id))
        conn.commit()
        conn.close()
        return jsonify({"success": True, "id": doc_id, "modified_at": now})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/update", methods=["POST"])
def api_update():
    """Re-scan and update documents. Maps to: qmd update"""
    data = request.get_json() or {}
    collection = data.get("collection")

    args = ["update"]
    if collection:
        args += ["-c", collection]

    result = _run_qmd_raw(*args)
    return jsonify(result)


@app.route("/api/optimize", methods=["POST"])
def api_optimize():
    """Optimize index. Maps to: qmd optimize"""
    data = request.get_json() or {}
    full = data.get("full", False)

    args = ["optimize"]
    if full:
        args.append("--full")

    result = _run_qmd_raw(*args)
    return jsonify(result)


@app.route("/api/import", methods=["POST"])
def api_import():
    """Batch import xl.meta files. Maps to: qmd import"""
    data = request.get_json() or {}
    base_dir = data.get("base_dir", "")

    if not base_dir:
        return jsonify({"error": "Missing 'base_dir'"}), 400

    args = ["import", base_dir]
    prefix = data.get("prefix")
    if prefix:
        args += ["-p", prefix]

    result = _run_qmd_raw(*args)
    return jsonify(result)


# --- Embedding (embed from tobi/qmd) ---

@app.route("/api/embed", methods=["POST"])
def api_embed():
    """Generate embeddings for indexed documents. Maps to: qmd embed."""
    data = request.get_json() or {}
    collection = data.get("collection")  # None = all collections

    if collection:
        result = _run_qmd_raw("embed", "-c", collection)
    else:
        result = _run_qmd_raw("embed")

    return jsonify(result)


# --- Query Expansion (expand from tobi/qmd) ---

@app.route("/api/expand", methods=["POST"])
def api_expand():
    """Expand a query using LLM-based expansion with the dedicated query-expansion model."""
    data = request.get_json() or {}
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Missing 'query' parameter"}), 400

    intent = data.get("intent", "")
    max_expansions = int(data.get("max", 3))

    expansions = []

    # Always include lex expansion
    expansions.append({"type": "lex", "query": f'"{query}"'})

    # Try LLM-powered expansion via the query-expansion Docker service
    llm_expansions = _call_query_expansion_llm(query, intent, max_expansions - 1)

    if llm_expansions:
        expansions.extend(llm_expansions)
    else:
        # Fallback: string-based expansion when LLM is unavailable
        if intent:
            expansions.append({"type": "vec", "query": intent})
        else:
            expansions.append({"type": "vec", "query": query})
        expansions.append({"type": "hyde", "query": f"Document about {query}"})

    # Run all expanded queries and collect results
    all_results = []
    for exp in expansions[:max_expansions]:
        try:
            r = _run_qmd_raw("search", exp["query"], "-n", "5")
            if "error" not in r:
                all_results.append({**exp, "output": r.get("output", "")})
        except Exception:
            pass

    return jsonify({
        "original_query": query,
        "intent": intent,
        "expansions": expansions[:max_expansions],
        "results": all_results,
    })


def _call_query_expansion_llm(query: str, intent: str, max_count: int) -> list:
    """Call the query-expansion Docker service to generate alternative queries.

    The model is a completion-only GGUF (QMD Query Expansion 1.7B), so we use
    the /v1/completions endpoint. Uses a structured prompt that elicits
    lex/vec/hyde style expansions. Returns list of {'type','query'} dicts,
    or empty list on failure.
    """
    if max_count < 1:
        return []
    base = SETTINGS.get("query_expansion_url", "").rstrip("/")
    url = base + "/completions"
    model = SETTINGS.get("query_expansion_model", "hf_tobil_qmd-query-expansion-1.7B-q4_k_m.gguf")
    if not url.startswith("http"):
        return []

    prompt = f"Query: {query}\nAlternative:"

    try:
        r = requests.post(url, json={
            "model": model,
            "prompt": prompt,
            "max_tokens": 64,
            "temperature": 0.7,
        }, timeout=30)
        if r.status_code == 200:
            body = r.json()
            text = (body.get("choices") or [{}])[0].get("text", "").strip()
            if text:
                import re as _re
                expansions = []
                for line in text.split("\n"):
                    line = line.strip().lstrip("0123456789.)-").strip()
                    if not line:
                        continue
                    # Extract alternative query after "Alternative:" or similar markers
                    alt = _re.sub(r'^(Query|Alternative)\s*:\s*', '', line, flags=_re.I).strip()
                    if alt and alt != query and alt != " ":
                        expansions.append({"type": "vec", "query": alt})
                        if len(expansions) >= max_count:
                            break
                if expansions:
                    return expansions
    except Exception:
        pass
    return []


# ============================================================
# File Upload + Embed
# ============================================================

@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Upload files and add to QMD index. Accepts multipart form data.
    
    Form fields:
      - files: one or more file fields (the actual file content)
      - collection: target collection name (optional)
      - embed: 'true' to trigger vector embedding after add (optional)
      - recursive: 'true' to scan subdirs if a zip is uploaded (optional)
    Returns: JSON with results
    """
    import time, shutil

    collection = request.form.get("collection", "").strip() or None
    do_embed = request.form.get("embed", "false").lower() == "true"
    uploaded = request.files.getlist("files")

    if not uploaded:
        return jsonify({"error": "No files uploaded"}), 400

    saved_paths = []
    results = []
    errors = []

    for f in uploaded:
        if not f.filename:
            continue
        # Sanitize filename
        safe_name = os.path.basename(f.filename)
        save_path = UPLOAD_DIR / safe_name
        # Avoid overwrite: append number if exists
        counter = 1
        while save_path.exists():
            stem = Path(safe_name).stem
            ext = Path(safe_name).suffix
            save_path = UPLOAD_DIR / f"{stem}_{counter}{ext}"
            counter += 1

        try:
            f.save(str(save_path))
            saved_paths.append(save_path)
        except Exception as e:
            errors.append({"file": safe_name, "error": f"Save failed: {e}"})
            continue

    # Run qmd.py add for each saved file
    for sp in saved_paths:
        args = ["add", str(sp)]
        if collection:
            args += ["-c", collection]
        result = _run_qmd_raw(*args, timeout=120)
        results.append({"file": sp.name, "path": str(sp), **result})
        if "error" in result:
            errors.append({"file": sp.name, "error": result["error"]})

    # Optionally run embedding in background thread
    embed_job_id = None
    embed_total = 0
    if do_embed and results and not errors:
        db_path = _find_db_path()
        if db_path:
            job_id = str(_uuid.uuid4())[:8]
            total_chunks = 0
            # Estimate total chunks
            try:
                est_conn = sqlite3.connect(db_path)
                for res in results:
                    fpath = res.get("path", "")
                    if not fpath:
                        continue
                    hr = est_conn.execute(
                        "SELECT hash FROM documents WHERE path LIKE ? ORDER BY id DESC LIMIT 1",
                        (f"%{os.path.basename(fpath)}",)
                    ).fetchone()
                    if hr:
                        dr = est_conn.execute("SELECT doc FROM content WHERE hash=?", (hr[0],)).fetchone()
                        if dr:
                            total_chunks += max(1, len(dr[0]) // 512)
                est_conn.close()
            except Exception:
                total_chunks = len(results) * 5

            state = {
                "job_id": job_id,
                "type": "embed",
                "status": "pending",
                "total": total_chunks,
                "current": 0,
                "done": 0,
                "errors": [],
                "file_errors": [],
            }
            _jobs[job_id] = state
            embed_job_id = job_id
            embed_total = total_chunks

            def _embed_worker():
                s = _jobs.get(job_id)
                if s: s["status"] = "running"
                embed_done = 0
                embed_file_errors = []
                try:
                    embed_conn = sqlite3.connect(db_path)
                    embed_conn.execute("PRAGMA journal_mode=WAL")
                    embed_conn.execute("PRAGMA synchronous=OFF")
                    embed_conn.enable_load_extension(True)
                    vec0_dll = os.path.join(str(BASE_DIR), "node_modules", "sqlite-vec-windows-x64", "vec0.dll")
                    if os.path.exists(vec0_dll):
                        embed_conn.load_extension(vec0_dll)

                    embed_url = SETTINGS.get("embedding_url", "http://127.0.0.1:8025/v1/embeddings")
                    embed_model = SETTINGS.get("embedding_model", "embeddinggemma-300M-Q8_0")
                    now = datetime.now(timezone.utc).isoformat()

                    for res in results:
                        fpath = res.get("path", "")
                        if not fpath:
                            continue
                        hash_row = embed_conn.execute(
                            "SELECT hash FROM documents WHERE path LIKE ? ORDER BY id DESC LIMIT 1",
                            (f"%{os.path.basename(fpath)}",)
                        ).fetchone()
                        if not hash_row:
                            embed_file_errors.append({"file": fpath, "error": "no matching doc hash"})
                            continue
                        doc_hash = hash_row[0]
                        doc_row = embed_conn.execute(
                            "SELECT doc FROM content WHERE hash=?", (doc_hash,)
                        ).fetchone()
                        if not doc_row:
                            embed_file_errors.append({"file": fpath, "error": "no content for hash"})
                            continue
                        doc_text = doc_row[0]

                        chunks = _chunk_text(doc_text, 512)
                        batch_texts = [c[2] for c in chunks]
                        try:
                            emb_resp = requests.post(embed_url, json={
                                "input": [t[:8000] for t in batch_texts],
                                "model": embed_model
                            }, timeout=300)
                            emb_resp.raise_for_status()
                            emb_data = emb_resp.json()
                            emb_list = sorted(emb_data["data"], key=lambda x: x["index"])
                        except Exception as e:
                            embed_file_errors.append({"file": fpath, "error": f"embed API: {e}"})
                            continue

                        for seq, ((pos, end, text), emb_item) in enumerate(zip(chunks, emb_list)):
                            hs = f"{doc_hash}_{seq}"
                            embedding = emb_item["embedding"]
                            embed_conn.execute(
                                "INSERT OR IGNORE INTO vectors_vec (hash_seq, embedding) VALUES (?, ?)",
                                (hs, json.dumps(embedding))
                            )
                            embed_conn.execute(
                                "INSERT OR IGNORE INTO content_vectors (hash, seq, pos, model, embedded_at) VALUES (?, ?, ?, ?, ?)",
                                (doc_hash, seq, pos, embed_model, now)
                            )
                            embed_done += 1
                            s = _jobs.get(job_id)
                            if s: s["current"] = embed_done

                    embed_conn.commit()
                    embed_conn.close()
                except Exception as e:
                    s = _jobs.get(job_id)
                    if s: s["file_errors"] = embed_file_errors + [{"error": str(e)}]

                s = _jobs.get(job_id)
                if s:
                    s["status"] = "done"
                    s["done"] = embed_done
                    s["file_errors"] = embed_file_errors

            _threading.Thread(target=_embed_worker, daemon=True).start()

    response = {
        "success": len(errors) == 0,
        "total": len(saved_paths),
        "added": len(saved_paths) - len(errors),
        "errors": errors,
        "results": results,
    }
    if embed_job_id:
        response["embed_job_id"] = embed_job_id
        response["embed_pending"] = True
        response["embed_total"] = embed_total
    return jsonify(response)


# ============================================================
# Settings persistence
# ============================================================

SETTINGS_PATH = Path(__file__).resolve().parent / "settings.json"

DEFAULT_SETTINGS = {
    "embedding_url": "http://127.0.0.1:8025/v1/embeddings",
    "embedding_model": "embeddinggemma-300M-Q8_0",
    "embedding_dim": 768,
    "reranker_url": "http://127.0.0.1:8024/v1/rerank",
    "reranker_model": "Qwen.Qwen3-Reranker-0.6B.Q8_0.gguf",
    "llm_url": "http://localhost:1235/v1",
    "llm_model": "qwen/qwen3.8-27b",
    "llm_key": "",
    "llm_ctx": 32768,
    "query_expansion_url": "http://127.0.0.1:8026/v1",
    "query_expansion_model": "hf_tobil_qmd-query-expansion-1.7B-q4_k_m.gguf",
}

def load_settings():
    if SETTINGS_PATH.exists():
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                return {**DEFAULT_SETTINGS, **json.load(f)}
        except Exception:
            pass
    return dict(DEFAULT_SETTINGS)

def save_settings(s):
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2, ensure_ascii=False)

SETTINGS = load_settings()

@app.route("/api/settings", methods=["GET"])
def api_get_settings():
    safe = {k: v for k, v in SETTINGS.items() if k != "llm_key"}
    safe["llm_key"] = "********" if SETTINGS.get("llm_key") else ""
    return jsonify(safe)

@app.route("/api/settings", methods=["PUT"])
def api_put_settings():
    data = request.get_json() or {}
    for k in DEFAULT_SETTINGS:
        if k in data:
            SETTINGS[k] = data[k]
    save_settings(SETTINGS)
    return jsonify({"success": True})

@app.route("/api/settings/test", methods=["POST"])
def api_test_settings():
    import urllib.request, urllib.error, json as _json
    results = []
    all_ok = True

    def test_endpoint(name, url, method="POST", payload=None, timeout=10):
        nonlocal all_ok
        try:
            if method == "POST":
                req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            else:
                req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")[:200]
                results.append({"name": name, "ok": True, "message": f"HTTP {resp.status} — {body[:80]}"})
        except urllib.error.HTTPError as e:
            # Some endpoints return 4xx for bad payloads but are reachable
            results.append({"name": name, "ok": True, "message": f"HTTP {e.code} (endpoint reachable)"})
        except Exception as e:
            results.append({"name": name, "ok": False, "message": str(e)[:100]})
            all_ok = False

    # Test embedding
    emb_payload = _json.dumps({"input": "test", "model": SETTINGS.get("embedding_model", "default")}).encode()
    test_endpoint("Embedding", SETTINGS.get("embedding_url", ""), payload=emb_payload)

    # Test reranker
    rerank_payload = _json.dumps({"query": "test", "documents": ["hello world"]}).encode()
    test_endpoint("Reranker", SETTINGS.get("reranker_url", ""), payload=rerank_payload)

    # Test LLM
    llm_payload = _json.dumps({"model": SETTINGS.get("llm_model", ""), "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}).encode()
    test_endpoint("LLM", SETTINGS.get("llm_url", "") + "/chat/completions", payload=llm_payload, timeout=15)

    # Test query-expansion
    qe_url = SETTINGS.get("query_expansion_url", "").rstrip("/") + "/chat/completions"
    qe_payload = _json.dumps({"model": SETTINGS.get("query_expansion_model", ""), "messages": [{"role": "user", "content": "test"}], "max_tokens": 5}).encode()
    test_endpoint("Query-Expansion", qe_url, payload=qe_payload, timeout=15)

    return jsonify({"results": results, "all_ok": all_ok})


# --- Version / Upgrade Check ---

@app.route("/api/version", methods=["GET"])
def api_version():
    """Get current QMD Index version and check for upgrades."""
    import pkg_resources

    # Read local version from qmd.py header
    local_version = "3.0"
    try:
        with open(QMD_PY, "r") as f:
            content = f.read()
            for line in content.split("\n"):
                line_stripped = line.strip().strip("=").strip()
                if line_stripped.startswith("QMD Index v"):
                    ver_str = line_stripped.replace("QMD Index v", "").strip()
                    # Extract just the version number (e.g., "3.0")
                    import re
                    match = re.search(r'v?(\d+\.\d+)', ver_str)
                    if match:
                        local_version = match.group(1)
    except Exception:
        pass

    # Check npm registry for latest @tobilu/qmd version (tobi's original)
    remote_version = None
    try:
        r = subprocess.run(
            ["npm", "view", "@tobilu/qmd", "version"],
            capture_output=True, text=True, timeout=15
        )
        if r.returncode == 0 and r.stdout.strip():
            remote_version = r.stdout.strip()
    except Exception:
        pass

    return jsonify({
        "qmd_index": {"local": local_version},
        "tobi_qmd": {"latest_remote": remote_version or "unknown"},
        "python": sys.version.split()[0],
        "platform": sys.platform,
    })


@app.route("/api/upgrade", methods=["POST"])
def api_upgrade():
    """Attempt to upgrade QMD Index (pull latest from git)."""
    import subprocess as sp

    result = sp.run(
        ["git", "-C", str(BASE_DIR), "pull"],
        capture_output=True, text=True, timeout=60
    )

    if result.returncode == 0:
        return jsonify({
            "success": True,
            "message": f"Upgrade completed.\n{result.stdout.strip()[:500]}",
            "output": result.stdout.strip(),
        })
    else:
        return jsonify({
            "success": False,
            "error": f"Git pull failed: {result.stderr.strip()[:500]}",
            "message": "This may not be a git repository or network is unavailable.",
        })


# ============================================================
# Resources API — Read workspace files
# ============================================================

@app.route("/api/resources/<path:filename>", methods=["GET"])
def get_resource(filename):
    """Read a file from the QMD workspace.
    
    Returns the raw content of a markdown/text file in the workspace.
    """
    # Security: prevent path traversal
    if ".." in filename or filename.startswith("/"):
        return jsonify({"error": "Invalid filename"}), 400
    
    # Look for the file in common locations
    search_paths = [
        BASE_DIR / filename,
        BASE_DIR / "docs" / filename,
        BASE_DIR / "workspace" / filename,
        Path("/app/workspace") / filename,
        Path("/app/docs") / filename,
    ]
    
    for path in search_paths:
        if path.exists() and path.is_file():
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
                return jsonify({
                    "filename": filename,
                    "content": content,
                    "size": len(content),
                })
            except Exception as e:
                return jsonify({"error": f"Failed to read file: {str(e)}"}), 500
    
    return jsonify({"error": f"File not found: {filename}"}), 404


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="QMD Index WebUI Server")
    parser.add_argument("--port", type=int, default=PORT, help="Port to listen on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()
    print(f"Starting QMD Index WebUI on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)

