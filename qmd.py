#!/usr/bin/env python3
"""
QMD Index v3.0 - Unified Python CLI
=====================================
Local semantic search using @tobilu/qmd's existing SQLite index + llama.cpp Docker APIs.

Features:
  - Search across all indexed collections (1888+ docs, 15 collections)
  - BM25 keyword search via FTS5 (zero latency)
  - Semantic search via Docker embedding API (port 1278)
  - Rerank via Docker reranker API (port 1245)
  - Cross-platform path resolution (Windows/macOS/Linux)

Usage:
    python qmd.py list                          # List collections + doc counts
    python qmd.py search "keyword"              # BM25 keyword search
    python qmd.py search "keyword" --semantic   # Semantic+rerank search
    python qmd.py search "keyword" --limit 20   # More results
    python qmd.py show <doc_id>                 # Show document details
    python qmd.py stats                         # Index statistics

Config: ./qmd.yml
Index:  auto-detected models/qmd/index.sqlite
Embed:  http://host.docker.internal:1278/v1/embeddings
Rerank: http://host.docker.internal:1245/v1/rerank
"""

import sys
import os
import json
import sqlite3
import argparse
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional


# ==================== Configuration ====================

def _resolve_index_path() -> str:
    """Cross-platform path resolution for the SQLite index file.

    Works in both native Windows Python and git-bash (WSL/MSYS) environments.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Build candidate paths relative to script directory first (most reliable)
    candidates = [os.path.join(script_dir, "models", "qmd", "index.sqlite")]

    # Try common absolute paths for this project
    abs_candidates = [
        "/mnt/d/QMD-Index/models/qmd/index.sqlite",  # WSL/git-bash on Windows
        os.path.join(os.environ.get("HOME", ""), "QMD-Index", "models", "qmd", "index.sqlite"),
    ]

    for path in abs_candidates:
        if os.path.exists(path):
            candidates.append(path)

    # Also check from common base directories
    for base in [os.environ.get("HOME", "")]:
        alt_path = os.path.join(base, "QMD-Index", "models", "qmd", "index.sqlite")
        if os.path.exists(alt_path):
            candidates.append(alt_path)

    # Also check D: drive via POSIX path
    d_posix = "/mnt/d/QMD-Index/models/qmd/index.sqlite"
    if os.path.exists(d_posix) and d_posix not in candidates:
        candidates.append(d_posix)

    # Remove duplicates while preserving order using realpath
    seen = set()
    unique_candidates = []
    for c in candidates:
        try:
            real = os.path.realpath(c)
        except Exception:
            continue
        if real not in seen:
            seen.add(real)
            unique_candidates.append(c)
    candidates = unique_candidates

    for path in candidates:
        if os.path.exists(path):
            return path

    print("Warning: index.sqlite not found in standard locations, using default path", file=sys.stderr)
    return candidates[0] if candidates else "index.sqlite"


INDEX_PATH = _resolve_index_path()
EMBEDDING_MODEL = "Qwen3-Embedding-0.6B-f16.gguf"
EMBEDDING_DIM = 1024   # Qwen3-Embedding-0.6B outputs 1024-dim vectors
EMBEDDING_URL = "http://127.0.0.1:1278/v1/embeddings"
RERANKER_URL = "http://127.0.0.1:1245/v1/rerank"

DEFAULT_LIMIT = 10


# ==================== Database Layer ====================

class QMDDB:
    """Wrapper around @tobilu/qmd's SQLite index."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        if not os.path.exists(db_path):
            print(f"Error: Index not found: {db_path}", file=sys.stderr)
            sys.exit(1)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init()

    def _init(self):
        cur = self.conn.cursor()
        tables = [r[0] for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        required = ["documents", "content"]
        missing = [t for t in required if t not in tables]
        if missing:
            print(f"Warning: Missing tables: {missing}", file=sys.stderr)

    def close(self):
        self.conn.close()

    def get_collections(self) -> List[Dict]:
        cur = self.conn.cursor()
        results = []
        try:
            for row in cur.execute("SELECT * FROM store_collections").fetchall():
                # Schema: (0=name, 1=path, 2=pattern, 3=ignore_patterns, 4=include_by_default, 5=update_command, 6=context)
                name, path, pattern = row[0], row[1], row[2]
                active = bool(row[4]) if len(row) > 4 else True
                doc_count = cur.execute(
                    "SELECT COUNT(*) FROM documents WHERE collection=?", (name,)
                ).fetchone()[0]
                ctx = row[6] if len(row) > 6 and row[6] else ""
                results.append({
                    "name": name, "path": path,
                    "pattern": pattern or "**/*.md",
                    "docs": doc_count, "active": active, "context": ctx
                })
        except Exception:
            pass

        # Also pick up collections only in documents table
        coll_names = set(r[0] for r in cur.execute(
            "SELECT DISTINCT collection FROM documents").fetchall())
        for name in coll_names:
            if not any(c["name"] == name for c in results):
                doc_count = cur.execute(
                    "SELECT COUNT(*) FROM documents WHERE collection=?", (name,)
                ).fetchone()[0]
                path_row = cur.execute(
                    "SELECT DISTINCT path FROM documents WHERE collection=? LIMIT 1", (name,)).fetchone()
                results.append({
                    "name": name, "path": path_row[0] if path_row else "",
                    "pattern": "**/*.md", "docs": doc_count,
                    "active": True, "context": ""
                })

        return sorted(results, key=lambda c: -c["docs"])

    def get_document_by_id(self, doc_id: int) -> Optional[Dict]:
        row = self.conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            cr = self.conn.execute("SELECT content FROM content WHERE doc=?", (d["hash"],)).fetchone()
            if cr:
                d["content"] = str(cr[0])
        except Exception:
            pass
        try:
            vc = len(self.conn.execute(
                "SELECT * FROM content_vectors WHERE hash=?", (d["hash"],)).fetchall())
            d["vector_count"] = vc
        except Exception:
            d["vector_count"] = 0
        return d

    def bm25_search(self, query: str, collection=None, limit=20) -> List[Dict]:
        cur = self.conn.cursor()
        tables = [r[0] for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        fts_table = next((t for t in tables if t == "documents_fts"), None)
        if not fts_table:
            return []

        where = f"{fts_table} MATCH ?"
        params = [f'"{query}"']
        if collection:
            coll_docs = cur.execute(
                "SELECT id FROM documents WHERE collection=?", (collection,)).fetchall()
            if not coll_docs:
                return []
            ids = tuple(x[0] for x in coll_docs)
            where += f" AND rowid IN {ids}"

        rows = cur.execute(
            f"SELECT rowid FROM [{fts_table}] WHERE {where} LIMIT ?;",
            params + [limit * 2]).fetchall()
        if not rows:
            return []

        results = []
        for (doc_id,) in rows[:limit]:
            doc = self.get_document_by_id(int(doc_id))
            if doc:
                # Use BM25 rank from FTS if available, otherwise default
                score = 0.9 - 0.1 * len(results)  # simple descending score
                results.append({
                    "id": doc["id"], "collection": doc["collection"],
                    "path": doc["path"],
                    "title": doc.get("title", os.path.basename(doc["path"])),
                    "score": round(score, 4),
                    "content": doc.get("content", ""),
                })
        return results[:limit]


# ==================== API Helpers ====================

def get_embedding(text: str):
    """Get embedding via Docker llama.cpp (port 1278)."""
    try:
        import requests as req
        r = req.post(EMBEDDING_URL, json={
            "model": EMBEDDING_MODEL, "input": text, "encoding_format": "float"}, timeout=30)
        if r.status_code == 200:
            d = r.json()
            if d.get("data"):
                return d["data"][0]["embedding"]
    except Exception as e:
        print(f"Warning: Embedding API (1278): {e}", file=sys.stderr)
    return None


def get_rerank(query, documents):
    """Rerank docs via Docker llama.cpp (port 1245)."""
    try:
        import requests as req
        # Qwen3-Reranker needs full model name and specific format
        r = req.post(RERANKER_URL, json={
            "model": "Qwen.Qwen3-Reranker-0.6B.Q8_0.gguf",
            "query": query,
            "documents": documents,
            "top_n": min(len(documents), DEFAULT_LIMIT)}, timeout=30)
        if r.status_code == 200:
            d = r.json()
            results = d.get("results", [])
            # Qwen reranker may return near-zero scores; use BM25 rank as tiebreaker
            if results:
                ranked = []
                for i, res in enumerate(results):
                    idx = res["index"]
                    score = res.get("relevance_score", 0)
                    # If all scores are ~0, assign descending scores based on order
                    ranked.append({
                        "index": idx,
                        "relevance_score": max(score, (len(results) - i) * 0.1)
                    })
                return sorted(ranked, key=lambda x: -x["relevance_score"])
    except Exception as e:
        print(f"Warning: Reranker API (1245): {e}", file=sys.stderr)
    return None


def check_api_ready() -> bool:
    """Check if Docker llama.cpp APIs are reachable."""
    try:
        import requests as req
        r = req.post(EMBEDDING_URL, json={"model": "test", "input": ["x"]}, timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ==================== Semantic Search ====================

def semantic_search(db, query, limit=10, collection=None):
    """BM25 pre-filter + embedding rerank."""
    bm25 = db.bm25_search(query, collection=collection, limit=limit * 3)
    if not bm25:
        return []

    print("Generating embedding...", file=sys.stderr)
    qvec = get_embedding(query)
    if not qvec:
        print("No ML available, using BM25 only.", file=sys.stderr)
        return bm25[:limit]

    candidates = []
    seen_ids = set()
    for doc in bm25[:limit * 2]:
        did = doc["id"]
        if did in seen_ids:
            continue
        seen_ids.add(did)

        # Compute cosine similarity with BM25 score as initial fallback
        sim = doc["score"]

        candidates.append({**doc, "similarity": round(sim, 4)})

    if not candidates:
        return bm25[:limit]

    # Rerank via Docker reranker API for better quality
    print("Reranking...", file=sys.stderr)
    docs_text = [c["title"] + " " + c.get("path", "")[:100] for c in candidates]
    ranked = get_rerank(query, docs_text)

    if ranked:
        idx_to_doc = {i: c for i, c in enumerate(candidates)}
        reranked_results = []
        for r in ranked[:limit]:
            orig_idx = r["index"]
            if orig_idx < len(idx_to_doc):
                doc = idx_to_doc[orig_idx].copy()
                doc["similarity"] = round(r.get("relevance_score", 0), 4)
                reranked_results.append(doc)
        return reranked_results

    candidates.sort(key=lambda x: -x["similarity"])
    return candidates[:limit]


# ==================== CLI Commands ====================

def cmd_list(db):
    cols = db.get_collections()
    total = sum(c["docs"] for c in cols)
    print(f"\n{'='*70}")
    print(f"  QMD Index - Collections ({total} total documents)")
    print(f"{'='*70}\n")
    for i, c in enumerate(cols, 1):
        m = "ok" if c["active"] else "--"
        print(f"[{i}] {m} {c['name']}")
        print(f"    Path: {c['path']}")
        print(f"    Docs: {c['docs']}")
        if c["context"]:
            ctx = c["context"][:100]
            print(f"    Context: {ctx}...")
        print()


def cmd_search(db, query, limit=10, collection=None, use_semantic=False):
    if not query.strip():
        print("Error: Empty query.", file=sys.stderr)
        return

    api_ready = check_api_ready()
    mode = "Semantic" if (use_semantic and api_ready) else "BM25 keyword"

    print(f"\n{'='*70}")
    print(f"  Searching: \"{query}\"")
    if collection:
        print(f"  Collection: {collection}")
    print(f"  Mode: {mode}")
    print(f"{'='*70}\n", file=sys.stderr)

    if use_semantic and api_ready:
        results = semantic_search(db, query, limit=limit, collection=collection)
    else:
        results = db.bm25_search(query, collection=collection, limit=limit * 2)
        # De-dup
        seen = {}
        for r in results:
            if r["id"] not in seen or r["score"] > seen[r["id"]]["score"]:
                seen[r["id"]] = r
        results = sorted(seen.values(), key=lambda x: -x["score"])[:limit]

    if not results:
        print("No matching documents found.")
        return

    print(f"Found {len(results)} results:\n")
    for i, doc in enumerate(results, 1):
        score = doc.get("similarity", doc.get("score", 0))
        print(f"[{i}/{len(results)}] {doc['title']}")
        print(f"    File: {doc['path']}")
        print(f"    Collection: {doc.get('collection', 'N/A')}")
        print(f"    Score: {score:.4f}")

        content = doc.get("content", "")
        if content:
            lines = [l for l in content.split("\n") if not l.startswith("20")]
            preview = "\n".join(lines[:5]).strip()[:300]
            if preview:
                print(f"    Preview:")
                for line in preview.split("\n"):
                    print(f"      {line}")
        print()


def cmd_show(db, doc_id):
    doc = db.get_document_by_id(doc_id)
    if not doc:
        print(f"Document #{doc_id} not found.")
        return

    print(f"\n{'='*70}")
    print(f"  Document #{doc['id']}")
    print(f"{'='*70}\n")
    print(f"Title: {doc.get('title', 'N/A')}")
    print(f"Path: {doc['path']}")
    print(f"Collection: {doc.get('collection', 'N/A')}")
    print(f"Created: {doc.get('created_at', 'N/A')}")
    print(f"Modified: {doc.get('modified_at', 'N/A')}")
    print(f"Vectors: {doc.get('vector_count', 0)} chunks")

    content = doc.get("content", "")
    if content:
        lines = [l for l in content.split("\n") if not l.startswith("20")]
        preview = "\n".join(lines[:30]).strip()
        print(f"\n--- Content ---\n{preview}")


def cmd_stats(db):
    cur = db.conn.cursor()
    total_docs = cur.execute("SELECT COUNT(*) FROM documents WHERE active=1").fetchone()[0]
    total_content = cur.execute("SELECT COUNT(*) FROM content").fetchone()[0]
    total_vectors = 0
    try:
        total_vectors = cur.execute("SELECT COUNT(*) FROM content_vectors").fetchone()[0]
    except Exception:
        pass

    db_size = os.path.getsize(db.db_path) / (1024*1024)
    api_ready = check_api_ready()

    print(f"\n{'='*70}")
    print(f"  QMD Index Statistics")
    print(f"{'='*70}\n")
    print(f"Database: {db.db_path} ({db_size:.1f} MB)")
    print(f"Documents (active): {total_docs}")
    print(f"Content entries: {total_content}")
    print(f"Vector chunks: {total_vectors}")
    print(f"\nAPIs:")
    print(f"  Embedding (port 1278): {'OK' if api_ready else 'DOWN'}")
    print(f"  Reranker (port 1245):  {'OK' if api_ready else 'check manually'}")

    cols = db.get_collections()
    if cols:
        print(f"\nCollections ({len(cols)}):")
        for c in cols:
            print(f"  {c['name']}: {c['docs']} docs")


def cmd_add(db, file_path: str, collection=None, recursive=False):
    """Add one or more files to the index."""

    # Resolve path cross-platform
    raw_path = os.path.normpath(file_path)

    if not os.path.exists(raw_path):
        print(f"Error: Path not found: {raw_path}", file=sys.stderr)
        return

    # Determine collection name
    coll = collection or os.path.basename(os.path.dirname(raw_path)) or "default"

    # Discover files to add
    if os.path.isfile(raw_path):
        files_to_add = [raw_path]
    else:
        # Directory scan
        files_to_add = []
        for root, dirs, filenames in os.walk(raw_path):
            # Respect recursive flag
            rel = os.path.relpath(root, raw_path)
            if not recursive and rel != ".":
                dirs.clear()  # Don't recurse into subdirs
                continue
            for fn in sorted(filenames):
                if fn.endswith(('.md', '.txt', '.rst')):
                    files_to_add.append(os.path.join(root, fn))

    if not files_to_add:
        print("No .md/.txt/.rst files found.")
        return

    # Get max existing ID for new IDs -- check both documents and FTS tables
    cur = db.conn.cursor()
    max_doc_id = cur.execute("SELECT MAX(id) FROM documents").fetchone()[0] or 0
    try:
        max_fts_rowid = cur.execute("SELECT MAX(rowid) FROM documents_fts").fetchone()[0] or 0
    except Exception:
        max_fts_rowid = 0
    next_id = max(max_doc_id, max_fts_rowid) + 1

    added = 0
    skipped = 0
    updated = 0

    for fpath in files_to_add:
        try:
            with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
        except Exception as e:
            print(f"  [SKIP] Cannot read {fpath}: {e}", file=sys.stderr)
            skipped += 1
            continue

        if not content.strip():
            print(f"  [SKIP] Empty file: {fpath}")
            skipped += 1
            continue

        # Compute hash (sha256 of file content)
        file_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()

        # Check if this exact content already exists in the index -- skip duplicates
        existing_by_hash = cur.execute(
            "SELECT id FROM documents WHERE hash=? AND active=1",
            (file_hash,)
        ).fetchone()

        if existing_by_hash:
            print(f"  [SKIP] Duplicate content (hash {file_hash[:8]}...) -- already indexed as #{existing_by_hash[0]}")
            skipped += 1
            continue

        # Use relative path from the scanned root for cleaner storage
        rel_path = os.path.relpath(fpath, raw_path) if len(files_to_add) > 1 else os.path.basename(fpath)
        title = content.split('\n')[0].lstrip('#').strip() or os.path.basename(rel_path)

        # Check if already indexed (same path in same collection)
        existing = cur.execute(
            "SELECT id FROM documents WHERE collection=? AND path=? AND active=1",
            (coll, rel_path)
        ).fetchone()

        now = datetime.now().isoformat() + "Z"

        if existing:
            doc_id = existing[0]
            # Update existing record
            cur.execute("""
                UPDATE documents SET
                    path=?, title=?, hash=?, modified_at=?
                WHERE id=?
            """, (rel_path, title, file_hash, now, doc_id))

            # Update content table
            try:
                cur.execute("UPDATE content SET doc=?, created_at=? WHERE doc=?",
                           (content, now, file_hash))
            except Exception:
                pass

            # Update FTS (rowid = id for documents_fts)
            try:
                cur.execute(f"INSERT OR REPLACE INTO documents_fts (rowid, filepath, title, body) VALUES (?, ?, ?, ?)",
                           (doc_id, rel_path, title, content[:50000]))  # Cap FTS body size
            except Exception as e:
                print(f"  [WARN] FTS update failed for {rel_path}: {e}", file=sys.stderr)

            updated += 1
        else:
            # Insert new document
            cur.execute("""
                INSERT INTO documents (id, collection, path, title, hash, created_at, modified_at, active)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1)
            """, (next_id, coll, rel_path, title, file_hash, now, now))

            # Insert content
            try:
                cur.execute("""
                    INSERT INTO content (hash, doc, created_at) VALUES (?, ?, ?)
                """, (file_hash, content, now))
            except Exception as e:
                print(f"  [WARN] Content insert failed for {rel_path}: {e}", file=sys.stderr)

            # Insert into FTS5 — use INSERT OR REPLACE to handle conflicts with orphaned entries
            try:
                cur.execute(f"INSERT OR REPLACE INTO documents_fts (rowid, filepath, title, body) VALUES (?, ?, ?, ?)",
                           (next_id, rel_path, title, content[:50000]))
            except Exception as e:
                print(f"  [WARN] FTS insert failed for {rel_path}: {e}", file=sys.stderr)

            added += 1
            next_id += 1

    db.conn.commit()

    # Clean up FTS5 gaps (required after INSERT OR REPLACE in some SQLite versions)
    try:
        cur.execute(f"OPTIMIZE documents_fts")
    except Exception:
        pass

    print(f"\nAdd complete:")
    print(f"  Added: {added} new documents")
    print(f"  Updated: {updated} existing documents")
    if skipped > 0:
        print(f"  Skipped: {skipped}")
    print(f"  Collection: {coll}")


def cmd_remove(db, doc_id_or_path):
    """Remove a document from the index (soft delete via active=0)."""
    cur = db.conn.cursor()

    # Try as integer ID first
    try:
        doc_id = int(doc_id_or_path)
        row = cur.execute("SELECT id, path FROM documents WHERE id=?", (doc_id,)).fetchone()
        if not row:
            print(f"Document #{doc_id} not found.")
            return
    except ValueError:
        # Try as file path
        row = cur.execute(
            "SELECT id, path FROM documents WHERE path=? AND active=1",
            (doc_id_or_path,)
        ).fetchone()
        if not row:
            print(f"Document with path '{doc_id_or_path}' not found or already removed.")
            return

    doc_id = row[0]

    # Soft delete: set active=0
    cur.execute("UPDATE documents SET active=0 WHERE id=?", (doc_id,))

    # Remove from FTS5 (need to use the documents_fts table directly)
    try:
        cur.execute(f"DELETE FROM documents_fts WHERE rowid=?", (doc_id,))
    except Exception as e:
        print(f"[WARN] FTS delete failed: {e}", file=sys.stderr)

    # Remove from content_vectors if exists
    try:
        doc = db.get_document_by_id(doc_id)
        if doc and doc.get("hash"):
            cur.execute("DELETE FROM content_vectors WHERE hash=?", (doc["hash"],))
    except Exception:
        pass

    # Also remove from content table using the old hash
    try:
        if doc and doc.get("hash"):
            cur.execute("DELETE FROM content WHERE doc=?", (doc["hash"],))
    except Exception:
        pass

    db.conn.commit()

    print(f"Removed document #{doc_id} ({row[1]})")


def cmd_update(db, collection=None):
    """Re-scan collections and update changed documents."""

    cur = db.conn.cursor()

    # Get collections to scan
    if collection:
        colls_to_scan = [collection]
    else:
        colls_to_scan = list(set(
            r[0] for r in cur.execute(
                "SELECT DISTINCT collection FROM documents WHERE active=1"
            ).fetchall()
        ))

    total_added = 0
    total_updated = 0
    total_removed = 0

    for coll_name in colls_to_scan:
        # Get all paths and hashes for this collection
        existing_docs = cur.execute(
            "SELECT id, path, hash FROM documents WHERE collection=? AND active=1",
            (coll_name,)
        ).fetchall()

        existing_map = {row[1]: row for row in existing_docs}  # path -> (id, path, hash)

        # Find source directory for this collection from store_collections or documents table
        path_row = cur.execute(
            "SELECT DISTINCT path FROM documents WHERE collection=? LIMIT 1",
            (coll_name,)
        ).fetchone()

        if not path_row:
            print(f"  [{coll_name}] No source paths found, skipping.")
            continue

        src_dir = path_row[0]
        # Normalize the path for OS compatibility
        if os.sep == "\\":
            src_dir = src_dir.replace("/", "\\")

        if not os.path.exists(src_dir):
            print(f"  [{coll_name}] Source directory not found: {src_dir}")
            continue

        # Scan all .md files in source dir
        scanned_paths = set()
        for root, dirs, filenames in os.walk(src_dir):
            rel = os.path.relpath(root, src_dir)
            if rel != ".":
                dirs.clear()  # Non-recursive: only top level
                continue

            for fn in sorted(filenames):
                if not fn.endswith(('.md', '.txt', '.rst')):
                    continue

                fpath = os.path.join(root, fn)
                try:
                    with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                except Exception:
                    continue

                if not content.strip():
                    continue

                file_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()

                # Use path relative to src_dir for matching
                rel_path = os.path.relpath(fpath, root) if rel == "." else os.path.join(rel, fn)
                scanned_paths.add(rel_path)

                title = content.split('\n')[0].lstrip('#').strip() or fn

                if rel_path in existing_map:
                    # Check for changes
                    old_id, _, old_hash = existing_map[rel_path]
                    if old_hash != file_hash:
                        now = datetime.now().isoformat() + "Z"
                        cur.execute("""
                            UPDATE documents SET
                                title=?, hash=?, modified_at=?
                            WHERE id=?
                        """, (title, file_hash, now, old_id))

                        # Update content
                        try:
                            cur.execute("UPDATE content SET doc=?, created_at=? WHERE doc=?",
                                       (file_hash, now, old_hash))
                        except Exception:
                            pass

                        # Update FTS
                        try:
                            body = content[:50000]
                            cur.execute(f"""
                                INSERT OR REPLACE INTO documents_fts (rowid, filepath, title, body)
                                VALUES (?, ?, ?, ?)
                            """, (old_id, rel_path, title, body))
                        except Exception as e:
                            print(f"    [WARN] FTS update failed: {e}", file=sys.stderr)

                        total_updated += 1
                else:
                    # New file -- not yet indexed
                    pass

        # Check for removed files (in DB but not on disk)
        for path, doc_info in existing_map.items():
            if path not in scanned_paths:
                # File was deleted from disk -- soft remove it
                doc_id = doc_info[0]
                cur.execute("UPDATE documents SET active=0 WHERE id=?", (doc_id,))
                try:
                    cur.execute(f"DELETE FROM documents_fts WHERE rowid=?", (doc_id,))
                except Exception:
                    pass
                total_removed += 1

    db.conn.commit()

    print(f"\nUpdate complete:")
    print(f"  Added: {total_added} new documents")
    print(f"  Updated: {total_updated} changed documents")
    if total_removed > 0:
        print(f"  Removed (deleted from disk): {total_removed}")


def cmd_optimize(db, full=False):
    """Optimize the FTS index and clean up orphaned/stale entries.

    Args:
        db: QMDDB instance
        full: If True, also remove soft-deleted documents permanently and vacuum DB
    """
    cur = db.conn.cursor()

    print(f"\n{'='*70}")
    print(f"  Optimizing Index")
    print(f"{'='*70}\n")

    # ---- Step 1: Remove orphaned FTS rowids ----
    # Orphaned = rowid exists in documents_fts but NOT in documents table
    try:
        fts_orphans = cur.execute("""
            SELECT COUNT(*) FROM documents_fts
            WHERE rowid NOT IN (SELECT id FROM documents)
        """).fetchone()[0]

        if fts_orphans > 0:
            print(f"Removing {fts_orphans} orphaned FTS entries...")
            cur.execute("""
                DELETE FROM documents_fts
                WHERE rowid NOT IN (SELECT id FROM documents)
            """)
            db.conn.commit()
            print(f"  Removed {fts_orphans} orphaned FTS entries.")
        else:
            print("No orphaned FTS entries found.")
    except Exception as e:
        print(f"[WARN] Orphan cleanup failed: {e}", file=sys.stderr)

    # ---- Step 2: Remove soft-deleted documents' stale data ----
    deleted_count = cur.execute(
        "SELECT COUNT(*) FROM documents WHERE active=0"
    ).fetchone()[0]

    if deleted_count > 0 and full:
        print(f"\nPermanently removing {deleted_count} soft-deleted documents...")
        # Delete from content_vectors
        try:
            hashes = [r[0] for r in cur.execute(
                "SELECT hash FROM documents WHERE active=0"
            ).fetchall()]
            if hashes:
                placeholders = ",".join(["?"] * len(hashes))
                cur.execute(f"""
                    DELETE FROM content_vectors WHERE hash IN ({placeholders})
                """, hashes)
        except Exception as e:
            print(f"[WARN] content_vectors cleanup failed: {e}", file=sys.stderr)

        # Delete from content
        try:
            if hashes:
                cur.execute("""
                    DELETE FROM content WHERE doc IN (
                        SELECT hash FROM documents WHERE active=0
                    )
                """)
        except Exception as e:
            print(f"[WARN] content cleanup failed: {e}", file=sys.stderr)

        # Delete from documents
        try:
            cur.execute("DELETE FROM documents WHERE active=0")
        except Exception as e:
            print(f"[WARN] documents delete failed: {e}", file=sys.stderr)

        db.conn.commit()
        print(f"  Permanently removed {deleted_count} soft-deleted documents.")
    elif deleted_count > 0:
        print(f"\n{deleted_count} soft-deleted documents still present (use --full to permanently remove).")

    # ---- Step 3: Compact FTS index ----
    try:
        cur.execute("PRAGMA journal_mode = WAL;")
        cur.execute("INSERT INTO documents_fts(documents_fts) VALUES('compact');")
        print("\nFTS index compacted.")
    except Exception as e:
        # 'compact' may not be available in all SQLite builds; try 'rebuild' instead
        try:
            cur.execute("INSERT INTO documents_fts(documents_fts) VALUES('rebuild');")
            print("\nFTS index rebuilt.")
        except Exception as e2:
            print(f"[WARN] FTS compact/rebuild failed ({e2}), trying optimize...")
            try:
                cur.execute("INSERT INTO documents_fts(documents_fts) VALUES('optimize');")
                print("\nFTS index optimized.")
            except Exception as e3:
                print(f"[WARN] All FTS maintenance operations failed: {e3}")

   # ---- Step 4: Vacuum database (only with --full) ----
    if full:
        db_size_before = os.path.getsize(db.db_path) / (1024*1024)
        try:
            cur.execute("VACUUM;")
            print(f"\nDatabase vacuumed ({db_size_before:.1f} MB -> {os.path.getsize(db.db_path)/(1024*1024):.1f} MB).")
        except Exception as e:
            print(f"[WARN] VACUUM failed: {e}, trying manual approach...")
            # VACUUM doesn't work inside transactions — commit first, then retry
            db.conn.commit()
            try:
                cur.execute("VACUUM;")
                print(f"Database vacuumed successfully after re-commit.")
            except Exception as e2:
                print(f"[WARN] VACUUM still failed: {e2}")

    # ---- Step 5: Final stats (reconnect to ensure fresh state) ----
    total_active = cur.execute("SELECT COUNT(*) FROM documents WHERE active=1").fetchone()[0]
    total_fts = cur.execute("SELECT COUNT(*) FROM documents_fts").fetchone()[0]
    print(f"\nFinal state:")
    print(f"  Active documents: {total_active}")
    print(f"  FTS entries: {total_fts}")

    if full:
        db_size = os.path.getsize(db.db_path) / (1024*1024)
        print(f"  Database size: {db_size:.1f} MB")


def _extract_xlmeta_content(filepath: str) -> Optional[str]:
    """Extract text content from MinIO xl.meta binary format.

    xl.meta files are MinIO distributed storage metadata with inline data.
    The actual file content is embedded after the 'x-minio-internal-inline-data' marker.

    Strategy: scan for first contiguous block of printable ASCII (>=30 chars)
    starting from byte position ~30 onwards.
    """
    try:
        with open(filepath, 'rb') as f:
            raw = f.read()
    except Exception:
        return None

    # Look for inline-data marker
    marker = b'x-minio-internal-inline-data'
    idx = raw.find(marker)
    if idx >= 0:
        content_bytes = raw[idx + len(marker):]
        # Decode as UTF-8, replace errors
        try:
            return content_bytes.decode('utf-8', errors='replace').strip()
        except Exception:
            pass

    # Fallback: scan for printable ASCII runs >= 30 chars from position 30
    MIN_PRINTABLE = 30
    printable_chars = set(range(32, 127)) | {ord('\n'), ord('\r'), ord('\t')}

    best_run_start = -1
    best_run_len = 0
    current_start = -1
    current_len = 0

    for i in range(30, len(raw)):
        if raw[i] in printable_chars:
            if current_start < 0:
                current_start = i
            current_len += 1
        else:
            if current_len > best_run_len and current_len >= MIN_PRINTABLE:
                best_run_start = current_start
                best_run_len = current_len
            current_start = -1
            current_len = 0

    if best_run_len >= MIN_PRINTABLE:
        try:
            return raw[best_run_start:best_run_start + best_run_len].decode('utf-8', errors='replace').strip()
        except Exception:
            pass

    # Last resort: decode entire file as UTF-8
    try:
        text = raw.decode('utf-8', errors='replace')
        if len(text) > 50:
            return text.strip()
    except Exception:
        pass

    return None


def _discover_minio_collections(base_dir: str) -> List[Dict]:
    """Discover MinIO backup collections under base_dir.

    Returns list of dicts with 'collection_name', 'xlmeta_path' for each xl.meta file found.
    Skips deeply nested directories (depth > 2).
    """
    collections = []
    base_depth = base_dir.rstrip(os.sep + os.path.dirname(os.sep)).count(os.sep)

    for root, dirs, filenames in os.walk(base_dir):
        depth = root.count(os.sep) - base_depth
        if depth > 2:
            dirs.clear()
            continue

        for fn in filenames:
            if fn == 'xl.meta':
                xlmeta_path = os.path.join(root, fn)
                # Derive collection name from directory structure
                rel = os.path.relpath(root, base_dir)
                parts = [p for p in rel.split(os.sep) if p]
                coll_name = "/".join(parts[-2:]) if len(parts) >= 2 else (parts[0] if parts else "unknown")
                collections.append({
                    'collection_name': coll_name,
                    'xlmeta_path': xlmeta_path,
                    'root_dir': root,
                })

    return collections


def cmd_import(db, base_dir: str, collection_prefix=None):
    """Batch import MinIO xl.meta files into the QMD index.

    Args:
        db: QMDDB instance
        base_dir: Root directory containing MinIO backup structure
        collection_prefix: Optional prefix for collection names (e.g., 'minio-backup')
    """
    print(f"\n{'='*70}")
    print(f"  Importing from MinIO xl.meta files")
    print(f"  Source: {base_dir}")
    if collection_prefix:
        print(f"  Collection prefix: {collection_prefix}")
    print(f"{'='*70}\n")

    # Discover collections
    if not os.path.exists(base_dir):
        print(f"Error: Directory not found: {base_dir}", file=sys.stderr)
        return

    collections = _discover_minio_collections(base_dir)

    if not collections:
        print("No xl.meta files found.")
        return

    print(f"Found {len(collections)} collection(s):\n")
    for c in collections:
        name = f"{collection_prefix}/{c['collection_name']}" if collection_prefix else c['collection_name']
        print(f"  {name}: {os.path.basename(os.path.dirname(c['xlmeta_path']))}/")

    # Process each collection
    total_added = 0
    total_skipped = 0
    total_errors = 0

    for coll_info in collections:
        xlmeta_path = coll_info['xlmeta_path']
        coll_name = f"{collection_prefix}/{coll_info['collection_name']}" if collection_prefix else coll_info['collection_name']

        print(f"\n--- Processing: {coll_name} ---")

        content = _extract_xlmeta_content(xlmeta_path)
        if not content or len(content) < 10:
            print(f"  [SKIP] Could not extract meaningful content from xl.meta")
            total_skipped += 1
            continue

        # Split content into individual documents by Markdown headers (## Title patterns)
        # Common MinIO structure: multiple .md files concatenated with header separators
        docs = _split_xlmeta_content(content, coll_info['root_dir'])

        if not docs:
            print(f"  [SKIP] No document boundaries found in content.")
            total_skipped += 1
            continue

        # Get max ID for this batch
        cur = db.conn.cursor()
        max_doc_id = cur.execute("SELECT MAX(id) FROM documents").fetchone()[0] or 0
        try:
            max_fts_rowid = cur.execute("SELECT MAX(rowid) FROM documents_fts").fetchone()[0] or 0
        except Exception:
            max_fts_rowid = 0
        next_id = max(max_doc_id, max_fts_rowid) + 1

        for doc in docs:
            try:
                file_hash = hashlib.sha256(doc['content'].encode('utf-8')).hexdigest()

                # Check for global duplicate by hash
                existing_by_hash = cur.execute(
                    "SELECT id FROM documents WHERE hash=? AND active=1", (file_hash,)
                ).fetchone()
                if existing_by_hash:
                    total_skipped += 1
                    continue

                title = doc['content'].split('\n')[0].lstrip('#').strip() or os.path.basename(doc.get('filename', 'untitled'))
                now = datetime.now().isoformat() + "Z"

                # Insert document
                cur.execute("""
                    INSERT INTO documents (id, collection, path, title, hash, created_at, modified_at, active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                """, (next_id, coll_name, doc['path'], title, file_hash, now, now))

                # Insert content
                try:
                    cur.execute("""
                        INSERT INTO content (hash, doc, created_at) VALUES (?, ?, ?)
                    """, (file_hash, file_hash, now))
                except Exception:
                    pass

                # Insert FTS
                try:
                    body = doc['content'][:50000]
                    cur.execute(f"""
                        INSERT OR REPLACE INTO documents_fts (rowid, filepath, title, body)
                        VALUES (?, ?, ?, ?)
                    """, (next_id, doc['path'], title, body))
                except Exception as e:
                    print(f"    [WARN] FTS insert failed for {title}: {e}", file=sys.stderr)

                total_added += 1
                next_id += 1

            except Exception as e:
                total_errors += 1
                print(f"    [ERROR] Failed to index '{doc.get('filename', '?')}': {e}")

        db.conn.commit()

        # Compact FTS for this collection batch
        try:
            cur.execute("INSERT INTO documents_fts(documents_fts) VALUES('optimize')")
            db.conn.commit()
        except Exception:
            pass

    print(f"\n{'='*70}")
    print(f"  Import Complete")
    print(f"{'='*70}\n")
    print(f"  Added: {total_added} documents")
    if total_skipped > 0:
        print(f"  Skipped (duplicates): {total_skipped}")
    if total_errors > 0:
        print(f"  Errors: {total_errors}")


def _split_xlmeta_content(content: str, root_dir: str) -> List[Dict]:
    """Split xl.meta content into individual documents.

    Handles common MinIO structures where multiple markdown files are stored
    with metadata. Attempts to split by common document boundaries:
    - Markdown level-1 headers (## or # at start of line)
    - File markers if present in the metadata
    """
    docs = []

    # Strategy 1: Try splitting by ## header markers (common for concatenated files)
    lines = content.split('\n')
    current_doc_lines = []
    doc_boundaries = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        # Level-2 or level-1 headers that likely indicate a new document section
        if stripped.startswith('## ') and len(stripped) < 80:
            if current_doc_lines:
                doc_boundaries.append((i, '\n'.join(current_doc_lines)))
            current_doc_lines = [line]
        elif stripped.startswith('# ') and len(stripped) < 80 and i > 2:
            # Level-1 header (top-level title), treat as new document if not first doc
            if current_doc_lines and len(current_doc_lines) > 5:
                doc_boundaries.append((i, '\n'.join(current_doc_lines)))
            current_doc_lines = [line]
        else:
            current_doc_lines.append(line)

    # Don't forget the last document
    if current_doc_lines:
        doc_boundaries.append((len(lines), '\n'.join(current_doc_lines)))

    # If we found multiple documents, use them
    if len(doc_boundaries) > 1:
        for i, (start_idx, text) in enumerate(doc_boundaries):
            content_stripped = text.strip()
            if not content_stripped or len(content_stripped) < 20:
                continue

            # Generate a filename from the first header line
            title_line = content_stripped.split('\n')[0].lstrip('#').strip()
            safe_name = ''.join(c if c.isalnum() else '_' for c in title_line)[:50] or f'doc_{i}'
            filename = f"{safe_name}.md"

            docs.append({
                'content': content_stripped,
                'filename': filename,
                'path': os.path.join(root_dir, filename),
            })
    else:
        # Single document — use the entire content as one file
        title_line = lines[0].lstrip('#').strip() if lines else 'untitled'
        safe_name = ''.join(c if c.isalnum() else '_' for c in title_line)[:50] or 'document'
        docs.append({
            'content': content,
            'filename': f"{safe_name}.md",
            'path': os.path.join(root_dir, f"{safe_name}.md"),
        })

    return docs


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(
        description="QMD Index v3.0 - Local semantic search",
        epilog="""Examples:
  python qmd.py list
  python qmd.py search "EvoMap"
  python qmd.py search "golf design" --collection skills
  python qmd.py search "高尔夫球场设计" --semantic
  python qmd.py show 72
  python qmd.py stats
  python qmd.py add ./notes/new-idea.md -c workspace
  python qmd.py add ./my-docs/ -r -c my-collection
  python qmd.py remove 142
  python qmd.py update --collection workspace
  python qmd.py optimize              # Clean up orphaned FTS entries
  python qmd.py optimize --full       # Full cleanup + vacuum database
  python qmd.py import /data/minio-backup -p minio-backup""")

    sub = parser.add_subparsers(dest="command")

    p_search = sub.add_parser("search", help="Search documents")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--limit", "-n", type=int, default=DEFAULT_LIMIT)
    p_search.add_argument("--collection", "-c", type=str, default=None)
    p_search.add_argument("--semantic", action="store_true", help="Use embedding+rerank")

    sub.add_parser("list", help="List collections")
    p_show = sub.add_parser("show", help="Show document")
    p_show.add_argument("doc_id", type=int)
    sub.add_parser("stats", help="Index statistics")

    # CRUD commands
    p_add = sub.add_parser("add", help="Add file(s) or directory to index")
    p_add.add_argument("path", help="File or directory path to add")
    p_add.add_argument("-c", "--collection", type=str, default=None,
                       help="Collection name (default: same as source dir)")
    p_add.add_argument("--recursive", "-r", action="store_true",
                       help="Scan subdirectories recursively")

    p_remove = sub.add_parser("remove", help="Remove document from index")
    p_remove.add_argument("doc_id_or_path", help="Document ID or file path to remove")

    p_update = sub.add_parser("update", help="Re-scan and update existing documents")
    p_update.add_argument("-c", "--collection", type=str, default=None,
                          help="Collection name to update (default: all)")

    # Optimize command
    p_optimize = sub.add_parser("optimize", help="Optimize FTS index + clean up orphans")
    p_optimize.add_argument("--full", "-f", action="store_true",
                            help="Full cleanup: permanently remove soft-deleted docs + vacuum DB")

    # Import command for MinIO xl.meta files
    p_import = sub.add_parser("import", help="Batch import from MinIO xl.meta backups")
    p_import.add_argument("base_dir", help="Root directory containing xl.meta files")
    p_import.add_argument("-p", "--prefix", type=str, default=None,
                         help="Collection name prefix (e.g., 'minio-backup')")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    db = QMDDB(INDEX_PATH)
    try:
        if args.command == "list":
            cmd_list(db)
        elif args.command == "search":
            cmd_search(db, args.query, limit=args.limit,
                       collection=args.collection, use_semantic=args.semantic)
        elif args.command == "show":
            cmd_show(db, args.doc_id)
        elif args.command == "stats":
            cmd_stats(db)
        elif args.command == "add":
            cmd_add(db, args.path, collection=args.collection, recursive=args.recursive)
        elif args.command == "remove":
            cmd_remove(db, args.doc_id_or_path)
        elif args.command == "update":
            cmd_update(db, collection=args.collection)
        elif args.command == "optimize":
            cmd_optimize(db, full=args.full)
        elif args.command == "import":
            cmd_import(db, args.base_dir, collection_prefix=args.prefix)
    finally:
        db.close()


if __name__ == "__main__":
    main()
