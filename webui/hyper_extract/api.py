"""
Hyper-Extract Flask Blueprint
Exposes /api/hyper/* endpoints for knowledge graph operations.
"""

import json
import sys
import os
import sqlite3
import shutil
import tempfile
import threading
from pathlib import Path

from flask import Blueprint, request, jsonify, Response

from .db import HyperDB
from .extractor import HyperExtractor
from .graph import HyperGraph

# ── Docker 9p volume workaround (read path) ────────────────────────
# Docker Desktop WSL2 mounts Windows directories through 9p (drvfs), which
# cannot host SQLite's WAL shared-memory file: while a -wal file is present,
# every statement on that mount fails with "disk I/O error". The same file
# works on the host. When the database sits on a 9p mount, reads are served
# from a local /tmp snapshot.
#
# The snapshot is read-only by design. A container is not a writer here:
# copying a modified database back over the file the native host server holds
# open leaves a stale -wal beside a replaced main file, which corrupts
# indexes. Mutations are delegated to the host API instead.
_IS_9P = False
_ORIG_DB_PATH = None
_TMP_DB_PATH = None

# Every connection points at the same /tmp snapshot, so refreshing it must
# not overlap another refresh.
_db_lock = threading.Lock()


def _detect_9p():
    """Resolve the DB path and record whether it sits on a 9p mount."""
    global _IS_9P, _ORIG_DB_PATH, _TMP_DB_PATH
    db = _find_db_path()
    if not db:
        return False
    _ORIG_DB_PATH = db
    _TMP_DB_PATH = os.path.join(tempfile.gettempdir(), "qmd_index_copy.sqlite")
    try:
        with open("/proc/mounts") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) < 3 or parts[2] != "9p":
                    continue
                mount_point = parts[1].rstrip("/") or "/"
                if mount_point == "/" or db.startswith(mount_point + "/"):
                    _IS_9P = True
                    return True
    except OSError:
        pass
    _IS_9P = False
    return False


def _ensure_tmp_db():
    """Return the DB path to read from, refreshing the /tmp snapshot."""
    if not _IS_9P or not _ORIG_DB_PATH:
        return _ORIG_DB_PATH
    if not os.path.exists(_ORIG_DB_PATH):
        return _ORIG_DB_PATH
    src_mtime = os.path.getmtime(_ORIG_DB_PATH)
    if (os.path.exists(_TMP_DB_PATH)
            and os.path.getmtime(_TMP_DB_PATH) >= src_mtime):
        return _TMP_DB_PATH
    with _db_lock:
        # Re-check under the lock: another request may have refreshed it.
        if (os.path.exists(_TMP_DB_PATH)
                and os.path.getmtime(_TMP_DB_PATH) >= src_mtime):
            return _TMP_DB_PATH
        shutil.copy2(_ORIG_DB_PATH, _TMP_DB_PATH)
        for suffix in ("-wal", "-shm"):
            src = _ORIG_DB_PATH + suffix
            if os.path.exists(src):
                try:
                    shutil.copy2(src, _TMP_DB_PATH + suffix)
                except OSError:
                    pass
    return _TMP_DB_PATH


hyper_api = Blueprint("hyper_extract", __name__, url_prefix="/api/hyper")

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def _get_settings():
    """Load runtime settings, allowing env var overrides for Docker."""
    settings_path = Path(__file__).resolve().parent.parent / "settings.json"
    try:
        with open(settings_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        cfg = {}
    # Environment variable overrides (for Docker)
    if os.environ.get("LLM_URL"):
        cfg["llm_url"] = os.environ["LLM_URL"]
    if os.environ.get("LLM_MODEL"):
        cfg["llm_model"] = os.environ["LLM_MODEL"]
    return cfg


def _find_db_path():
    """Find the QMD index database path by scanning common locations."""
    candidates = [
        BASE_DIR / "models" / "qmd" / "index.sqlite",
        BASE_DIR.parent / "models" / "qmd" / "index.sqlite",
        Path("/app/models/qmd/index.sqlite"),
    ]
    for p in candidates:
        p = p.resolve()
        if p.exists():
            return str(p)
    return None


# Resolve the read path once, now that _find_db_path exists.
_detect_9p()


def _get_hyper_db():
    """Create a new HyperDB instance per request (thread-safe)."""
    db_path = _ensure_tmp_db()
    if not db_path or not os.path.exists(db_path):
        return None, "Cannot find index database"
    try:
        hdb = HyperDB(db_path)
        # Attach a reference to the lock so handlers can use it
        hdb._lock = _db_lock
        return hdb, None
    except Exception as e:
        return None, str(e)


def _get_extractor():
    settings = _get_settings()
    return HyperExtractor.from_settings(settings)


# ── Status ──

@hyper_api.route("/status", methods=["GET"])
def status():
    hdb, err = _get_hyper_db()
    if err:
        return jsonify({"error": err}), 500
    s = hdb.stats()
    return jsonify(s)


# ── Entities ──

@hyper_api.route("/entities", methods=["GET"])
def list_entities():
    hdb, err = _get_hyper_db()
    if err:
        return jsonify({"error": err}), 500
    typ = request.args.get("type")
    query = request.args.get("q")
    limit = int(request.args.get("limit", 200))
    offset = int(request.args.get("offset", 0))
    if query:
        results = hdb.search_entities(query, limit=limit)
    else:
        results = hdb.list_entities(typ=typ, limit=limit, offset=offset)
    types = hdb.get_entity_types()
    return jsonify({"entities": results, "types": types, "count": len(results)})


@hyper_api.route("/entities/<int:eid>", methods=["GET"])
def get_entity(eid):
    hdb, err = _get_hyper_db()
    if err:
        return jsonify({"error": err}), 500
    entity = hdb.get_entity(eid)
    if not entity:
        return jsonify({"error": "Entity not found"}), 404
    rels = hdb.get_relationships(entity_id=eid)
    docs = hdb.get_entity_docs(eid)
    return jsonify({"entity": entity, "relationships": rels, "documents": docs})


@hyper_api.route("/entities/<int:eid>", methods=["DELETE"])
def delete_entity(eid):
    hdb, err = _get_hyper_db()
    if err:
        return jsonify({"error": err}), 500
    hdb.delete_entity(eid)
    return jsonify({"success": True})


@hyper_api.route("/entities/merge", methods=["POST"])
def merge_entities():
    """Merge two entities into one (keep target, redirect source)."""
    hdb, err = _get_hyper_db()
    if err:
        return jsonify({"error": err}), 500
    data = request.get_json() or {}
    source_id = data.get("source_id")
    target_id = data.get("target_id")
    if not source_id or not target_id:
        return jsonify({"error": "source_id and target_id required"}), 400
    if source_id == target_id:
        return jsonify({"error": "Cannot merge entity with itself"}), 400
    src = hdb.get_entity(source_id)
    tgt = hdb.get_entity(target_id)
    if not src or not tgt:
        return jsonify({"error": "Entity not found"}), 404

    # Reassign all relationships from source to target
    hdb.conn.execute(
        "UPDATE hyper_relationships SET source_id=? WHERE source_id=?",
        (target_id, source_id))
    hdb.conn.execute(
        "UPDATE hyper_relationships SET target_id=? WHERE target_id=?",
        (target_id, source_id))
    # Reassign doc links
    hdb.conn.execute(
        "UPDATE hyper_doc_entities SET entity_id=? WHERE entity_id=?",
        (target_id, source_id))
    # Delete source entity
    hdb.conn.execute("DELETE FROM hyper_entities WHERE id=?", (source_id,))
    hdb.conn.commit()
    return jsonify({"success": True, "source": src["name"], "target": tgt["name"]})


@hyper_api.route("/entities", methods=["POST"])
def create_entity():
    """Create a new entity manually."""
    hdb, err = _get_hyper_db()
    if err:
        return jsonify({"error": err}), 500
    data = request.get_json() or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "name required"}), 400
    typ = data.get("type", "concept")
    description = data.get("description", "")
    try:
        eid = hdb.upsert_entity(name, typ, description)
        hdb.conn.commit()
        ent = hdb.get_entity(eid)
        return jsonify({"success": True, "entity": ent}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@hyper_api.route("/entities/<int:eid>", methods=["PUT"])
def update_entity(eid):
    """Update an existing entity's name, type, or description."""
    hdb, err = _get_hyper_db()
    if err:
        return jsonify({"error": err}), 500
    existing = hdb.get_entity(eid)
    if not existing:
        return jsonify({"error": "Entity not found"}), 404
    data = request.get_json() or {}
    name = data.get("name", existing["name"])
    typ = data.get("type", existing["type"])
    description = data.get("description", existing.get("description", ""))
    try:
        hdb.upsert_entity(name, typ, description)
        hdb.conn.commit()
        ent = hdb.get_entity(eid)
        return jsonify({"success": True, "entity": ent})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Relationships ──

@hyper_api.route("/relationships", methods=["POST"])
def create_relationship():
    """Create a new relationship between two entities."""
    hdb, err = _get_hyper_db()
    if err:
        return jsonify({"error": err}), 500
    data = request.get_json() or {}
    source_id = data.get("source_id")
    target_id = data.get("target_id")
    if not source_id or not target_id:
        return jsonify({"error": "source_id and target_id required"}), 400
    rel_type = data.get("rel_type", "related_to")
    weight = float(data.get("weight", 1.0))
    context = data.get("context", "")
    src = hdb.get_entity(source_id)
    tgt = hdb.get_entity(target_id)
    if not src:
        return jsonify({"error": f"Source entity #{source_id} not found"}), 404
    if not tgt:
        return jsonify({"error": f"Target entity #{target_id} not found"}), 404
    try:
        rid = hdb.upsert_relationship(source_id, target_id, rel_type, weight, context)
        hdb.conn.commit()
        rels = hdb.get_relationships(entity_id=source_id, limit=50)
        return jsonify({"success": True, "relationship_id": rid, "relationships": rels}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@hyper_api.route("/relationships/<int:rid>", methods=["DELETE"])
def delete_relationship(rid):
    """Delete a relationship."""
    hdb, err = _get_hyper_db()
    if err:
        return jsonify({"error": err}), 500
    hdb.conn.execute("DELETE FROM hyper_relationships WHERE id=?", (rid,))
    hdb.conn.commit()
    return jsonify({"success": True})


# ── Relationships ──

@hyper_api.route("/relationships", methods=["GET"])
def list_relationships():
    hdb, err = _get_hyper_db()
    if err:
        return jsonify({"error": err}), 500
    eid = request.args.get("entity_id", type=int)
    limit = int(request.args.get("limit", 200))
    results = hdb.get_relationships(entity_id=eid, limit=limit)
    types = hdb.get_relationship_types()
    return jsonify({"relationships": results, "types": types, "count": len(results)})


# ── Graph export (vis-network compatible) ──

@hyper_api.route("/graph", methods=["GET"])
def graph():
    hdb, err = _get_hyper_db()
    if err:
        return jsonify({"error": err}), 500
    limit = int(request.args.get("limit", 300))
    min_weight = float(request.args.get("min_weight", 0.0))
    g = hdb.export_graph(limit_entities=limit, min_weight=min_weight)
    return jsonify(g)


# ── Neighborhood ──

@hyper_api.route("/neighborhood/<int:eid>", methods=["GET"])
def neighborhood(eid):
    hdb, err = _get_hyper_db()
    if err:
        return jsonify({"error": err}), 500
    settings = _get_settings()
    extractor = HyperExtractor.from_settings(settings)
    hg = HyperGraph(hdb, extractor)
    depth = int(request.args.get("depth", 1))
    result = hg.get_entity_neighborhood(eid, depth=depth)
    return jsonify(result)


# ── Extraction ──

@hyper_api.route("/extract", methods=["POST"])
def extract_document():
    """Extract knowledge from a specific document by ID."""
    hdb, err = _get_hyper_db()
    if err:
        return jsonify({"error": err}), 500
    data = request.get_json() or {}
    doc_id = data.get("doc_id")
    if not doc_id:
        return jsonify({"error": "doc_id required"}), 400

    # Get document content from the main index DB
    doc = hdb.conn.execute(
        "SELECT id, hash, collection FROM documents WHERE id=? AND active=1",
        (doc_id,)).fetchone()
    if not doc:
        return jsonify({"error": f"Document #{doc_id} not found"}), 404
    content_row = hdb.conn.execute(
        "SELECT doc FROM content WHERE hash=?", (doc["hash"],)).fetchone()
    if not content_row or not content_row[0]:
        return jsonify({"error": "Document has no content"}), 404

    settings = _get_settings()
    extractor = HyperExtractor.from_settings(settings)
    hg = HyperGraph(hdb, extractor)
    result = hg.extract_and_store(doc_id, str(content_row[0]),
                                  doc["collection"] or "")
    return jsonify(result)


@hyper_api.route("/extract/batch", methods=["POST"])
def extract_batch():
    """Extract knowledge from a collection or list of doc IDs with SSE progress."""
    hdb, err = _get_hyper_db()
    if err:
        return jsonify({"error": err}), 500
    data = request.get_json() or {}
    collection = data.get("collection")
    doc_ids = data.get("doc_ids", [])

    settings = _get_settings()
    extractor = HyperExtractor.from_settings(settings)
    hg = HyperGraph(hdb, extractor)

    def _sse_event(event_type, **kwargs):
        return f"event: {event_type}\ndata: {json.dumps(kwargs, ensure_ascii=False)}\n\n"

    def generate():
        if collection:
            rows = hdb.conn.execute(
                "SELECT id, hash FROM documents WHERE collection=? AND active=1",
                (collection,)).fetchall()
            total = len(rows)
            yield _sse_event("progress", status="started", total=total, collection=collection)
            processed = 0
            errors = []
            for row in rows:
                doc_id = row[0]
                yield _sse_event("progress", status="processing", processed=processed,
                                 total=total, doc_id=doc_id)
                content_row = hdb.conn.execute(
                    "SELECT doc FROM content WHERE hash=?", (row[1],)).fetchone()
                if not content_row or not content_row[0]:
                    errors.append({"doc_id": doc_id, "error": "No content"})
                    processed += 1
                    continue
                try:
                    r = hg.extract_and_store(doc_id, str(content_row[0]), collection)
                    if r.get("error"):
                        errors.append({"doc_id": doc_id, "error": r["error"]})
                except Exception as e:
                    errors.append({"doc_id": doc_id, "error": str(e)})
                processed += 1
            yield _sse_event("complete", status="complete", processed=processed,
                             total=total, errors=len(errors))
            hdb.conn.close()
            return

        if doc_ids:
            total = len(doc_ids)
            yield _sse_event("progress", status="started", total=total)
            processed = 0
            errors = []
            for did in doc_ids:
                yield _sse_event("progress", status="processing", processed=processed,
                                 total=total, doc_id=did)
                doc = hdb.conn.execute(
                    "SELECT id, hash, collection FROM documents WHERE id=? AND active=1",
                    (did,)).fetchone()
                if not doc:
                    errors.append({"doc_id": did, "error": "Not found"})
                    processed += 1
                    continue
                content_row = hdb.conn.execute(
                    "SELECT doc FROM content WHERE hash=?",
                    (doc["hash"],)).fetchone()
                if not content_row or not content_row[0]:
                    errors.append({"doc_id": did, "error": "No content"})
                    processed += 1
                    continue
                try:
                    r = hg.extract_and_store(did, str(content_row[0]),
                                             doc["collection"] or "")
                    if r.get("error"):
                        errors.append({"doc_id": did, "error": r["error"]})
                except Exception as e:
                    errors.append({"doc_id": did, "error": str(e)})
                processed += 1
            yield _sse_event("complete", status="complete", processed=processed,
                             total=total, errors=len(errors))
            hdb.conn.close()
            return

        yield _sse_event("error", status="error", message="Provide collection or doc_ids")

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Types ──

@hyper_api.route("/types", methods=["GET"])
def list_types():
    hdb, err = _get_hyper_db()
    if err:
        return jsonify({"error": err}), 500
    ent_types = hdb.get_entity_types()
    rel_types = hdb.get_relationship_types()
    return jsonify({"entity_types": ent_types, "relationship_types": rel_types})


# ── Clear ──

@hyper_api.route("/clear", methods=["POST"])
def clear_graph():
    """Clear all hyper-extract data."""
    hdb, err = _get_hyper_db()
    if err:
        return jsonify({"error": err}), 500
    hdb.conn.execute("DELETE FROM hyper_doc_entities")
    hdb.conn.execute("DELETE FROM hyper_relationships")
    hdb.conn.execute("DELETE FROM hyper_entities")
    hdb.conn.commit()
    return jsonify({"success": True})


# ── Agent API (Hermes / tool-calling friendly) ──

@hyper_api.route("/agent/lookup", methods=["POST"])
def agent_lookup():
    """Look up an entity by name. Returns entity info, relationships, and linked documents.
    Designed for agent tool-calling (e.g. Hermes)."""
    hdb, err = _get_hyper_db()
    if err:
        return jsonify({"error": err}), 500
    data = request.get_json() or {}
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "name required"}), 400

    # Find entity by exact or fuzzy name match
    entity = hdb.conn.execute(
        "SELECT id, name, type, description, metadata FROM hyper_entities WHERE name LIKE ? LIMIT 1",
        (f"%{name}%",)
    ).fetchone()
    if not entity:
        # Try token-level matching
        tokens = [w.strip() for w in name.split() if len(w.strip()) > 1]
        for tok in tokens:
            entity = hdb.conn.execute(
                "SELECT id, name, type, description, metadata FROM hyper_entities WHERE name LIKE ? LIMIT 1",
                (f"%{tok}%",)
            ).fetchone()
            if entity:
                break
    if not entity:
        return jsonify({"found": False, "query": name})

    eid = entity["id"]
    # Get relationships
    rels = hdb.get_relationships(entity_id=eid)
    # Get linked documents
    docs = hdb.get_entity_docs(eid)
    # Get neighborhood (1 hop)
    hg = HyperGraph(hdb, _get_extractor())
    neighborhood = hg.get_entity_neighborhood(eid, depth=1)

    return jsonify({
        "found": True,
        "entity": {"id": eid, "name": entity["name"], "type": entity["type"],
                    "description": entity["description"]},
        "relationships": rels,
        "documents": docs,
        "neighborhood": neighborhood,
    })


@hyper_api.route("/agent/search", methods=["POST"])
def agent_search():
    """Search entities by keyword. Returns matching entities with their relationships and docs.
    Designed for agent tool-calling."""
    hdb, err = _get_hyper_db()
    if err:
        return jsonify({"error": err}), 500
    data = request.get_json() or {}
    query = data.get("query", "").strip()
    limit = min(int(data.get("limit", 10)), 50)
    if not query:
        return jsonify({"error": "query required"}), 400

    entities = hdb.search_entities(query, limit=limit)
    results = []
    for e in entities:
        eid = e["id"]
        rels = hdb.get_relationships(entity_id=eid, limit=20)
        docs = hdb.get_entity_docs(eid)
        results.append({
            "entity": {"id": eid, "name": e["name"], "type": e["type"],
                        "description": e.get("description", "")},
            "relationships": rels,
            "documents": docs,
        })
    return jsonify({"query": query, "count": len(results), "results": results})
