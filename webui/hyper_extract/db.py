import sqlite3
import json
from datetime import datetime, timezone


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS hyper_entities (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    type        TEXT DEFAULT 'concept',
    description TEXT DEFAULT '',
    metadata    TEXT DEFAULT '{}',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS hyper_relationships (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id    INTEGER NOT NULL,
    target_id    INTEGER NOT NULL,
    rel_type     TEXT DEFAULT 'related_to',
    weight       REAL DEFAULT 1.0,
    context      TEXT DEFAULT '',
    source_doc   TEXT DEFAULT '',
    metadata     TEXT DEFAULT '{}',
    created_at   TEXT NOT NULL,
    FOREIGN KEY (source_id) REFERENCES hyper_entities(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES hyper_entities(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS hyper_doc_entities (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id     INTEGER NOT NULL,
    entity_id  INTEGER NOT NULL,
    mentions   INTEGER DEFAULT 1,
    contexts   TEXT DEFAULT '[]',
    FOREIGN KEY (entity_id) REFERENCES hyper_entities(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_hyper_rel_source ON hyper_relationships(source_id);
CREATE INDEX IF NOT EXISTS idx_hyper_rel_target ON hyper_relationships(target_id);
CREATE INDEX IF NOT EXISTS idx_hyper_de_doc   ON hyper_doc_entities(doc_id);
CREATE INDEX IF NOT EXISTS idx_hyper_de_entity ON hyper_doc_entities(entity_id);
CREATE INDEX IF NOT EXISTS idx_hyper_ent_type  ON hyper_entities(type);
"""


class HyperDB:
    """Database layer for the Hyper-Extract knowledge graph."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    def close(self):
        self.conn.close()

    # ── Entity CRUD ──

    def upsert_entity(self, name: str, typ: str = "concept",
                      description: str = "", metadata: dict = None) -> int:
        now = datetime.now(timezone.utc).isoformat()
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        cur = self.conn.execute("""
            INSERT INTO hyper_entities (name, type, description, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                type        = COALESCE(NULLIF(?,''), type),
                description = CASE WHEN ? != '' THEN ? ELSE description END,
                metadata    = CASE WHEN ? != '{}' THEN ? ELSE metadata END,
                updated_at  = ?
        """, (name, typ, description, meta_json, now, now,
              typ, description, description, meta_json, meta_json, now))
        self.conn.commit()
        return cur.lastrowid or self.conn.execute(
            "SELECT id FROM hyper_entities WHERE name=?", (name,)).fetchone()[0]

    def get_entity(self, entity_id: int):
        row = self.conn.execute(
            "SELECT * FROM hyper_entities WHERE id=?", (entity_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["metadata"] = json.loads(d.get("metadata", "{}"))
        return d

    def find_entity(self, name: str):
        row = self.conn.execute(
            "SELECT * FROM hyper_entities WHERE name=?", (name,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["metadata"] = json.loads(d.get("metadata", "{}"))
        return d

    def search_entities(self, query: str, limit: int = 50):
        rows = self.conn.execute("""
            SELECT * FROM hyper_entities
            WHERE name LIKE ? OR description LIKE ?
            ORDER BY name LIMIT ?
        """, (f"%{query}%", f"%{query}%", limit)).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["metadata"] = json.loads(d.get("metadata", "{}"))
            results.append(d)
        return results

    def list_entities(self, typ: str = None, limit: int = 200, offset: int = 0):
        if typ:
            rows = self.conn.execute(
                "SELECT * FROM hyper_entities WHERE type=? ORDER BY name LIMIT ? OFFSET ?",
                (typ, limit, offset)).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM hyper_entities ORDER BY name LIMIT ? OFFSET ?",
                (limit, offset)).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["metadata"] = json.loads(d.get("metadata", "{}"))
            results.append(d)
        return results

    def delete_entity(self, entity_id: int):
        self.conn.execute("DELETE FROM hyper_doc_entities WHERE entity_id=?", (entity_id,))
        self.conn.execute("DELETE FROM hyper_relationships WHERE source_id=? OR target_id=?",
                          (entity_id, entity_id))
        self.conn.execute("DELETE FROM hyper_entities WHERE id=?", (entity_id,))
        self.conn.commit()

    def get_entity_types(self):
        rows = self.conn.execute(
            "SELECT DISTINCT type FROM hyper_entities ORDER BY type").fetchall()
        return [r[0] for r in rows if r[0]]

    # ── Relationship CRUD ──

    def upsert_relationship(self, source_id: int, target_id: int,
                            rel_type: str = "related_to", weight: float = 1.0,
                            context: str = "", source_doc: str = "",
                            metadata: dict = None) -> int:
        now = datetime.now(timezone.utc).isoformat()
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        existing = self.conn.execute("""
            SELECT id, weight FROM hyper_relationships
            WHERE source_id=? AND target_id=? AND rel_type=?
        """, (source_id, target_id, rel_type)).fetchone()
        if existing:
            self.conn.execute("""
                UPDATE hyper_relationships SET
                    weight = MAX(weight, ?),
                    context = CASE WHEN ? != '' THEN ? ELSE context END,
                    source_doc = CASE WHEN ? != '' THEN ? ELSE source_doc END,
                    metadata = ?,
                    created_at = ?
                WHERE id=?
            """, (weight, context, context, source_doc, source_doc,
                  meta_json, now, existing[0]))
            self.conn.commit()
            return existing[0]
        else:
            cur = self.conn.execute("""
                INSERT INTO hyper_relationships
                    (source_id, target_id, rel_type, weight, context, source_doc, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (source_id, target_id, rel_type, weight, context, source_doc, meta_json, now))
            self.conn.commit()
            return cur.lastrowid

    def get_relationships(self, entity_id: int = None, limit: int = 200):
        if entity_id:
            rows = self.conn.execute("""
                SELECT r.*, s.name AS source_name, s.type AS source_type,
                       t.name AS target_name, t.type AS target_type
                FROM hyper_relationships r
                JOIN hyper_entities s ON s.id = r.source_id
                JOIN hyper_entities t ON t.id = r.target_id
                WHERE r.source_id=? OR r.target_id=?
                ORDER BY r.weight DESC LIMIT ?
            """, (entity_id, entity_id, limit)).fetchall()
        else:
            rows = self.conn.execute("""
                SELECT r.*, s.name AS source_name, s.type AS source_type,
                       t.name AS target_name, t.type AS target_type
                FROM hyper_relationships r
                JOIN hyper_entities s ON s.id = r.source_id
                JOIN hyper_entities t ON t.id = r.target_id
                ORDER BY r.weight DESC LIMIT ?
            """, (limit,)).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["metadata"] = json.loads(d.get("metadata", "{}"))
            results.append(d)
        return results

    def get_relationship_types(self):
        rows = self.conn.execute(
            "SELECT DISTINCT rel_type FROM hyper_relationships ORDER BY rel_type").fetchall()
        return [r[0] for r in rows if r[0]]

    # ── Doc-Entity Linking ──

    def link_doc_entity(self, doc_id: int, entity_id: int,
                        contexts: list = None):
        existing = self.conn.execute(
            "SELECT id, mentions, contexts FROM hyper_doc_entities WHERE doc_id=? AND entity_id=?",
            (doc_id, entity_id)).fetchone()
        ctx_json = json.dumps(contexts or [], ensure_ascii=False)
        if existing:
            self.conn.execute("""
                UPDATE hyper_doc_entities SET
                    mentions = mentions + 1,
                    contexts = ?
                WHERE id=?
            """, (ctx_json, existing[0]))
            self.conn.commit()
        else:
            self.conn.execute("""
                INSERT INTO hyper_doc_entities (doc_id, entity_id, mentions, contexts)
                VALUES (?, ?, 1, ?)
            """, (doc_id, entity_id, ctx_json))
            self.conn.commit()

    def get_doc_entities(self, doc_id: int):
        rows = self.conn.execute("""
            SELECT de.*, e.name, e.type, e.description
            FROM hyper_doc_entities de
            JOIN hyper_entities e ON e.id = de.entity_id
            WHERE de.doc_id=?
            ORDER BY de.mentions DESC
        """, (doc_id,)).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["contexts"] = json.loads(d.get("contexts", "[]"))
            results.append(d)
        return results

    def get_entity_docs(self, entity_id: int):
        rows = self.conn.execute("""
            SELECT de.*, d.title, d.path, d.collection
            FROM hyper_doc_entities de
            JOIN documents d ON d.id = de.doc_id
            WHERE de.entity_id=? AND d.active=1
            ORDER BY de.mentions DESC
        """, (entity_id,)).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["contexts"] = json.loads(d.get("contexts", "[]"))
            results.append(d)
        return results

    # ── Stats ──

    def stats(self):
        entity_count = self.conn.execute(
            "SELECT COUNT(*) FROM hyper_entities").fetchone()[0]
        rel_count = self.conn.execute(
            "SELECT COUNT(*) FROM hyper_relationships").fetchone()[0]
        link_count = self.conn.execute(
            "SELECT COUNT(*) FROM hyper_doc_entities").fetchone()[0]
        return {
            "entities": entity_count,
            "relationships": rel_count,
            "doc_links": link_count,
        }

    # ── Graph export for vis-network ──

    def export_graph(self, limit_entities: int = 300,
                     min_weight: float = 0.0):
        entities = self.conn.execute("""
            SELECT id, name, type, description FROM hyper_entities
            ORDER BY id LIMIT ?
        """, (limit_entities,)).fetchall()
        node_list = []
        ent_ids = set()
        for e in entities:
            node_list.append({
                "id": e["id"],
                "label": e["name"],
                "title": f"{e['name']} ({e['type']})",
                "group": e["type"] or "concept",
                "description": e["description"] or "",
            })
            ent_ids.add(e["id"])

        if not ent_ids:
            return {"nodes": [], "edges": []}

        placeholders = ",".join("?" for _ in ent_ids)
        rels = self.conn.execute(f"""
            SELECT r.id, r.source_id, r.target_id, r.rel_type, r.weight,
                   s.name AS source_name, t.name AS target_name
            FROM hyper_relationships r
            JOIN hyper_entities s ON s.id = r.source_id
            JOIN hyper_entities t ON t.id = r.target_id
            WHERE (r.source_id IN ({placeholders}) OR r.target_id IN ({placeholders}))
              AND r.weight >= ?
        """, (*ent_ids, *ent_ids, min_weight)).fetchall()
        edge_list = []
        for r in rels:
            edge_list.append({
                "from": r["source_id"],
                "to": r["target_id"],
                "label": r["rel_type"],
                "title": f"{r['source_name']} → {r['target_name']}: {r['rel_type']}",
                "value": r["weight"],
                "width": max(1, min(10, r["weight"] * 2)),
            })
        return {"nodes": node_list, "edges": edge_list}
