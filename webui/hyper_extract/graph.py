"""
Knowledge Graph manager — bridges HyperDB documents and the
extraction pipeline for incremental graph building.
"""

from .db import HyperDB
from .extractor import HyperExtractor


class HyperGraph:
    """High-level graph operations combining DB + extraction."""

    def __init__(self, db: HyperDB, extractor: HyperExtractor):
        self.db = db
        self.extractor = extractor

    def extract_and_store(self, doc_id: int, doc_text: str,
                          doc_collection: str = "") -> dict:
        """Extract knowledge from document text and store in graph."""
        result = self.extractor.extract(doc_text)
        entities = result.get("entities", [])
        relationships = result.get("relationships", [])
        errors = result.get("error")

        created_entities = []
        for ent in entities:
            eid = self.db.upsert_entity(
                name=ent["name"],
                typ=ent.get("type", "concept"),
                description=ent.get("description", ""),
            )
            created_entities.append({"id": eid, "name": ent["name"]})
            self.db.link_doc_entity(
                doc_id, eid,
                contexts=[ent.get("description", "")],
            )

        created_rels = []
        for rel in relationships:
            src = self.db.find_entity(rel["source"])
            tgt = self.db.find_entity(rel["target"])

            # Auto-create any missing entities
            if not src:
                eid = self.db.upsert_entity(name=rel["source"], typ="concept")
                self.db.link_doc_entity(doc_id, eid, contexts=[])
                src = self.db.find_entity(rel["source"])
            if not tgt:
                eid = self.db.upsert_entity(name=rel["target"], typ="concept")
                self.db.link_doc_entity(doc_id, eid, contexts=[])
                tgt = self.db.find_entity(rel["target"])

            if src and tgt:
                rid = self.db.upsert_relationship(
                    source_id=src["id"],
                    target_id=tgt["id"],
                    rel_type=rel.get("type", "related-to"),
                    weight=1.0,
                    context=rel.get("context", ""),
                    source_doc=f"{doc_collection}/{doc_id}",
                )
                created_rels.append({
                    "id": rid,
                    "source": rel["source"],
                    "target": rel["target"],
                    "type": rel.get("type", "related-to"),
                })

        self.db.conn.commit()
        return {
            "doc_id": doc_id,
            "entities_found": len(entities),
            "relationships_found": len(relationships),
            "entities_created": len(created_entities),
            "relationships_created": len(created_rels),
            "error": errors,
            "entities": created_entities,
            "relationships": created_rels,
        }

    def extract_collection(self, conn, collection: str,
                           callback=None) -> dict:
        """Extract knowledge from all documents in a collection."""
        rows = conn.execute(
            "SELECT id, hash FROM documents WHERE collection=? AND active=1",
            (collection,)).fetchall()
        total = len(rows)
        done = 0
        results = []
        errors = []
        for row in rows:
            doc_id = row[0]
            content_row = conn.execute(
                "SELECT doc FROM content WHERE hash=?", (row[1],)).fetchone()
            if not content_row or not content_row[0]:
                errors.append({"doc_id": doc_id,
                               "error": "No content found"})
                done += 1
                if callback:
                    callback(done, total, doc_id, None)
                continue
            try:
                r = self.extract_and_store(
                    doc_id, str(content_row[0]), collection)
                results.append(r)
            except Exception as e:
                errors.append({"doc_id": doc_id, "error": str(e)})
            done += 1
            if callback:
                callback(done, total, doc_id, r)
        return {
            "collection": collection,
            "total": total,
            "processed": len(results),
            "errors": len(errors),
            "results": results,
            "error_details": errors,
        }

    def get_entity_neighborhood(self, entity_id: int, depth: int = 1):
        """Get entities connected within N hops."""
        entity = self.db.get_entity(entity_id)
        if not entity:
            return {"entity": None, "nodes": [], "edges": []}

        nodes = {entity_id: entity}
        edges = []
        visited = {entity_id}
        current = {entity_id}

        for _ in range(depth):
            if not current:
                break
            rels = self.db.get_relationships(entity_id=list(current)[0]
                                              if len(current) == 1
                                              else None,
                                              limit=500)
            # Actually we need per-entity relationships; let's do a broader query
            next_set = set()
            placeholders = ",".join("?" for _ in current)
            rows = self.db.conn.execute(f"""
                SELECT r.*, s.name AS source_name, s.type AS source_type,
                       t.name AS target_name, t.type AS target_type
                FROM hyper_relationships r
                JOIN hyper_entities s ON s.id = r.source_id
                JOIN hyper_entities t ON t.id = r.target_id
                WHERE r.source_id IN ({placeholders})
                   OR r.target_id IN ({placeholders})
                ORDER BY r.weight DESC LIMIT 500
            """, (*current, *current)).fetchall()
            for r in rows:
                edge = {
                    "from": r["source_id"], "to": r["target_id"],
                    "type": r["rel_type"], "weight": r["weight"],
                }
                edges.append(edge)
                for eid in (r["source_id"], r["target_id"]):
                    if eid not in visited:
                        en = self.db.get_entity(eid)
                        if en:
                            nodes[eid] = en
                            next_set.add(eid)
                            visited.add(eid)
            current = next_set

        node_list = []
        for nid, ndata in nodes.items():
            node_list.append({
                "id": nid, "name": ndata["name"],
                "type": ndata["type"],
                "description": ndata["description"],
            })
        return {"entity": entity, "nodes": node_list, "edges": edges}
