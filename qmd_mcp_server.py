"""
QMD-Extract MCP Server
Exposes QMD-Extract knowledge graph capabilities as MCP tools.
Integrates with Hermes Agent's llm-wiki skill.
Includes translation support via HY-MT15 model.
"""

import sys
import os
import time
import concurrent.futures
from pathlib import Path
from typing import Optional
import json

from mcp.server.fastmcp import FastMCP
import httpx

# Add src to path for translator import
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from translator import translate_text, translate_text_auto, detect_language, TranslationSession
except ImportError:
    translate_text = None
    translate_text_auto = None
    detect_language = None
    TranslationSession = None

# RAGFlow bridge
try:
    from ragflow_bridge import (
        list_datasets,
        create_dataset,
        search_knowledge,
        list_documents,
        delete_dataset,
        delete_document,
        upload_document,
    )
except ImportError:
    list_datasets = None
    create_dataset = None
    search_knowledge = None
    list_documents = None
    delete_dataset = None
    delete_document = None
    upload_document = None


# Create MCP server
mcp = FastMCP("qmd-extract")

# QMD-Index API base URL (configurable via environment variable)
QMD_API_BASE = os.environ.get("QMD_API_BASE", "http://localhost:8091")

# RAGFlow config
RAGFLOW_API_BASE = os.environ.get("RAGFLOW_API_BASE", "http://localhost:9393")
RAGFLOW_API_KEY = os.environ.get("RAGFLOW_API_KEY")

# Cache for RAGFlow datasets check (avoid repeated API calls)
_ragflow_datasets_cache = None
_ragflow_datasets_cache_time = 0


def _check_ragflow_datasets():
    """Check if RAGFlow has any datasets the current tenant owns. Cache result for 5 minutes."""
    global _ragflow_datasets_cache, _ragflow_datasets_cache_time
    
    # Use cache if valid (5 minutes)
    if _ragflow_datasets_cache is not None and (time.time() - _ragflow_datasets_cache_time) < 300:
        return _ragflow_datasets_cache
    
    try:
        # Step 1: Get all datasets from list API
        resp = httpx.get(
            f"{RAGFLOW_API_BASE}/api/v1/datasets",
            headers={"Authorization": f"Bearer {RAGFLOW_API_KEY}"},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        all_datasets = data.get("data", [])
        
        if not all_datasets:
            _ragflow_datasets_cache = {"has_datasets": False, "dataset_ids": []}
            _ragflow_datasets_cache_time = time.time()
            return _ragflow_datasets_cache
        
        # Step 2: Test ownership by making a lightweight retrieval call for each dataset
        owned_ids = []
        for ds in all_datasets:
            ds_id = ds.get("id")
            if not ds_id:
                continue
            try:
                test_resp = httpx.post(
                    f"{RAGFLOW_API_BASE}/api/v1/retrieval",
                    headers={"Authorization": f"Bearer {RAGFLOW_API_KEY}"},
                    json={"dataset_ids": [ds_id], "question": "test", "top_k": 1},
                    timeout=10,
                )
                test_data = test_resp.json()
                if test_data.get("code") == 0:
                    owned_ids.append(ds_id)
            except Exception:
                pass
        
        has_datasets = len(owned_ids) > 0
        _ragflow_datasets_cache = {"has_datasets": has_datasets, "dataset_ids": owned_ids}
        _ragflow_datasets_cache_time = time.time()
        return _ragflow_datasets_cache
    except Exception:
        return {"has_datasets": False, "dataset_ids": []}


def _classify_error(e: Exception, tool_name: str) -> str:
    """Classify HTTP/connection errors into structured messages."""
    err_str = str(e).lower()
    if "connect" in err_str or "refused" in err_str or "10061" in err_str:
        return json.dumps({
            "error": "connection_refused",
            "tool": tool_name,
            "message": f"Cannot reach QMD-Index at {QMD_API_BASE}. Is it running?",
            "hint": "Start QMD-Index: cd D:\\QMD-Index\\webui && python server.py --port 8090",
        })
    if "timeout" in err_str or "timed out" in err_str:
        return json.dumps({
            "error": "timeout",
            "tool": tool_name,
            "message": f"Request to QMD-Index timed out after {e.args[0] if e.args else 'N/A'}s",
        })
    if isinstance(e, httpx.HTTPStatusError):
        status = e.response.status_code
        body = ""
        try:
            body = e.response.json().get("error", "")
        except Exception:
            pass
        return json.dumps({
            "error": f"http_{status}",
            "tool": tool_name,
            "message": f"QMD-Index returned HTTP {status}: {body}",
            "hint": "Check QMD-Index logs for database or configuration issues",
        })
    if "database" in err_str:
        return json.dumps({
            "error": "database_error",
            "tool": tool_name,
            "message": f"QMD-Index database error: {body if 'body' in dir() else str(e)}",
            "hint": "Restart QMD-Index to recover the SQLite database",
        })
    return json.dumps({
        "error": "unknown",
        "tool": tool_name,
        "message": f"Unexpected error: {str(e)}",
    })


# ============================================================================
# Resources — expose knowledge graph data via MCP Resource API
# ============================================================================

@mcp.resource("hyper://status")
def resource_status() -> str:
    """Knowledge graph status and statistics."""
    try:
        response = httpx.get(f"{QMD_API_BASE}/api/hyper/status", timeout=10)
        response.raise_for_status()
        return json.dumps(response.json(), indent=2, ensure_ascii=False)
    except Exception as e:
        return _classify_error(e, "resource_status")


@mcp.resource("hyper://entities")
def resource_entities() -> str:
    """List all entities (first 50)."""
    try:
        response = httpx.get(
            f"{QMD_API_BASE}/api/hyper/entities",
            params={"limit": 50},
            timeout=10
        )
        response.raise_for_status()
        return json.dumps(response.json(), indent=2, ensure_ascii=False)
    except Exception as e:
        return _classify_error(e, "resource_entities")


@mcp.resource("hyper://entities/{entity_id}")
def resource_entity(entity_id: int) -> str:
    """Get details for a specific entity by ID."""
    try:
        response = httpx.get(
            f"{QMD_API_BASE}/api/hyper/entities/{entity_id}",
            timeout=10
        )
        response.raise_for_status()
        return json.dumps(response.json(), indent=2, ensure_ascii=False)
    except Exception as e:
        return _classify_error(e, "resource_entity")


@mcp.resource("hyper://graph")
def resource_graph() -> str:
    """Knowledge graph visualization data (top 30 nodes)."""
    try:
        response = httpx.get(
            f"{QMD_API_BASE}/api/hyper/graph",
            params={"limit": 30},
            timeout=10
        )
        response.raise_for_status()
        return json.dumps(response.json(), indent=2, ensure_ascii=False)
    except Exception as e:
        return _classify_error(e, "resource_graph")


@mcp.resource("hyper://types")
def resource_types() -> str:
    """All entity types and relationship types with counts."""
    try:
        response = httpx.get(f"{QMD_API_BASE}/api/hyper/types", timeout=10)
        response.raise_for_status()
        return json.dumps(response.json(), indent=2, ensure_ascii=False)
    except Exception as e:
        return _classify_error(e, "resource_types")


# ============================================================================
# Tool 1: qmd_status - Get knowledge graph statistics
# ============================================================================
@mcp.tool()
def qmd_status() -> str:
    """Get QMD-Extract knowledge graph statistics.
    
    Returns counts of entities, relationships, and document links.
    Use this to check the current state of the knowledge graph.
    """
    try:
        response = httpx.get(f"{QMD_API_BASE}/api/hyper/status", timeout=10)
        response.raise_for_status()
        stats = response.json()
        
        return f"""Knowledge Graph Status:
- Entities: {stats.get('entity_count', stats.get('entities', 0))}
- Relationships: {stats.get('relationship_count', stats.get('relationships', 0))}
- Document Links: {stats.get('doc_links', 0)}"""
    except Exception as e:
        return _classify_error(e, "qmd_status")


# ============================================================================
# Tool 2: qmd_search_entities - Search for entities (auto-searches both QMD + RAGFlow)
# ============================================================================
@mcp.tool()
def qmd_search_entities(query: str, limit: int = 20) -> str:
    """Search for entities in the knowledge graph AND RAGFlow knowledge base.
    
    Automatically searches both QMD knowledge graph and RAGFlow (if available).
    Searches run in parallel for speed.
    
    Args:
        query: Search query (entity name or keyword)
        limit: Maximum number of results per source (default: 20)
    
    Returns merged results from both knowledge sources.
    """
    results = {"qmd": None, "ragflow": None}
    
    def search_qmd():
        try:
            response = httpx.get(
                f"{QMD_API_BASE}/api/hyper/entities",
                params={"q": query, "limit": limit},
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            return data.get("entities", data) if isinstance(data, dict) else []
        except Exception:
            return []
    
    def search_ragflow(dataset_ids):
        try:
            response = httpx.post(
                f"{RAGFLOW_API_BASE}/api/v1/retrieval",
                headers={"Authorization": f"Bearer {RAGFLOW_API_KEY}"},
                json={"dataset_ids": dataset_ids, "question": query, "top_k": limit},
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get("code") == 100 and "embedding model" in data.get("message", "").lower():
                return [{"content": "[RAGFlow] Embedding model not configured", "document_keyword": "Config Required"}]
            
            return data.get("data", {}).get("chunks", []) if isinstance(data, dict) else []
        except Exception:
            return []
    
    # Check RAGFlow datasets
    ragflow_info = _check_ragflow_datasets()
    
    # Parallel search
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        qmd_future = executor.submit(search_qmd)
        ragflow_future = executor.submit(search_ragflow, ragflow_info["dataset_ids"]) if ragflow_info["has_datasets"] and ragflow_info["dataset_ids"] else None
        qmd_results = qmd_future.result()
        ragflow_results = ragflow_future.result() if ragflow_future else []
    
    # Format results
    lines = []
    
    if qmd_results:
        lines.append(f"=== QMD Knowledge Graph ({len(qmd_results)} entities) ===")
        for e in qmd_results[:limit]:
            name = e.get("name", "?")
            etype = e.get("type", "?")
            eid = e.get("id", "?")
            desc = e.get("description", "")[:80]
            lines.append(f"- [{eid}] {name} ({etype})")
            if desc:
                lines.append(f"  {desc}...")
    
    if ragflow_results:
        lines.append(f"\n=== RAGFlow Knowledge Base ({len(ragflow_results)} chunks) ===")
        for i, chunk in enumerate(ragflow_results[:limit], 1):
            content = chunk.get("content", "")[:100]
            doc_name = chunk.get("document_keyword", "Unknown")
            similarity = chunk.get("similarity", 0)
            lines.append(f"- [{i}] {doc_name} (sim={similarity:.3f})")
            if content:
                lines.append(f"  {content}...")
    
    if not lines:
        return f"No results found for '{query}'"
    
    total = len(qmd_results) + len(ragflow_results)
    lines.insert(0, f"Found {total} results for '{query}':\n")
    
    return "\n".join(lines)


# ============================================================================
# Tool 3: qmd_get_entity - Get entity details
# ============================================================================
@mcp.tool()
def qmd_get_entity(entity_id: int) -> str:
    """Get detailed information about a specific entity.
    
    Args:
        entity_id: The entity ID to look up
    
    Returns entity details including relationships and linked documents.
    """
    try:
        response = httpx.get(
            f"{QMD_API_BASE}/api/hyper/entities/{entity_id}",
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        entity = data.get("entity", data) if isinstance(data, dict) else None
        if not entity or (isinstance(entity, dict) and "error" in entity):
            return f"Entity #{entity_id} not found"
        
        rels = data.get("relationships", [])
        docs = data.get("documents", [])
        
        lines = [
            f"Entity: {entity.get('name', entity.get('title', 'Unknown'))}",
            f"Type: {entity.get('type', 'N/A')}",
            f"Description: {entity.get('description', 'N/A')}",
            f"\nRelationships ({len(rels)}):"
        ]
        
        for r in rels:
            src_id = r.get('source_id', 0)
            direction = "->" if src_id == entity_id else "<-"
            other = r.get('target_name', '') if src_id == entity_id else r.get('source_name', '')
            lines.append(f"  {direction} {r.get('rel_type', r.get('type', ''))} -> {other}")
        
        if docs:
            lines.append(f"\nLinked Documents ({len(docs)}):")
            for d in docs[:10]:
                lines.append(f"  - {d.get('title', d.get('path', 'Unknown'))}")
        
        return "\n".join(lines)
    except Exception as e:
        return _classify_error(e, "qmd_get_entity")


# ============================================================================
# Tool 4: qmd_search_relationships - Search for relationships
# ============================================================================
@mcp.tool()
def qmd_search_relationships(
    source_query: Optional[str] = None,
    target_query: Optional[str] = None,
    rel_type: Optional[str] = None,
    limit: int = 50
) -> str:
    """Search for relationships in the knowledge graph.
    
    Args:
        source_query: Filter by source entity name (partial match)
        target_query: Filter by target entity name (partial match)
        rel_type: Filter by relationship type (e.g., 'part-of', 'related-to')
        limit: Maximum number of results (default: 50)
    
    Returns matching relationships.
    """
    try:
        params = {"limit": limit}
        if source_query:
            params["source"] = source_query
        if target_query:
            params["target"] = target_query
        if rel_type:
            params["rel_type"] = rel_type
        
        response = httpx.get(
            f"{QMD_API_BASE}/api/hyper/relationships",
            params=params,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        rels = data.get("relationships", [])
        if not rels:
            return "No relationships found matching the criteria"
        
        lines = [f"Found {len(rels)} relationships:"]
        for r in rels:
            lines.append(f"- [{r['id']}] {r['source_name']} --[{r['rel_type']}]--> {r['target_name']} (weight: {r['weight']})")
        
        return "\n".join(lines)
    except Exception as e:
        return _classify_error(e, "qmd_search_relationships")


# ============================================================================
# Tool 5: qmd_get_graph - Export knowledge graph for visualization
# ============================================================================
@mcp.tool()
def qmd_get_graph(limit: int = 100, min_weight: float = 0.0) -> str:
    """Export knowledge graph data for visualization.
    
    Args:
        limit: Maximum number of entities to include (default: 100)
        min_weight: Minimum relationship weight (default: 0.0)
    
    Returns JSON with nodes and edges for vis-network visualization.
    Useful for building graph.html files.
    """
    try:
        response = httpx.get(
            f"{QMD_API_BASE}/api/hyper/graph",
            params={"limit": limit, "min_weight": min_weight},
            timeout=10
        )
        response.raise_for_status()
        graph_data = response.json()
        
        return json.dumps({
            "node_count": len(graph_data.get("nodes", [])),
            "edge_count": len(graph_data.get("edges", [])),
            "nodes": graph_data.get("nodes", []),
            "edges": graph_data.get("edges", [])
        }, indent=2, ensure_ascii=False)
    except Exception as e:
        return _classify_error(e, "qmd_get_graph")


# ============================================================================
# Tool 6: qmd_get_entity_neighborhood - Get entity neighborhood
# ============================================================================
@mcp.tool()
def qmd_get_entity_neighborhood(entity_id: int, depth: int = 1) -> str:
    """Get the neighborhood of an entity (entities within N hops).
    
    Args:
        entity_id: The center entity ID
        depth: Number of hops to traverse (default: 1)
    
    Returns connected entities and relationships within the specified depth.
    """
    try:
        response = httpx.get(
            f"{QMD_API_BASE}/api/hyper/neighborhood/{entity_id}",
            params={"depth": depth},
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        
        return json.dumps(result, indent=2, ensure_ascii=False)
    except Exception as e:
        return _classify_error(e, "qmd_get_entity_neighborhood")


# ============================================================================
# Tool 7: qmd_extract_from_text - Extract entities from text
# ============================================================================
@mcp.tool()
def qmd_extract_from_text(text: str, collection: str = "wiki") -> str:
    """Extract entities and relationships from text using LLM.
    
    Args:
        text: The text to extract knowledge from
        collection: Collection name (default: 'wiki')
    
    Returns extracted entities and relationships.
    Note: This calls the LLM API, may take a few seconds.
    """
    try:
        response = httpx.post(
            f"{QMD_API_BASE}/api/hyper/extract",
            json={"text": text, "collection": collection},
            timeout=300
        )
        response.raise_for_status()
        result = response.json()
        
        return json.dumps(result, indent=2, ensure_ascii=False)
    except Exception as e:
        return _classify_error(e, "qmd_extract_from_text")


# ============================================================================
# Tool 8: qmd_list_entity_types - List all entity types
# ============================================================================
@mcp.tool()
def qmd_list_entity_types() -> str:
    """List all entity types in the knowledge graph.
    
    Returns available entity types and their counts.
    """
    try:
        response = httpx.get(
            f"{QMD_API_BASE}/api/hyper/types",
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        entity_types = data.get("entity_types", [])
        if not entity_types:
            return "No entity types found"
        
        lines = [f"Entity Types ({len(entity_types)}):"]
        for t in entity_types:
            if isinstance(t, dict):
                lines.append(f"- {t.get('type', t)} ({t.get('count', '?')} entities)")
            else:
                lines.append(f"- {t}")
        
        return "\n".join(lines)
    except Exception as e:
        return _classify_error(e, "qmd_list_entity_types")


# ============================================================================
# Tool 9: qmd_list_relationship_types - List all relationship types
# ============================================================================
@mcp.tool()
def qmd_list_relationship_types() -> str:
    """List all relationship types in the knowledge graph.
    
    Returns available relationship types and their counts.
    """
    try:
        response = httpx.get(
            f"{QMD_API_BASE}/api/hyper/types",
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        rel_types = data.get("relationship_types", [])
        if not rel_types:
            return "No relationship types found"
        
        lines = [f"Relationship Types ({len(rel_types)}):"]
        for t in rel_types:
            if isinstance(t, dict):
                lines.append(f"- {t.get('type', t)} ({t.get('count', '?')} relationships)")
            else:
                lines.append(f"- {t}")
        
        return "\n".join(lines)
    except Exception as e:
        return _classify_error(e, "qmd_list_relationship_types")


# ============================================================================
# Tool 10: qmd_start_minio - Start MinIO for image access
# ============================================================================
@mcp.tool()
def qmd_start_minio() -> str:
    """Start MinIO service for accessing images in the knowledge graph.
    
    MinIO stores images referenced by markdown files.
    This tool checks if MinIO is running and starts it if needed.
    """
    import subprocess
    import sys as _sys
    
    MINIO_EXE = Path(os.environ.get("MINIO_EXE_PATH", Path.home() / "AppData/Local/Microsoft/WinGet/Packages/MinIO.Server_Microsoft.Winget.Source_8wekyb3d8bbwe/minio.exe"))
    MINIO_DATA = Path(os.environ.get("MINIO_DATA_DIR", "D:/QMD-Index/data/minio"))
    MINIO_PORT = int(os.environ.get("MINIO_PORT", 9000))
    
    try:
        # Check if MinIO is already running
        try:
            _resp = httpx.get(f"http://localhost:{MINIO_PORT}/minio/health/live", timeout=2)
            if _resp.status_code == 200:
                return f"MinIO is already running on port {MINIO_PORT}"
        except Exception:
            pass
        
        if not MINIO_EXE.exists():
            return f"MinIO not found at: {MINIO_EXE}"
        
        MINIO_DATA.mkdir(parents=True, exist_ok=True)
        
        subprocess.Popen(
            [str(MINIO_EXE), "server", str(MINIO_DATA)],
            creationflags=subprocess.CREATE_NO_WINDOW if _sys.platform == "win32" else 0
        )
        
        for _ in range(10):
            time.sleep(1)
            try:
                _resp = httpx.get(f"http://localhost:{MINIO_PORT}/minio/health/live", timeout=2)
                if _resp.status_code == 200:
                    return f"MinIO started successfully on port {MINIO_PORT}"
            except Exception:
                pass
        
        return "MinIO startup timeout"
    except Exception as e:
        return _classify_error(e, "qmd_start_minio")


# ============================================================================
# Tool 11: qmd_translate - Translate text using HY-MT15
# ============================================================================
@mcp.tool()
def qmd_translate(text: str, target_lang: str = "zh", source_lang: str = "auto") -> str:
    """Translate text between languages using HY-MT15 model.
    
    Args:
        text: Text to translate
        target_lang: Target language code (zh/en/es/fr/de/ja/ko/it/pt/ru/ar)
        source_lang: Source language code or 'auto' for auto-detection
    
    Returns translated text with language detection info.
    """
    try:
        detected = detect_language(text)
        translated = translate_text_auto(text, target_lang=target_lang, source_lang=source_lang)
        return json.dumps({
            "original": text,
            "translated": translated,
            "source_lang": detected,
            "target_lang": target_lang,
        }, ensure_ascii=False)
    except Exception as e:
        return _classify_error(e, "qmd_translate")


# ============================================================================
# Tool 12: qmd_search_translated - Search with auto-translation
# ============================================================================
@mcp.tool()
def qmd_search_translated(query: str, target_lang: str = "zh", limit: int = 20) -> str:
    """Search entities and translate results to target language.
    
    Searches the knowledge graph and translates non-matching language results
    to the specified target language using HY-MT15 model.
    
    Args:
        query: Search query
        target_lang: Language to translate results into (default: zh)
        limit: Maximum number of results (default: 20)
    
    Returns entities with translated descriptions.
    """
    if not translate_text or not TranslationSession:
        return json.dumps({"error": "Translation module not available. Check translator.py"})
    
    try:
        response = httpx.get(
            f"{QMD_API_BASE}/api/hyper/entities",
            params={"q": query, "limit": limit},
            timeout=15
        )
        response.raise_for_status()
        data = response.json()
        
        entities = data.get("entities", data) if isinstance(data, dict) else []
        if not entities:
            return f"No entities found matching '{query}'"
        
        # Translate descriptions
        translated_entities = []
        with TranslationSession():
            for e in entities:
                desc = e.get("description", "")
                if desc and detect_language(desc) != target_lang:
                    try:
                        desc = translate_text(desc, target_lang=target_lang)
                    except Exception:
                        pass
            translated_entities.append({
                "id": e.get("id"),
                "name": e.get("name", ""),
                "type": e.get("type", ""),
                "description": desc,
            })
        
        lines = [f"Found {len(translated_entities)} entities (translated to {target_lang}):"]
        for e in translated_entities:
            lines.append(f"- [{e['id']}] {e['name']} ({e['type']})")
            if e["description"]:
                lines.append(f"  {e['description'][:100]}...")
        
        return "\n".join(lines)
    except Exception as e:
        return _classify_error(e, "qmd_search_translated")


# ============================================================================
# Tool 13: qmd_ragflow_datasets - List RAGFlow datasets
# ============================================================================
@mcp.tool()
def qmd_ragflow_datasets(page: int = 1, page_size: int = 30) -> str:
    """List all datasets in RAGFlow.
    
    Args:
        page: Page number (default: 1)
        page_size: Items per page (default: 30)
    
    Returns list of datasets with their IDs and names.
    """
    try:
        result = list_datasets(page=page, page_size=page_size)
        datasets = result.get("data", [])
        if not datasets:
            return "No datasets found in RAGFlow"
        
        lines = [f"RAGFlow Datasets ({len(datasets)}):"]
        for d in datasets:
            lines.append(f"- [{d.get('id', '?')}] {d.get('name', 'Unknown')}")
        
        return "\n".join(lines)
    except Exception as e:
        return _classify_error(e, "qmd_ragflow_datasets")


# ============================================================================
# Tool 14: qmd_ragflow_search - Search RAGFlow knowledge base
# ============================================================================
@mcp.tool()
def qmd_ragflow_search(dataset_ids: list, question: str, top_k: int = 10) -> str:
    """Search RAGFlow knowledge base using RAG retrieval.
    
    Args:
        dataset_ids: List of dataset IDs to search
        question: Search question
        top_k: Number of results to return (default: 10)
    
    Returns retrieved chunks with similarity scores.
    """
    try:
        result = search_knowledge(
            dataset_ids=dataset_ids,
            question=question,
            top_k=top_k,
        )
        return json.dumps(result, indent=2, ensure_ascii=False)
    except Exception as e:
        return _classify_error(e, "qmd_ragflow_search")


# ============================================================================
# Tool 15: qmd_unified_search - Search both QMD-Index and RAGFlow
# ============================================================================
# Tool 15: qmd_ragflow_create_dataset - Create RAGFlow dataset
# ============================================================================
@mcp.tool()
def qmd_ragflow_create_dataset(name: str, description: str = "") -> str:
    """Create a new dataset in RAGFlow.
    
    Args:
        name: Dataset name (must be unique)
        description: Optional description
    
    Returns dataset ID on success.
    """
    if not create_dataset:
        return json.dumps({"error": "RAGFlow bridge not available. Check ragflow_bridge.py"})
    
    try:
        result = create_dataset(name=name, description=description)
        dataset_id = result.get("data", {}).get("id", "unknown")
        return json.dumps({
            "status": "created",
            "dataset_id": dataset_id,
            "name": name,
            "message": f"Dataset '{name}' created with ID: {dataset_id}",
        }, ensure_ascii=False)
    except Exception as e:
        return _classify_error(e, "qmd_ragflow_create_dataset")


# ============================================================================
# Tool 16: qmd_ragflow_upload - Upload document to RAGFlow
# ============================================================================
@mcp.tool()
def qmd_ragflow_upload(dataset_id: str, file_path: str, chunk_method: str = "naive") -> str:
    """Upload a document to a RAGFlow dataset.
    
    Args:
        dataset_id: Target dataset ID
        file_path: Local file path to upload (markdown, PDF, etc.)
        chunk_method: Chunking method (naive, book, paper, etc.)
    
    Returns document ID on success.
    """
    if not upload_document:
        return json.dumps({"error": "RAGFlow bridge not available. Check ragflow_bridge.py"})
    
    try:
        import os as _os
        if not _os.path.exists(file_path):
            return json.dumps({"error": "file_not_found", "path": file_path})
        
        result = upload_document(dataset_id=dataset_id, file_path=file_path, chunk_method=chunk_method)
        doc_id = result.get("data", {}).get("id", "unknown")
        return json.dumps({
            "status": "uploaded",
            "document_id": doc_id,
            "file": file_path.split("/")[-1].split("\\")[-1],
            "dataset_id": dataset_id,
        }, ensure_ascii=False)
    except Exception as e:
        return _classify_error(e, "qmd_ragflow_upload")


# ============================================================================
# Tool 17: qmd_ragflow_delete_document - Delete RAGFlow document
# ============================================================================
@mcp.tool()
def qmd_ragflow_delete_document(dataset_id: str, document_id: str) -> str:
    """Delete a document from a RAGFlow dataset.
    
    Args:
        dataset_id: Dataset ID
        document_id: Document ID to delete
    
    Returns deletion status.
    """
    if not delete_document:
        return json.dumps({"error": "RAGFlow bridge not available. Check ragflow_bridge.py"})
    
    try:
        delete_document(dataset_id=dataset_id, document_id=document_id)
        return json.dumps({
            "status": "deleted",
            "dataset_id": dataset_id,
            "document_id": document_id,
        }, ensure_ascii=False)
    except Exception as e:
        return _classify_error(e, "qmd_ragflow_delete_document")


# ============================================================================
# Run the server
# ============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", default="stdio", choices=["stdio", "sse", "streamable-http"])
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    mcp.settings.port = args.port
    mcp.settings.host = "0.0.0.0"
    mcp.run(transport=args.transport)
