"""
QMD MCP Server - Exposes QMD's vector search as MCP tools.
Uses the same SQLite index as qmd.py.
"""
import sqlite3, os, json, hashlib, urllib.request, urllib.parse, re, math
from mcp.server.fastmcp import FastMCP

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "models", "qmd", "index.sqlite")
VEC0_DLL = os.path.join(SCRIPT_DIR, "node_modules", "sqlite-vec-windows-x64", "vec0.dll")
EMBEDDING_URL = 'http://127.0.0.1:1278/v1/embeddings'
TAVILY_API_URL = 'https://api.tavily.com/search'
SERPAPI_URL = 'https://serpapi.com/search.json'

mcp = FastMCP("QMD Search", log_level="WARNING")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    try:
        conn.load_extension(VEC0_DLL)
    except Exception:
        pass
    return conn

def get_embedding(text: str):
    data = json.dumps({"model": "Qwen3-Embedding-0.6B-f16.gguf", "input": text, "encoding_format": "float"}).encode()
    req = urllib.request.Request(EMBEDDING_URL, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            result = json.loads(r.read())
            return result['data'][0]['embedding']
    except Exception as e:
        return None

def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0

def has_cjk(text):
    return bool(re.search(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]', text))

def clean_fts_query(query: str) -> str:
    """Prepare Chinese queries for FTS5 by removing spaces between CJK chars."""
    if has_cjk(query):
        query = re.sub(r'\s+', '', query)
    return query.strip()

def search_tavily(query: str, limit: int = 5):
    api_key = (os.environ.get("TAVILY_API_KEY") or "").strip()
    if not api_key:
        return None
    body = json.dumps({"api_key": api_key, "query": query, "search_depth": "basic", "max_results": limit, "include_answer": True}).encode()
    req = urllib.request.Request(TAVILY_API_URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
            results = [{"title": item.get("title",""), "url": item.get("url",""), "content": item.get("content","")[:500], "source": "tavily"} for item in data.get("results",[])]
            if data.get("answer"):
                results.insert(0, {"title": "AI Answer", "url": "", "content": data["answer"][:1000], "source": "tavily"})
            return results
    except Exception as e:
        return None

def search_serpapi(query: str, limit: int = 5):
    api_key = (os.environ.get("SERPAPI_API_KEY") or "").strip()
    if not api_key:
        return None
    params = urllib.parse.urlencode({"api_key": api_key, "q": query, "num": limit, "engine": "google"})
    try:
        with urllib.request.urlopen(f"{SERPAPI_URL}?{params}", timeout=15) as r:
            data = json.loads(r.read())
            results = [{"title": item.get("title",""), "url": item.get("link",""), "content": item.get("snippet","")[:500], "source": "serpapi"} for item in data.get("organic_results",[])]
            return results
    except Exception as e:
        return None

def fts_search(cur, query: str, collection: str = "", limit: int = 5):
    """Run FTS5 BM25 search."""
    try:
        if collection:
            cur.execute('''
                SELECT d.id, d.title, d.path, d.collection, d.hash
                FROM documents_fts f JOIN documents d ON d.id = f.rowid
                WHERE documents_fts MATCH ? AND d.active = 1 AND d.collection = ?
                ORDER BY rank LIMIT ?
            ''', (query, collection, limit))
        else:
            cur.execute('''
                SELECT d.id, d.title, d.path, d.collection, d.hash
                FROM documents_fts f JOIN documents d ON d.id = f.rowid
                WHERE documents_fts MATCH ? AND d.active = 1
                ORDER BY rank LIMIT ?
            ''', (query, limit))
        return [{"title": r["title"], "path": r["path"], "collection": r["collection"], "score": 0.0, "source": "qmd"} for r in cur.fetchall()]
    except Exception as e:
        return None

def embedding_search(query: str, collection: str = "", limit: int = 5):
    """Semantic search using vec0 ANN on stored 1024-dim embeddings (fast).
    Queries vec0 first, then resolves document details via a second query (vec0 + JOIN hangs).
    """
    qvec = get_embedding(query)
    if not qvec:
        return None
    db = get_db()
    cur = db.cursor()
    try:
        # Step 1: vec0 ANN (no JOIN to avoid hang)
        cur.execute('SELECT hash_seq, distance FROM vectors_vec WHERE embedding MATCH ? AND k=?',
            (json.dumps(qvec), limit * 5))
        vec_rows = cur.fetchall()
    except Exception as e:
        vec_rows = []
    if not vec_rows:
        db.close()
        return None

    # Step 2: resolve hash_seq -> document details
    scored = []
    seen = set()
    for vr in vec_rows:
        hash_seq = vr["hash_seq"]
        dist = vr["distance"]
        parts = hash_seq.rsplit('_', 1)
        if len(parts) != 2:
            continue
        content_hash = parts[0]
        cur.execute(
            'SELECT d.id, d.title, d.path, d.collection FROM documents d WHERE d.hash=? AND d.active=1',
            (content_hash,))
        doc = cur.fetchone()
        if doc and doc["id"] not in seen:
            seen.add(doc["id"])
            scored.append({
                "title": doc["title"], "path": doc["path"],
                "collection": doc["collection"], "score": round(1 - dist, 4),
                "source": "qmd"
            })
            if len(scored) >= limit:
                break
    db.close()
    return scored if scored else None

def search_qmd(query: str, collection: str = "", limit: int = 5):
    """Multi-pass search: FTS5 BM25 → embedding semantic → empty."""
    # Pass 1: Clean FTS5 query (strip spaces for CJK)
    clean_q = clean_fts_query(query)
    results = _fts_with_previews(clean_q, collection, limit)
    if results:
        return results

    # Pass 2: Try broader FTS5 with individual CJK chars
    if has_cjk(query):
        chars = " AND ".join(re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]', clean_q))
        if chars:
            results = _fts_with_previews(chars, collection, limit)
            if results:
                return results

    # Pass 3: Embedding-based semantic search (slow but catches sem gap)
    results = embedding_search(query, collection, limit)
    if results:
        return results

    return []

def _fts_with_previews(query: str, collection: str = "", limit: int = 5):
    """FTS5 search + add previews, returns list or []."""
    db = get_db()
    cur = db.cursor()
    results = fts_search(cur, query, collection, limit)
    if not results:
        db.close()
        return []
    for r in results:
        cur.execute('SELECT SUBSTR(doc, 1, 300) as preview FROM content WHERE hash = (SELECT hash FROM documents WHERE id = ?)', (r["title"],))
        p = cur.fetchone()
        r["preview"] = p["preview"] if p else ""
    db.close()
    return results[:limit]

@mcp.tool()
def query(query: str, collection: str = "", limit: int = 5) -> str:
    """
Search QMD index using BM25 full-text search.
For Chinese queries, spaces are auto-removed for better matching.
Results are returned as JSON array with title, path, collection, score, and preview.
For semantic search with embedding, use search_priority which auto-falls back to web.
"""
    q = clean_fts_query(query)
    db = get_db()
    cur = db.cursor()
    if collection:
        cur.execute('''
            SELECT d.id, d.title, d.path, d.collection, d.hash
            FROM documents_fts f JOIN documents d ON d.id = f.rowid
            WHERE documents_fts MATCH ? AND d.active = 1 AND d.collection = ?
            ORDER BY rank LIMIT ?
        ''', (q, collection, limit))
    else:
        cur.execute('''
            SELECT d.id, d.title, d.path, d.collection, d.hash
            FROM documents_fts f JOIN documents d ON d.id = f.rowid
            WHERE documents_fts MATCH ? AND d.active = 1
            ORDER BY rank LIMIT ?
        ''', (q, limit))
    rows = cur.fetchall()
    results = [{"title": r["title"], "path": r["path"], "collection": r["collection"], "score": 0.0} for r in rows]
    for r in results:
        cur.execute('SELECT SUBSTR(doc, 1, 300) as preview FROM content WHERE hash = (SELECT hash FROM documents WHERE id = ?)', (r["title"],))
        p = cur.fetchone()
        r["preview"] = p["preview"] if p else ""
    db.close()
    return json.dumps(results, ensure_ascii=False, indent=2)

@mcp.tool()
def get(path: str, lines: int = 0) -> str:
    """
Retrieve full document content by path or document ID.
Use lines > 0 to get only first N lines.
Returns JSON with id, title, path, collection, content, and metadata.
"""
    db = get_db()
    cur = db.cursor()
    cur.execute('SELECT d.*, c.doc as content FROM documents d LEFT JOIN content c ON d.hash = c.hash WHERE d.id = ? AND d.active = 1', (path,))
    row = cur.fetchone()
    if not row:
        cur.execute('SELECT d.*, c.doc as content FROM documents d LEFT JOIN content c ON d.hash = c.hash WHERE d.path LIKE ? AND d.active = 1 LIMIT 1', (f'%{path}%',))
        row = cur.fetchone()
    if not row:
        db.close()
        return json.dumps({"error": "not found"})
    content = row["content"] or ""
    if lines > 0:
        content = "\n".join(content.split("\n")[:lines])
    result = {"id": row["id"], "title": row["title"], "path": row["path"], "collection": row["collection"], "content": content}
    db.close()
    return json.dumps(result, ensure_ascii=False, indent=2)

@mcp.tool()
def list_collections() -> str:
    """List all collections with document count and status."""
    db = get_db()
    cur = db.cursor()
    cur.execute('''
        SELECT collection as name, COUNT(*) as doc_count
        FROM documents WHERE active=1
        GROUP BY collection ORDER BY doc_count DESC
    ''')
    rows = cur.fetchall()
    db.close()
    return json.dumps([{"name": r["name"], "docs": r["doc_count"]} for r in rows], ensure_ascii=False, indent=2)

@mcp.tool()
def search_priority(query: str, collection: str = "", limit: int = 5) -> str:
    """
Priority search: searches QMD local knowledge base FIRST.
Three passes: 1) BM25 fulltext  2) Embedding semantic (catches synonyms)  3) Web fallback
QMD results show source='qmd'. Web results show source='web'/'tavily'/'serpapi'.
Returns JSON with source, results array (title, content/preview, url/path, score).
"""
    qmd_results = search_qmd(query, collection, limit)
    if qmd_results:
        return json.dumps({"source": "qmd", "results": qmd_results}, ensure_ascii=False, indent=2)

    web_results = search_tavily(query, limit)
    if web_results is None:
        web_results = search_serpapi(query, limit)
    if web_results:
        return json.dumps({"source": "web", "engine": web_results[0].get("source", "unknown"), "results": web_results}, ensure_ascii=False, indent=2)

    return json.dumps({"source": "none", "results": [], "message": "No results found from QMD or web search."}, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    mcp.run(transport="stdio")
