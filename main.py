import os
import sys
import pickle
import numpy as np
import faiss

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cache.semantic_cache import SemanticCache
from scripts.corpus_search import search_corpus, format_result
from config import FAISS_INDEX_PATH, DOC_STORE_PATH, GMM_ARTIFACTS_PATH, CLUSTER_META_PATH


class AppState:
    cache: SemanticCache = None
    corpus_index = None
    doc_store = None
    gmm_artifacts = None
    cluster_meta = None


state = AppState()


def load_artifacts():
    if not os.path.exists(FAISS_INDEX_PATH):
        raise RuntimeError("faiss index not found. run preprocess + build_index first.")

    state.corpus_index = faiss.read_index(FAISS_INDEX_PATH)

    with open(DOC_STORE_PATH, "rb") as f:
        state.doc_store = pickle.load(f)

    if os.path.exists(GMM_ARTIFACTS_PATH):
        with open(GMM_ARTIFACTS_PATH, "rb") as f:
            state.gmm_artifacts = pickle.load(f)

    if os.path.exists(CLUSTER_META_PATH):
        with open(CLUSTER_META_PATH, "rb") as f:
            state.cluster_meta = pickle.load(f)

    state.cache = SemanticCache()
    if state.gmm_artifacts:
        state.cache.load_corpus_resources(
            state.corpus_index,
            state.doc_store,
            state.gmm_artifacts,
            state.cluster_meta
        )
    print("artifacts loaded")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="Semantic Cache Engine",
    description="Vector similarity search cache built with FAISS and sentence-transformers. Queries are routed through GMM cluster assignments for fast semantic lookup.",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan
)


_DOCS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Semantic Cache Engine</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css" />
  <style>
    *, *::before, *::after { box-sizing: border-box; }
    body {
      margin: 0;
      background: #f1f5f9;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .site-header {
      background: #1e293b;
      padding: 16px 28px;
      display: flex;
      align-items: center;
      gap: 14px;
      border-bottom: 3px solid #4f46e5;
    }
    .site-header h1 {
      color: #f8fafc;
      font-size: 17px;
      font-weight: 600;
      margin: 0;
      letter-spacing: -0.2px;
    }
    .site-header .pill {
      background: #4f46e5;
      color: #fff;
      font-size: 10px;
      font-weight: 700;
      padding: 2px 8px;
      border-radius: 99px;
      letter-spacing: 0.6px;
      text-transform: uppercase;
    }
    .site-header .sub {
      color: #94a3b8;
      font-size: 12px;
      margin-left: auto;
    }
    #swagger-ui {
      max-width: 980px;
      margin: 0 auto;
      padding: 28px 20px 80px;
    }
    .swagger-ui .topbar { display: none !important; }
    .swagger-ui .info .version,
    .swagger-ui .info small,
    .swagger-ui .info .version-stamp { display: none !important; }
    .swagger-ui .info { margin: 0 0 24px; }
    .swagger-ui .info .title { display: none !important; }
    .swagger-ui .info .description p {
      color: #475569;
      font-size: 14px;
      margin: 0;
      line-height: 1.6;
    }
    .swagger-ui .opblock-tag {
      font-size: 14px;
      font-weight: 600;
      color: #1e293b;
      border-bottom: 1px solid #e2e8f0;
      padding-bottom: 6px;
    }
    .swagger-ui .opblock {
      border-radius: 7px;
      margin-bottom: 10px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .swagger-ui .opblock-summary-method {
      border-radius: 4px;
      font-size: 12px;
      font-weight: 700;
      min-width: 62px;
    }
    .swagger-ui .opblock.opblock-post {
      border-color: #4f46e5;
      background: rgba(79,70,229,0.04);
    }
    .swagger-ui .opblock.opblock-post .opblock-summary { border-color: #4f46e5; }
    .swagger-ui .opblock.opblock-post .opblock-summary-method { background: #4f46e5; }
    .swagger-ui .opblock.opblock-get {
      border-color: #0284c7;
      background: rgba(2,132,199,0.04);
    }
    .swagger-ui .opblock.opblock-get .opblock-summary-method { background: #0284c7; }
    .swagger-ui .opblock.opblock-delete {
      border-color: #dc2626;
      background: rgba(220,38,38,0.04);
    }
    .swagger-ui .opblock.opblock-delete .opblock-summary-method { background: #dc2626; }
    .swagger-ui .btn.execute {
      background: #4f46e5;
      border-color: #4f46e5;
      border-radius: 5px;
      font-weight: 600;
    }
    .swagger-ui .btn.execute:hover { background: #4338ca; }
    .swagger-ui .btn.try-out__btn {
      border-radius: 5px;
      border-color: #94a3b8;
      color: #64748b;
    }
    .swagger-ui textarea,
    .swagger-ui input[type=text] {
      border-radius: 5px !important;
      border: 1px solid #cbd5e1 !important;
      font-family: 'SF Mono', 'Fira Code', monospace;
      font-size: 13px;
    }
    .swagger-ui .response-col_status { font-weight: 700; }
    .swagger-ui .highlight-code { border-radius: 6px; }
    .swagger-ui section.models { display: none !important; }
  </style>
</head>
<body>
  <div class="site-header">
    <h1>Semantic Cache Engine</h1>
    <span class="pill">API</span>
    <span class="sub">FAISS &middot; sentence-transformers &middot; GMM</span>
  </div>
  <div id="swagger-ui"></div>
  <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    SwaggerUIBundle({
      url: "/openapi.json",
      dom_id: "#swagger-ui",
      presets: [SwaggerUIBundle.presets.apis, SwaggerUIBundle.SwaggerUIStandalonePreset],
      layout: "BaseLayout",
      deepLinking: true,
      defaultModelsExpandDepth: -1,
      docExpansion: "list"
    })
  </script>
</body>
</html>
"""


@app.get("/docs", include_in_schema=False)
async def custom_docs():
    return HTMLResponse(content=_DOCS_HTML)


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
def handle_query(body: QueryRequest):
    q = body.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="query cannot be empty")

    cache = state.cache
    hit, entry, score = cache.lookup(q)

    if hit:
        return {
            "query": q,
            "cache_hit": True,
            "matched_query": entry["query"],
            "similarity_score": round(score, 4),
            "result": entry["result"],
            "dominant_cluster": entry["dominant_cluster"]
        }

    query_emb = cache._embed(q)
    docs = search_corpus(query_emb, state.corpus_index, state.doc_store)
    result_text = format_result(docs)

    cluster_id = None
    if state.gmm_artifacts:
        cluster_id, _ = cache._get_dominant_cluster(query_emb)

    stored = cache.store(q, result_text, cluster_id=cluster_id)

    return {
        "query": q,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": round(score, 4),
        "result": result_text,
        "dominant_cluster": stored["dominant_cluster"]
    }


@app.get("/cache/stats")
def cache_stats():
    return state.cache.stats()


@app.delete("/cache")
def flush_cache():
    state.cache.flush()
    return {"status": "cache cleared"}
