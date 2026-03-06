import os
import sys
import pickle
import numpy as np
import faiss

from fastapi import FastAPI, HTTPException
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


app = FastAPI(title="semantic cache engine", lifespan=lifespan)


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
