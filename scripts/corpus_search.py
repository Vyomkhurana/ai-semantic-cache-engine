import numpy as np
import faiss


def search_corpus(query_emb, corpus_index, doc_store, top_k=5):
    """
    query_emb: normalized float32 vector (1d)
    returns list of matching docs with scores
    """
    q = query_emb.reshape(1, -1).astype(np.float32)
    scores, indices = corpus_index.search(q, k=top_k)

    docs = doc_store["docs"]
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        doc = docs[idx]
        results.append({
            "score": float(score),
            "category": doc["category"],
            "text": doc["text"][:500]
        })
    return results


def format_result(docs):
    if not docs:
        return "no relevant documents found."
    top = docs[0]
    return (
        f"[{top['category']}] (score={top['score']:.3f})\n"
        f"{top['text']}"
    )
