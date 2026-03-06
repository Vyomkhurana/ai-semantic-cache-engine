import numpy as np
import faiss
import os
import sys
from collections import defaultdict
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EMBED_MODEL, EMBED_DIM, SIMILARITY_THRESHOLD


class SemanticCache:
    def __init__(self, threshold=SIMILARITY_THRESHOLD, use_cluster_routing=True):
        self.threshold = threshold
        self.use_cluster_routing = use_cluster_routing
        self.model = SentenceTransformer(EMBED_MODEL)
        self.index = faiss.IndexFlatIP(EMBED_DIM)
        self.entries = []
        self.hit_count = 0
        self.miss_count = 0
        self.cluster_buckets = defaultdict(list)
        self._gmm_artifacts = None
        self._cluster_meta = None

    def load_corpus_resources(self, corpus_index, doc_store, gmm_artifacts, cluster_meta):
        self._corpus_index = corpus_index
        self._corpus_store = doc_store
        self._gmm_artifacts = gmm_artifacts
        self._cluster_meta = cluster_meta

    def _get_dominant_cluster(self, embedding):
        if self._gmm_artifacts is None:
            return None, None
        gmm = self._gmm_artifacts["gmm"]
        pca = self._gmm_artifacts["pca"]
        reduced = pca.transform(embedding.reshape(1, -1))
        probs = gmm.predict_proba(reduced)[0]
        return int(np.argmax(probs)), probs

    def _embed(self, text):
        emb = self.model.encode([text], normalize_embeddings=True)
        return emb[0].astype(np.float32)

    def _candidate_indices(self, cluster_id):
        # only route by cluster once cache is big enough for it to matter
        if self.use_cluster_routing and cluster_id is not None and len(self.entries) > 50:
            candidates = self.cluster_buckets.get(cluster_id, [])
            return candidates if candidates else list(range(len(self.entries)))
        return list(range(len(self.entries)))

    def lookup(self, query):
        if len(self.entries) == 0:
            return False, None, 0.0

        query_emb = self._embed(query)
        cluster_id, _ = self._get_dominant_cluster(query_emb) if self._gmm_artifacts else (None, None)
        candidates = self._candidate_indices(cluster_id)

        if not candidates:
            return False, None, 0.0

        cand_embs = np.stack([self.entries[i]["embedding"] for i in candidates])
        mini_index = faiss.IndexFlatIP(EMBED_DIM)
        mini_index.add(cand_embs)

        scores, positions = mini_index.search(query_emb.reshape(1, -1), k=1)
        best_score = float(scores[0][0])
        best_entry = self.entries[candidates[positions[0][0]]]

        if best_score >= self.threshold:
            self.hit_count += 1
            return True, best_entry, best_score
        return False, None, best_score

    def store(self, query, result, cluster_id=None):
        emb = self._embed(query)
        if cluster_id is None and self._gmm_artifacts:
            cluster_id, _ = self._get_dominant_cluster(emb)

        entry = {
            "query": query,
            "embedding": emb,
            "result": result,
            "dominant_cluster": cluster_id
        }
        entry_idx = len(self.entries)
        self.entries.append(entry)
        self.index.add(emb.reshape(1, -1))

        if cluster_id is not None:
            self.cluster_buckets[cluster_id].append(entry_idx)

        self.miss_count += 1
        return entry

    def flush(self):
        self.index = faiss.IndexFlatIP(EMBED_DIM)
        self.entries = []
        self.cluster_buckets = defaultdict(list)
        self.hit_count = 0
        self.miss_count = 0

    def stats(self):
        total = self.hit_count + self.miss_count
        return {
            "total_entries": len(self.entries),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": round(self.hit_count / total, 4) if total > 0 else 0.0
        }
