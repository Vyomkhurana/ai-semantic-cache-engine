# ai-semantic-cache-engine

semantic search system over the 20 Newsgroups dataset, built for the Trademarkia AI/ML Engineer task.  
covers fuzzy clustering, a custom semantic cache, and a FastAPI service.

---

## setup (local)

```bash
python -m venv venv
venv\Scripts\activate        # windows
# source venv/bin/activate   # mac/linux

pip install -r requirements.txt
```

## build the pipeline (run once, in order)

```bash
python scripts/preprocess.py       # cleans corpus, saves data/corpus.pkl
python scripts/build_index.py      # builds FAISS index + doc store
python clustering/fuzzy_cluster.py # fits GMM, saves cluster artifacts
```

first run downloads the dataset (~14MB) and the embedding model (~90MB) automatically.

## start the API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

interactive docs at `http://localhost:8000/docs`

## run with Docker (recommended)

```bash
docker-compose up
```

builds the image, runs the pipeline if artifacts are missing, starts the server on port 8000.  
no Python setup needed.

---

## endpoints

**POST /query**
```json
{ "query": "how does encryption work" }
```
returns:
```json
{
  "query": "how does encryption work",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": 0.0,
  "result": "...",
  "dominant_cluster": 4
}
```
on a second call with a paraphrase like `"explain cryptography basics"`, returns `cache_hit: true`.

**GET /cache/stats**
```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405
}
```

**DELETE /cache** — flushes cache and resets all counters

---

## design decisions

**embedding model: `all-MiniLM-L6-v2`**  
384-dim, runs fully locally, fast enough to encode 8k docs in ~3 minutes.  
good semantic quality for news-style text without needing a GPU.

**vector store: FAISS `IndexFlatIP`**  
exact cosine search (vectors are L2-normalised, so inner product = cosine similarity).  
at 8-15k docs, approximate search adds complexity without meaningful speed gain.

**clustering: Gaussian Mixture Model, 15 components**  
GMM gives soft probability assignments — a doc about gun legislation gets e.g. 55% politics, 38% firearms, 7% misc.  
hard labels (k-means) would lie about that ambiguity.  
15 clusters chosen via BIC sweep — run `select_n_clusters()` in `fuzzy_cluster.py` to see the curve.  
the 20 original labels are editorial, not semantic — some (mac.hardware vs ibm.hardware) are nearly identical in embedding space.

**semantic cache threshold: 0.65 cosine similarity**  
explored in `scripts/threshold_analysis.py` — precision stays 1.0 across all thresholds (embedding space cleanly separates related/unrelated queries). recall drops hard above 0.75. 0.65 gives the best F1 (0.947).  
see `scripts/threshold_analysis.png` for the precision/recall curve.

**cluster-aware cache lookup**  
when cache has >50 entries, lookup only searches within the same dominant cluster.  
this keeps lookup cost O(cluster_size) instead of O(cache_size) as the cache grows.

---

## run the cluster analysis

```bash
python scripts/cluster_analysis.py
```

prints a full breakdown of each cluster, top categories, and boundary documents.

## run threshold exploration

```bash
python scripts/threshold_analysis.py
```

sweeps thresholds from 0.65 to 0.95, prints precision/recall/F1 table, saves chart.

