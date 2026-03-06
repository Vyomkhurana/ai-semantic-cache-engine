# ai-semantic-cache-engine

semantic search system over the 20 Newsgroups corpus, with fuzzy clustering and a custom semantic cache layer.

## setup

```bash
python -m venv venv
venv\Scripts\activate      # windows
# source venv/bin/activate   # mac/linux

pip install -r requirements.txt
```

## build the data pipeline

run these in order once:

```bash
python scripts/preprocess.py
python scripts/build_index.py
python clustering/fuzzy_cluster.py
```

this downloads the dataset, cleans it, builds the faiss index, and fits the GMM.

## start the API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## endpoints

**POST /query**
```json
{ "query": "what are the best linux distributions?" }
```

**GET /cache/stats**

**DELETE /cache**

## design notes

- embedding model: `all-MiniLM-L6-v2` — good quality/speed trade-off, 384d, runs locally
- vector store: FAISS `IndexFlatIP` — exact cosine search, fine at this corpus size
- clustering: gaussian mixture model, 15 components, soft assignments — each doc gets a probability distribution over clusters, not a hard label
- cache threshold: 0.82 cosine similarity — this is the main tunable. too high = behaves like exact match. too low = wrong results surface. explored in `notebooks/threshold_exploration.ipynb`
- cluster routing: on cache lookup, we only compare against entries in the same dominant cluster. keeps lookup fast as cache grows.
