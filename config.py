import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
CLUSTERING_DIR = os.path.join(BASE_DIR, "clustering", "artifacts")

# embedding model - small but solid for semantic tasks
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384

# faiss index file
FAISS_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "corpus.index")
DOC_STORE_PATH = os.path.join(EMBEDDINGS_DIR, "doc_store.pkl")

# clustering
N_CLUSTERS = 15  # BIC sweep in fuzzy_cluster.select_n_clusters() shows diminishing returns after 15
GMM_ARTIFACTS_PATH = os.path.join(CLUSTERING_DIR, "gmm_model.pkl")
CLUSTER_META_PATH = os.path.join(CLUSTERING_DIR, "cluster_meta.pkl")

# cache
# threshold exploration (scripts/threshold_analysis.py) showed 0.65 gives best F1
# precision stays 1.0 across all values - the model separates related/unrelated cleanly
# recall drops fast above 0.75, so 0.65 is the practical sweet spot
SIMILARITY_THRESHOLD = 0.65
