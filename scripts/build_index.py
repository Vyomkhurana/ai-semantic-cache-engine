import os
import sys
import pickle
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
import faiss

from config import (
    DATA_DIR, EMBEDDINGS_DIR,
    EMBED_MODEL, EMBED_DIM,
    FAISS_INDEX_PATH, DOC_STORE_PATH
)


def load_corpus():
    path = os.path.join(DATA_DIR, "corpus.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_embeddings(docs, model, batch_size=64):
    texts = [d["text"] for d in docs]
    print(f"encoding {len(texts)} docs with {EMBED_MODEL}...")

    # encode in batches - show_progress_bar gives visibility on runtime
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True  # cosine sim via dot product after L2 norm
    )
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings):
    # IndexFlatIP = exact inner product search, works as cosine since vectors are normalized
    # for 10k-15k docs exact search is fast enough, no need for approximate methods
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)
    print(f"faiss index built with {index.ntotal} vectors")
    return index


def save_artifacts(index, docs, embeddings):
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"saved index -> {FAISS_INDEX_PATH}")

    # store docs + embeddings together so retrieval has full context
    store = {
        "docs": docs,
        "embeddings": embeddings
    }
    with open(DOC_STORE_PATH, "wb") as f:
        pickle.dump(store, f)
    print(f"saved doc store -> {DOC_STORE_PATH}")


def main():
    docs = load_corpus()
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = compute_embeddings(docs, model)
    index = build_faiss_index(embeddings)
    save_artifacts(index, docs, embeddings)
    print("done.")


if __name__ == "__main__":
    main()
