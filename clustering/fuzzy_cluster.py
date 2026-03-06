import os
import sys
import pickle
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    EMBEDDINGS_DIR, CLUSTERING_DIR,
    DOC_STORE_PATH, N_CLUSTERS,
    GMM_ARTIFACTS_PATH, CLUSTER_META_PATH
)


PCA_DIMS = 64


def load_embeddings():
    with open(DOC_STORE_PATH, "rb") as f:
        store = pickle.load(f)
    return store["embeddings"], store["docs"]


def reduce_dims(embeddings, n_components=PCA_DIMS):
    print(f"reducing {embeddings.shape[1]}d -> {n_components}d with PCA...")
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(embeddings)
    explained = pca.explained_variance_ratio_.sum()
    print(f"explained variance: {explained:.3f}")
    return reduced, pca


def select_n_clusters(reduced, max_k=25):
    # sweep BIC to find elbow - lower BIC = better fit without over-complicating
    print("running BIC sweep to validate cluster count...")
    bic_scores = []
    ks = range(5, max_k + 1, 2)
    for k in ks:
        gmm = GaussianMixture(n_components=k, covariance_type="diag",
                              random_state=42, max_iter=200, n_init=3)
        gmm.fit(reduced)
        bic_scores.append((k, gmm.bic(reduced)))
        print(f"  k={k}  BIC={gmm.bic(reduced):.1f}")
    return bic_scores


def fit_gmm(reduced, n_clusters):
    print(f"fitting GMM with {n_clusters} components...")
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="diag",  # full covariance is way too expensive in 64d
        random_state=42,
        max_iter=300,
        n_init=5,
        verbose=0
    )
    gmm.fit(reduced)
    return gmm


def get_soft_assignments(gmm, reduced):
    # shape: (n_docs, n_clusters) - probabilities sum to 1 per row
    probs = gmm.predict_proba(reduced)
    hard = np.argmax(probs, axis=1)
    return probs, hard


def label_clusters(docs, hard_labels, n_clusters):
    cluster_cats = {i: {} for i in range(n_clusters)}
    for doc, cluster in zip(docs, hard_labels):
        cat = doc["category"]
        cluster_cats[cluster][cat] = cluster_cats[cluster].get(cat, 0) + 1

    labels = {}
    for c, cats in cluster_cats.items():
        top = sorted(cats.items(), key=lambda x: -x[1])[:2]
        labels[c] = " / ".join([t[0].split(".")[-1] for t in top])
    return labels


def find_boundary_docs(probs, docs, top_n=10):
    # small margin between top-2 probs = genuinely ambiguous doc
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    margin = sorted_probs[:, 0] - sorted_probs[:, 1]  # smaller = more uncertain

    boundary_idx = np.argsort(margin)[:top_n]
    results = []
    for i in boundary_idx:
        top2 = np.argsort(probs[i])[::-1][:2]
        results.append({
            "doc_id": docs[i]["id"],
            "category": docs[i]["category"],
            "text_snippet": docs[i]["text"][:200],
            "top_cluster": int(top2[0]),
            "top_prob": float(probs[i][top2[0]]),
            "second_cluster": int(top2[1]),
            "second_prob": float(probs[i][top2[1]]),
            "margin": float(margin[i])
        })
    return results


def save_results(gmm, pca, probs, hard_labels, cluster_labels, docs):
    os.makedirs(CLUSTERING_DIR, exist_ok=True)

    with open(GMM_ARTIFACTS_PATH, "wb") as f:
        pickle.dump({"gmm": gmm, "pca": pca}, f)

    meta = {
        "probs": probs,
        "hard_labels": hard_labels,
        "cluster_labels": cluster_labels,
        "n_clusters": N_CLUSTERS,
        "doc_ids": [d["id"] for d in docs]
    }
    with open(CLUSTER_META_PATH, "wb") as f:
        pickle.dump(meta, f)

    print(f"saved GMM artifacts -> {GMM_ARTIFACTS_PATH}")
    print(f"saved cluster meta  -> {CLUSTER_META_PATH}")


def main():
    embeddings, docs = load_embeddings()
    reduced, pca = reduce_dims(embeddings)

    # optional: run BIC sweep to double-check N_CLUSTERS choice
    # select_n_clusters(reduced)

    gmm = fit_gmm(reduced, N_CLUSTERS)
    probs, hard = get_soft_assignments(gmm, reduced)
    cluster_labels = label_clusters(docs, hard, N_CLUSTERS)

    print("\ncluster summary:")
    for c, label in cluster_labels.items():
        count = (hard == c).sum()
        print(f"  [{c:2d}] {label:<40} ({count} docs)")

    boundary = find_boundary_docs(probs, docs)
    print("\nboundary / ambiguous docs:")
    for b in boundary[:5]:
        print(f"  category={b['category']} "
              f"cluster={b['top_cluster']}({b['top_prob']:.2f}) "
              f"vs {b['second_cluster']}({b['second_prob']:.2f})")

    save_results(gmm, pca, probs, hard, cluster_labels, docs)
    print("clustering done.")


if __name__ == "__main__":
    main()
