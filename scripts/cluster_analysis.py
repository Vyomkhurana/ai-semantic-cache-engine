import os
import sys
import pickle
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DOC_STORE_PATH, GMM_ARTIFACTS_PATH, CLUSTER_META_PATH, CLUSTERING_DIR


def load_all():
    with open(DOC_STORE_PATH, "rb") as f:
        store = pickle.load(f)
    with open(GMM_ARTIFACTS_PATH, "rb") as f:
        artifacts = pickle.load(f)
    with open(CLUSTER_META_PATH, "rb") as f:
        meta = pickle.load(f)
    return store["docs"], artifacts, meta


def cluster_breakdown(docs, meta):
    probs = meta["probs"]
    hard = meta["hard_labels"]
    cluster_labels = meta["cluster_labels"]
    n = meta["n_clusters"]

    print("=" * 65)
    print("CLUSTER BREAKDOWN")
    print("=" * 65)

    for c in range(n):
        members = [docs[i] for i in range(len(docs)) if hard[i] == c]
        if not members:
            continue

        # category distribution
        cat_counts = {}
        for d in members:
            cat_counts[d["category"]] = cat_counts.get(d["category"], 0) + 1
        top_cats = sorted(cat_counts.items(), key=lambda x: -x[1])[:4]

        # avg confidence for members
        member_idx = [i for i in range(len(docs)) if hard[i] == c]
        avg_conf = np.mean([probs[i][c] for i in member_idx])

        print(f"\ncluster {c:2d}  [{cluster_labels[c]}]  ({len(members)} docs, avg confidence={avg_conf:.2f})")
        print("  top categories:")
        for cat, cnt in top_cats:
            pct = cnt / len(members) * 100
            print(f"    {cat:<35} {cnt:>4} ({pct:.0f}%)")

        # sample doc
        sample = members[0]["text"][:180].replace("\n", " ")
        print(f"  sample: ...{sample}...")


def boundary_analysis(docs, meta):
    probs = meta["probs"]
    cluster_labels = meta["cluster_labels"]

    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    margin = sorted_probs[:, 0] - sorted_probs[:, 1]
    boundary_idx = np.argsort(margin)[:12]

    print("\n" + "=" * 65)
    print("BOUNDARY / AMBIGUOUS DOCUMENTS")
    print("these docs sit between two clusters - the interesting cases")
    print("=" * 65)

    for i in boundary_idx:
        top2 = np.argsort(probs[i])[::-1][:2]
        c1, c2 = int(top2[0]), int(top2[1])
        p1, p2 = float(probs[i][c1]), float(probs[i][c2])
        doc = docs[i]
        snippet = doc["text"][:160].replace("\n", " ")

        print(f"\n  category : {doc['category']}")
        print(f"  cluster  : {c1} ({cluster_labels[c1]}, p={p1:.3f})  vs  {c2} ({cluster_labels[c2]}, p={p2:.3f})")
        print(f"  text     : {snippet}...")


def high_confidence_members(docs, meta, top_n=3):
    probs = meta["probs"]
    hard = meta["hard_labels"]
    cluster_labels = meta["cluster_labels"]
    n = meta["n_clusters"]

    print("\n" + "=" * 65)
    print("HIGH CONFIDENCE MEMBERS (most 'pure' cluster examples)")
    print("=" * 65)

    for c in range(n):
        member_idx = [i for i in range(len(docs)) if hard[i] == c]
        if not member_idx:
            continue
        # sort by confidence descending
        sorted_by_conf = sorted(member_idx, key=lambda i: -probs[i][c])
        print(f"\ncluster {c} [{cluster_labels[c]}]")
        for i in sorted_by_conf[:top_n]:
            conf = probs[i][c]
            snippet = docs[i]["text"][:120].replace("\n", " ")
            print(f"  [{conf:.3f}] {docs[i]['category']:30} {snippet[:80]}...")


def main():
    if not os.path.exists(CLUSTER_META_PATH):
        print("artifacts not found - run clustering/fuzzy_cluster.py first")
        sys.exit(1)

    docs, artifacts, meta = load_all()

    cluster_breakdown(docs, meta)
    high_confidence_members(docs, meta)
    boundary_analysis(docs, meta)

    print("\n" + "=" * 65)
    n = meta["n_clusters"]
    probs = meta["probs"]
    avg_entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1).mean()
    print(f"corpus stats: {len(docs)} docs, {n} clusters")
    print(f"avg assignment entropy: {avg_entropy:.3f}  (0=perfectly hard, log({n})={np.log(n):.2f}=uniform)")
    print(f"  -> lower entropy means clusters are well-separated")


if __name__ == "__main__":
    main()
