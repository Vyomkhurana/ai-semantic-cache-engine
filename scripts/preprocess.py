import re
import os
import pickle
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.datasets import fetch_20newsgroups
from config import DATA_DIR


def strip_headers(text):
    # newsgroup posts have 'From:', 'Subject:' etc at top - not useful for semantic content
    lines = text.split("\n")
    body_start = 0
    for i, line in enumerate(lines):
        if line.strip() == "":
            body_start = i + 1
            break
    return "\n".join(lines[body_start:])


def remove_quotes(text):
    # quoted reply lines start with > - they're someone else's words
    lines = [l for l in text.split("\n") if not l.strip().startswith(">")]
    return "\n".join(lines)


def clean_text(text):
    text = strip_headers(text)
    text = remove_quotes(text)

    # remove email addresses and urls
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)

    # remove special chars but keep sentence structure
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?\-]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_corpus(subset="all"):
    print(f"fetching 20newsgroups ({subset})...")
    # remove='headers,footers,quotes' is sklearn's built-in but we do manual too
    # keeping categories=None to get all 20
    data = fetch_20newsgroups(
        subset=subset,
        remove=("headers", "footers", "quotes"),
        random_state=42
    )
    return data


def build_clean_corpus(min_len=80, max_len=3000):
    # p5/p95 of word count distribution - drops one-liners and faq digests
    raw = load_corpus()

    docs = []
    for idx, (text, label) in enumerate(zip(raw.data, raw.target)):
        cleaned = clean_text(text)
        word_count = len(cleaned.split())

        if word_count < min_len or word_count > max_len:
            continue

        docs.append({
            "id": idx,
            "text": cleaned,
            "label": int(label),
            "category": raw.target_names[label],
            "word_count": word_count
        })

    print(f"kept {len(docs)} / {len(raw.data)} documents after filtering")

    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, "corpus.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(docs, f)

    print(f"saved to {out_path}")
    return docs


if __name__ == "__main__":
    docs = build_clean_corpus()
    cats = {}
    for d in docs:
        cats[d["category"]] = cats.get(d["category"], 0) + 1
    print("\ndocs per category:")
    for k, v in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
