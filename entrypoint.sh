#!/bin/bash
set -e

if [ ! -f "data/corpus.pkl" ]; then
    echo "building corpus..."
    python scripts/preprocess.py
fi

if [ ! -f "embeddings/corpus.index" ]; then
    echo "building faiss index..."
    python scripts/build_index.py
fi

if [ ! -f "clustering/artifacts/gmm_model.pkl" ]; then
    echo "fitting clusters..."
    python clustering/fuzzy_cluster.py
fi

exec uvicorn main:app --host 0.0.0.0 --port 8000
