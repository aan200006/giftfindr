#!/usr/bin/env python3
"""
evaluate.py

Automated evaluation script for the Gift-Recommender project.
Computes Precision@K, Recall@K, MRR@K, and nDCG@K over a gold-standard
set of queries and a FAISS index of products.

Usage:
    python evaluate.py \
      --products products.json \
      --id-map id_map.json \
      --index faiss_index.idx \
      --gold gold.json \
      --k 1 5 10 20 \
      --model all-mpnet-base-v2
"""

import os
import argparse
import json
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


BASE = os.path.dirname(__file__)
DATA = os.path.abspath(os.path.join(BASE, "../data"))
OUTPUT = os.path.abspath(os.path.join(BASE, "../output"))
os.makedirs(OUTPUT, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(...)
    parser.add_argument(
        "--products",
        type=str,
        default=os.path.join(DATA, "products.json"),
        help="Path to products JSON",
    )
    parser.add_argument(
        "--id-map",
        type=str,
        default=os.path.join(OUTPUT, "id_map.json"),
        help="Path to ID map JSON",
    )
    parser.add_argument(
        "--index",
        type=str,
        default="faiss_index.idx",
        help="Path to FAISS index file",
    )
    parser.add_argument(
        "--gold",
        type=str,
        default=os.path.join(DATA, "gold.json"),
        help="Path to gold-standard JSON",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="List of K values for metrics (e.g. --k 1 5 10 20)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-mpnet-base-v2",
        help="SentenceTransformer model name for embeddings",
    )
    return parser.parse_args()


def load_resources(products_path, id_map_path, index_path, gold_path, model_name):
    # Check if paths are absolute, otherwise join with appropriate directory
    if not os.path.isabs(products_path):
        products_path = os.path.join(DATA, products_path)

    if not os.path.isabs(id_map_path):
        id_map_path = os.path.join(OUTPUT, id_map_path)

    if not os.path.isabs(index_path):
        index_path = os.path.join(OUTPUT, index_path)

    if not os.path.isabs(gold_path):
        gold_path = os.path.join(DATA, gold_path)

    # Load product metadata
    products = {
        item["id"]: item
        for item in json.load(open(products_path, "r", encoding="utf-8"))
    }

    # Load FAISS index and id mapping
    index = faiss.read_index(index_path)
    id_map = json.load(open(id_map_path, "r", encoding="utf-8"))

    # Load gold queries
    # Format: [{"query": "...", "relevant_ids": [id1, id2, ...]}, ...]
    gold_list = json.load(open(gold_path, "r", encoding="utf-8"))
    queries = [item["query"] for item in gold_list]
    gold_sets = [set(item["relevant_ids"]) for item in gold_list]

    # Load embedding model
    model = SentenceTransformer(model_name)

    return products, index, id_map, queries, gold_sets, model


def semantic_search(model, index, id_map, query, k):
    emb = model.encode([query]).astype("float32")
    distances, indices = index.search(emb, k)
    # Map index positions back to product IDs
    return [id_map[idx] for idx in indices[0]], distances[0].tolist()


def precision_recall_at_k(recs, golds, k):
    precisions, recalls = [], []
    for rec_ids, gold in zip(recs, golds):
        preds = rec_ids[:k]
        tp = len(set(preds) & gold)
        precisions.append(tp / k)
        recalls.append(tp / (len(gold) if gold else 1))
    return np.mean(precisions), np.mean(recalls)


def mrr_at_k(recs, golds, k):
    rr_scores = []
    for rec_ids, gold in zip(recs, golds):
        rr = 0.0
        for rank, pid in enumerate(rec_ids[:k], start=1):
            if pid in gold:
                rr = 1.0 / rank
                break
        rr_scores.append(rr)
    return np.mean(rr_scores)


def ndcg_at_k(recs, golds, k):
    def dcg(scores):
        return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(scores))

    ndcgs = []
    for rec_ids, gold in zip(recs, golds):
        rels = [1 if pid in gold else 0 for pid in rec_ids[:k]]
        dcg_val = dcg(rels)
        ideal_rels = sorted(rels, reverse=True)
        idcg_val = dcg(ideal_rels)
        ndcgs.append(dcg_val / idcg_val if idcg_val > 0 else 0.0)
    return np.mean(ndcgs)


def main():
    args = parse_args()
    # Load all resources
    products, index, id_map, queries, gold_sets, model = load_resources(
        args.products, args.id_map, args.index, args.gold, args.model
    )

    metrics = {}
    recs_all = {k: [] for k in args.k}

    # Perform retrieval for each query and K
    for query in queries:
        for k in args.k:
            rec_ids, _ = semantic_search(model, index, id_map, query, k)
            recs_all[k].append(rec_ids)

    # Compute metrics for each K
    rows = []
    for k in args.k:
        p, r = precision_recall_at_k(recs_all[k], gold_sets, k)
        mrr = mrr_at_k(recs_all[k], gold_sets, k)
        ndcg = ndcg_at_k(recs_all[k], gold_sets, k)
        rows.append(
            {
                "K": k,
                "Precision": round(p, 4),
                "Recall": round(r, 4),
                "MRR": round(mrr, 4),
                "nDCG": round(ndcg, 4),
            }
        )
        metrics[k] = {"precision": p, "recall": r, "mrr": mrr, "ndcg": ndcg}

    # Display as a DataFrame
    df = pd.DataFrame(rows).set_index("K")
    print("\nRetrieval Evaluation Metrics:")
    print(df)

    # Save metrics into output/
    with open(os.path.join(OUTPUT, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    df.to_csv(os.path.join(OUTPUT, "eval_results.csv"))
    print(f"Saved metrics → {OUTPUT}")


if __name__ == "__main__":
    main()
