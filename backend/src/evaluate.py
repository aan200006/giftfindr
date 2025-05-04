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
      --index faiss_index_flat.idx \
      --gold gold.json \
      --k 1 5 10 20 \
      --model text-embedding-3-small \
      --backend openai
"""

import os
import argparse
import json
import numpy as np
import faiss
import pandas as pd
import traceback
from sentence_transformers import SentenceTransformer


BASE = os.path.dirname(__file__)
DATA = os.path.abspath(os.path.join(BASE, "../data"))
OUTPUT = os.path.abspath(os.path.join(BASE, "../output"))
os.makedirs(OUTPUT, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate gift recommendation system")
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
        default="faiss_index_flat.idx",
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
        default="text-embedding-3-small",
        help="Model name for embeddings",
    )
    parser.add_argument(
        "--backend",
        choices=["sentence-transformers", "openai"],
        default="openai",
        help="Embedding backend to use",
    )
    return parser.parse_args()


def load_resources(
    products_path, id_map_path, index_path, gold_path, model_name, backend="openai"
):
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

    try:
        id_map = json.load(open(id_map_path, "r", encoding="utf-8"))
        print(f"Loaded ID map with {len(id_map)} entries")

        # Verify index size matches id_map size
        if index.ntotal != len(id_map):
            print(
                f"WARNING: FAISS index has {index.ntotal} entries but ID map has {len(id_map)} entries"
            )
    except Exception as e:
        print(f"Error loading ID map: {e}")
        raise

    # Load gold queries
    # Format: [{"query": "...", "relevant_ids": [id1, id2, ...]}, ...]
    gold_list = json.load(open(gold_path, "r", encoding="utf-8"))
    queries = [item["query"] for item in gold_list]
    gold_sets = [set(item["relevant_ids"]) for item in gold_list]

    # Load embedding model based on backend
    if backend == "sentence-transformers":
        model = SentenceTransformer(model_name)
    elif backend == "openai":
        try:
            import openai
            from tenacity import retry, stop_after_attempt, wait_random_exponential

            @retry(
                wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6)
            )
            def get_embedding(text):
                return (
                    openai.embeddings.create(input=text, model=model_name)
                    .data[0]
                    .embedding
                )

            # Create a model-like object with encode method for consistency
            class OpenAIModel:
                def encode(self, texts):
                    embeddings = [get_embedding(text) for text in texts]
                    return np.array(embeddings, dtype="float32")

            model = OpenAIModel()
        except ImportError:
            raise ImportError("Please install openai package: pip install openai")

    return products, index, id_map, queries, gold_sets, model


def semantic_search(model, index, id_map, query, k):
    emb = model.encode([query]).astype("float32")

    # Check dimensions match
    if emb.shape[1] != index.d:
        raise ValueError(
            f"Embedding dimension mismatch: Model produces {emb.shape[1]}D vectors but index expects {index.d}D vectors. "
            f"Make sure you're using the same model that was used to build the index."
        )

    distances, indices = index.search(emb, k)

    # Debug info for indices
    max_idx = max(indices[0]) if len(indices[0]) > 0 else -1
    print(
        f"Query: '{query}', Max index returned: {max_idx}, ID map length: {len(id_map)}"
    )

    # Safely map index positions back to product IDs
    result_ids = []
    for idx in indices[0]:
        if 0 <= idx < len(id_map):
            result_ids.append(id_map[idx])
        else:
            print(
                f"WARNING: Index {idx} out of range for ID map (length: {len(id_map)})"
            )
            # Use a placeholder ID or skip
            result_ids.append(-1)  # Use -1 as a placeholder for invalid IDs

    return result_ids, distances[0].tolist()


def precision_recall_at_k(recs, golds, k):
    precisions, recalls = [], []
    for rec_ids, gold in zip(recs, golds):
        preds = rec_ids[:k]
        # Filter out invalid IDs (placeholder -1)
        valid_preds = [pid for pid in preds if pid != -1]
        tp = len(set(valid_preds) & gold)
        precisions.append(tp / (len(valid_preds) if valid_preds else 1))
        recalls.append(tp / (len(gold) if gold else 1))
    return np.mean(precisions), np.mean(recalls)


def mrr_at_k(recs, golds, k):
    rr_scores = []
    for rec_ids, gold in zip(recs, golds):
        rr = 0.0
        for rank, pid in enumerate(rec_ids[:k], start=1):
            if pid != -1 and pid in gold:
                rr = 1.0 / rank
                break
        rr_scores.append(rr)
    return np.mean(rr_scores)


def ndcg_at_k(recs, golds, k):
    def dcg(scores):
        return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(scores))

    ndcgs = []
    for rec_ids, gold in zip(recs, golds):
        rels = [1 if (pid != -1 and pid in gold) else 0 for pid in rec_ids[:k]]
        dcg_val = dcg(rels)
        ideal_rels = sorted(rels, reverse=True)
        idcg_val = dcg(ideal_rels)
        ndcgs.append(dcg_val / idcg_val if idcg_val > 0 else 0.0)
    return np.mean(ndcgs)


def main():
    args = parse_args()

    try:
        # Load all resources
        products, index, id_map, queries, gold_sets, model = load_resources(
            args.products, args.id_map, args.index, args.gold, args.model, args.backend
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
        with open(
            os.path.join(OUTPUT, "eval_results.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(metrics, f, indent=2)
        df.to_csv(os.path.join(OUTPUT, "eval_results.csv"))
        print(f"Saved metrics → {OUTPUT}")

    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
