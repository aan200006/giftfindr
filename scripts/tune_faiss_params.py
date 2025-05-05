#!/usr/bin/env python3
"""
Script to find optimal FAISS parameters for different index types
by performing grid search and evaluating performance metrics.
"""

import os
import json
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import faiss
from pathlib import Path
import subprocess
from tqdm import tqdm

BASE = os.path.dirname(__file__)
OUTPUT_DIR = os.path.abspath(os.path.join(BASE, "../backend/output/faiss_tuning"))
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Tune FAISS parameters")
    parser.add_argument(
        "--backend",
        default="openai",
        choices=["sentence-transformers", "openai"],
        help="Embedding backend to use",
    )
    parser.add_argument(
        "--model",
        default="text-embedding-3-small",
        help="Model name for embeddings",
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        help="Path to pre-computed embeddings file (.npy)",
    )
    parser.add_argument(
        "--query-embeddings",
        type=str,
        help="Path to pre-computed query embeddings for testing (.npy)",
    )
    parser.add_argument(
        "--gold",
        type=str,
        default="../backend/data/gold.json",
        help="Path to gold standard queries for evaluation",
    )
    return parser.parse_args()


def get_embeddings(args):
    """Get or generate embeddings for tuning"""
    if args.embeddings and os.path.exists(args.embeddings):
        print(f"Loading embeddings from {args.embeddings}")
        return np.load(args.embeddings)
    else:
        # Generate embeddings using build_index.py but don't build the index
        print("Generating embeddings (this may take a while)...")
        embeddings_path = os.path.join(
            OUTPUT_DIR, f"embeddings_{args.backend}_{args.model.replace('/', '-')}.npy"
        )

        # Build a dummy flat index to generate and cache embeddings
        cmd = [
            "python",
            "../backend/src/build_index.py",
            "--backend",
            args.backend,
            "--model",
            args.model,
        ]
        subprocess.run(cmd, check=True)

        # The embeddings should now be cached in the output directory
        cached_path = os.path.join(
            "../backend/output",
            f"embeddings_{args.backend}_{args.model.replace('/', '-')}.npy",
        )
        if os.path.exists(cached_path):
            return np.load(cached_path)
        else:
            raise FileNotFoundError(
                f"Could not find or generate embeddings: {cached_path}"
            )


def get_query_embeddings(args, gold_data):
    """Get or generate query embeddings for testing"""
    if args.query_embeddings and os.path.exists(args.query_embeddings):
        print(f"Loading query embeddings from {args.query_embeddings}")
        return np.load(args.query_embeddings)

    # Use the same backend to generate query embeddings
    print("Generating query embeddings...")

    queries = [item["query"] for item in gold_data]

    if args.backend == "sentence-transformers":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(args.model)
        embeddings = model.encode(queries, show_progress_bar=True)
    else:  # openai
        import openai
        from tenacity import retry, stop_after_attempt, wait_random_exponential

        @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
        def get_embedding(text):
            return (
                openai.embeddings.create(input=text, model=args.model).data[0].embedding
            )

        embeddings = []
        for query in tqdm(queries):
            embeddings.append(get_embedding(query))
        embeddings = np.array(embeddings, dtype="float32")

    # Save for future use
    save_path = os.path.join(
        OUTPUT_DIR,
        f"query_embeddings_{args.backend}_{args.model.replace('/', '-')}.npy",
    )
    np.save(save_path, embeddings)
    print(f"Saved query embeddings to {save_path}")

    return embeddings


def build_and_test_index(
    embeddings, params, index_type, query_embeddings, gold_data, k=10
):
    """Build an index with the given parameters and test its performance"""
    dim = embeddings.shape[1]
    build_start = time.time()

    if index_type == "flat":
        index = faiss.IndexFlatL2(dim)
    elif index_type == "ivfpq":
        nlist = params.get("nlist", 100)
        m = params.get("m", 8)
        nbits = params.get("nbits", 8)

        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
        index.train(embeddings)
    elif index_type == "hnsw":
        M = params.get("hnsw_m", 32)
        ef_construction = params.get("ef_construction", 40)

        index = faiss.IndexHNSWFlat(dim, M)
        # Set ef_construction (higher = more accurate but slower build)
        index.hnsw.efConstruction = ef_construction

    # Add vectors to index
    index.add(embeddings)
    build_time = time.time() - build_start

    # Test search performance
    search_times = []
    recalls = []

    # Load ID mapping to map index positions to actual IDs
    with open("../backend/output/id_map.json", "r") as f:
        id_map = json.load(f)

    # Convert gold IDs to strings for consistent comparison
    gold_sets = [set(str(id_) for id_ in item["relevant_ids"]) for item in gold_data]

    # For IVF indexes, we also want to test different nprobe values
    if index_type == "ivfpq":
        nprobe_values = [1, 8, 16, 32, 64, 128]
        nprobe_results = []

        for nprobe in nprobe_values:
            index.nprobe = nprobe

            # Search and measure recall
            recall_sum = 0
            times = []

            for i, query_vector in enumerate(query_embeddings):
                search_start = time.time()
                _, I = index.search(query_vector.reshape(1, -1), k)
                search_time = time.time() - search_start

                # Map returned indices to product IDs
                result_ids = [
                    str(id_map[idx]) for idx in I[0] if 0 <= idx < len(id_map)
                ]

                # Calculate recall by comparing with gold set
                relevant_count = len(set(result_ids) & gold_sets[i])
                recall = relevant_count / len(gold_sets[i]) if gold_sets[i] else 0

                recall_sum += recall
                times.append(search_time)

            avg_recall = recall_sum / len(query_embeddings)
            avg_time = sum(times) / len(times)

            nprobe_results.append(
                {"nprobe": nprobe, "recall": avg_recall, "search_time": avg_time}
            )

        # Get the best recall result
        best_nprobe_result = max(nprobe_results, key=lambda x: x["recall"])
        recalls.append(best_nprobe_result["recall"])
        search_times.append(best_nprobe_result["search_time"])

        # Add best nprobe to results
        params["best_nprobe"] = best_nprobe_result["nprobe"]
        params["nprobe_results"] = nprobe_results
    else:
        # For non-IVF indexes, just do a single search test
        recall_sum = 0
        times = []

        for i, query_vector in enumerate(query_embeddings):
            search_start = time.time()
            _, I = index.search(query_vector.reshape(1, -1), k)
            search_time = time.time() - search_start

            # Map returned indices to product IDs
            result_ids = [str(id_map[idx]) for idx in I[0] if 0 <= idx < len(id_map)]

            # Calculate recall by comparing with gold set
            relevant_count = len(set(result_ids) & gold_sets[i])
            recall = relevant_count / len(gold_sets[i]) if gold_sets[i] else 0

            recall_sum += recall
            times.append(search_time)

        avg_recall = recall_sum / len(query_embeddings)
        avg_time = sum(times) / len(times)

        recalls.append(avg_recall)
        search_times.append(avg_time)

    # Index size on disk
    index_path = os.path.join(OUTPUT_DIR, f"temp_index_{index_type}.idx")
    faiss.write_index(index, index_path)
    index_size = os.path.getsize(index_path) / (1024 * 1024)  # in MB
    os.remove(index_path)  # Clean up

    # Extract relevant metrics
    return {
        "index_type": index_type,
        "params": params,
        "build_time": build_time,
        "search_time": search_times[0],
        "recall": recalls[0],
        "index_size_mb": index_size,
    }


def tune_flat_index(embeddings, query_embeddings, gold_data):
    """Tune parameters for the flat index (not much to tune)"""
    result = build_and_test_index(embeddings, {}, "flat", query_embeddings, gold_data)
    return [result]


def tune_ivfpq_index(embeddings, query_embeddings, gold_data):
    """Tune parameters for the IVF-PQ index"""
    results = []

    # Grid search parameters
    nlist_values = [50, 100, 256, 512, 1024]
    m_values = [4, 8, 16]  # number of subquantizers
    nbits_values = [8]  # usually fixed at 8

    total_combinations = len(nlist_values) * len(m_values) * len(nbits_values)
    print(f"Testing {total_combinations} IVF-PQ parameter combinations...")

    for nlist in nlist_values:
        for m in m_values:
            for nbits in nbits_values:
                params = {"nlist": nlist, "m": m, "nbits": nbits}
                print(f"Testing IVF-PQ with {params}...")

                result = build_and_test_index(
                    embeddings, params, "ivfpq", query_embeddings, gold_data
                )
                results.append(result)

    return results


def tune_hnsw_index(embeddings, query_embeddings, gold_data):
    """Tune parameters for the HNSW index"""
    results = []

    # Grid search parameters
    M_values = [16, 32, 64, 128]  # neighbors per layer
    ef_construction_values = [40, 80, 200]  # higher = more accurate but slower build

    total_combinations = len(M_values) * len(ef_construction_values)
    print(f"Testing {total_combinations} HNSW parameter combinations...")

    for M in M_values:
        for ef_construction in ef_construction_values:
            params = {"hnsw_m": M, "ef_construction": ef_construction}
            print(f"Testing HNSW with {params}...")

            result = build_and_test_index(
                embeddings, params, "hnsw", query_embeddings, gold_data
            )
            results.append(result)

    return results


def visualize_results(results):
    """Create visualizations of the parameter tuning results"""
    df = pd.DataFrame(results)

    # Create output directory for plots
    plots_dir = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Scatter plot of recall vs search time
    plt.figure(figsize=(10, 6))
    for index_type in df["index_type"].unique():
        subset = df[df["index_type"] == index_type]
        plt.scatter(
            subset["search_time"] * 1000,  # convert to ms
            subset["recall"],
            label=index_type,
            alpha=0.7,
            s=100,
        )

    plt.xlabel("Search Time (ms)")
    plt.ylabel("Recall @ 10")
    plt.title("Recall vs Search Time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "recall_vs_search_time.png"))

    # 2. Bar plot of best configurations for each index type
    best_per_type = df.loc[df.groupby("index_type")["recall"].idxmax()]

    plt.figure(figsize=(12, 6))
    sns.barplot(x="index_type", y="recall", data=best_per_type)
    plt.title("Best Recall by Index Type")
    plt.ylabel("Recall @ 10")
    plt.ylim(0, 1.0)

    # Add parameter annotations
    for i, row in enumerate(best_per_type.itertuples()):
        param_text = ""
        if row.index_type == "ivfpq":
            param_text = f"nlist={row.params['nlist']}, m={row.params['m']}\nnprobe={row.params.get('best_nprobe', 'N/A')}"
        elif row.index_type == "hnsw":
            param_text = f"M={row.params['hnsw_m']}, ef_construction={row.params['ef_construction']}"

        plt.annotate(param_text, (i, row.recall + 0.01), ha="center", fontsize=8)

    plt.savefig(os.path.join(plots_dir, "best_recall_by_index_type.png"))

    # 3. Index size comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(x="index_type", y="index_size_mb", data=best_per_type)
    plt.title("Index Size by Type")
    plt.ylabel("Size (MB)")
    plt.savefig(os.path.join(plots_dir, "index_size_comparison.png"))

    # 4. For IVF-PQ, plot nprobe vs recall for the best configuration
    ivfpq_results = [r for r in results if r["index_type"] == "ivfpq"]
    if ivfpq_results:
        best_ivfpq = max(ivfpq_results, key=lambda x: x["recall"])
        if "nprobe_results" in best_ivfpq["params"]:
            nprobe_df = pd.DataFrame(best_ivfpq["params"]["nprobe_results"])

            plt.figure(figsize=(10, 6))
            plt.plot(nprobe_df["nprobe"], nprobe_df["recall"], marker="o")
            plt.title(f"Effect of nprobe on Recall (IVF-PQ)")
            plt.xlabel("nprobe")
            plt.ylabel("Recall @ 10")
            plt.grid(True, alpha=0.3)
            plt.xscale("log")
            plt.savefig(os.path.join(plots_dir, "nprobe_vs_recall.png"))

            plt.figure(figsize=(10, 6))
            plt.plot(
                nprobe_df["nprobe"], nprobe_df["search_time"] * 1000, marker="o"
            )  # ms
            plt.title(f"Effect of nprobe on Search Time (IVF-PQ)")
            plt.xlabel("nprobe")
            plt.ylabel("Search Time (ms)")
            plt.grid(True, alpha=0.3)
            plt.xscale("log")
            plt.savefig(os.path.join(plots_dir, "nprobe_vs_time.png"))

    # 5. HNSW parameter heatmap
    hnsw_results = [r for r in results if r["index_type"] == "hnsw"]
    if len(hnsw_results) >= 4:  # Only if we have enough data points
        hnsw_df = pd.DataFrame(
            [
                {
                    "M": r["params"]["hnsw_m"],
                    "ef_construction": r["params"]["ef_construction"],
                    "recall": r["recall"],
                }
                for r in hnsw_results
            ]
        )

        pivot = hnsw_df.pivot_table(
            index="M", columns="ef_construction", values="recall"
        )

        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu")
        plt.title("HNSW Parameter Impact on Recall")
        plt.savefig(os.path.join(plots_dir, "hnsw_parameter_heatmap.png"))


def main():
    args = parse_args()

    # Load or generate embeddings
    embeddings = get_embeddings(args)
    print(f"Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")

    # Load gold standard data
    with open(args.gold, "r") as f:
        gold_data = json.load(f)
    print(f"Loaded {len(gold_data)} gold standard queries")

    # Get query embeddings
    query_embeddings = get_query_embeddings(args, gold_data)
    print(f"Using {len(query_embeddings)} query embeddings")

    # Run parameter tuning for each index type
    results = []

    print("\n1. Tuning Flat index (baseline)...")
    flat_results = tune_flat_index(embeddings, query_embeddings, gold_data)
    results.extend(flat_results)

    print("\n2. Tuning IVF-PQ index...")
    ivfpq_results = tune_ivfpq_index(embeddings, query_embeddings, gold_data)
    results.extend(ivfpq_results)

    print("\n3. Tuning HNSW index...")
    hnsw_results = tune_hnsw_index(embeddings, query_embeddings, gold_data)
    results.extend(hnsw_results)

    # Save all results
    with open(os.path.join(OUTPUT_DIR, "faiss_tuning_results.json"), "w") as f:
        json.dump(
            {
                "backend": args.backend,
                "model": args.model,
                "embeddings_dim": embeddings.shape[1],
                "num_vectors": len(embeddings),
                "results": results,
            },
            f,
            indent=2,
        )

    # Find best configuration overall
    best_config = max(results, key=lambda x: x["recall"])
    print("\n=== BEST FAISS CONFIGURATION ===")
    print(f"Index Type: {best_config['index_type']}")

    param_str = ""
    if best_config["index_type"] == "flat":
        param_str = "No parameters (exact search)"
    elif best_config["index_type"] == "ivfpq":
        param_str = (
            f"nlist={best_config['params']['nlist']}, "
            f"m={best_config['params']['m']}, "
            f"nbits={best_config['params']['nbits']}, "
            f"nprobe={best_config['params'].get('best_nprobe', 'N/A')}"
        )
    elif best_config["index_type"] == "hnsw":
        param_str = (
            f"M={best_config['params']['hnsw_m']}, "
            f"ef_construction={best_config['params']['ef_construction']}"
        )

    print(f"Parameters: {param_str}")
    print(f"Recall@10: {best_config['recall']:.4f}")
    print(f"Search Time: {best_config['search_time']*1000:.2f} ms")
    print(f"Index Size: {best_config['index_size_mb']:.2f} MB")

    # Save best configuration
    with open(os.path.join(OUTPUT_DIR, "best_faiss_config.json"), "w") as f:
        json.dump(
            {
                "index_type": best_config["index_type"],
                "params": best_config["params"],
                "performance": {
                    "recall": best_config["recall"],
                    "search_time": best_config["search_time"],
                    "index_size_mb": best_config["index_size_mb"],
                },
            },
            f,
            indent=2,
        )

    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_results(results)

    print(f"\nAll results saved to {OUTPUT_DIR}")
    print(f"To use the optimal configuration, run:")

    cmd = f"python ../backend/src/build_index.py --backend {args.backend} --model {args.model} --index-type {best_config['index_type']}"

    if best_config["index_type"] == "ivfpq":
        cmd += (
            f" --nlist {best_config['params']['nlist']}"
            f" --m {best_config['params']['m']}"
            f" --nbits {best_config['params']['nbits']}"
        )
    elif best_config["index_type"] == "hnsw":
        cmd += f" --hnsw-m {best_config['params']['hnsw_m']}"

    print(f"\n{cmd}")


if __name__ == "__main__":
    main()
