#!/usr/bin/env python3
"""
Script to compare different embedding models for the gift recommendation system.
Tests various embedding models with the optimal FAISS configuration to determine
which embedding model provides the best performance.
"""

import os
import json
import argparse
import subprocess
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

BASE = os.path.dirname(__file__)
OUTPUT_DIR = os.path.abspath(
    os.path.join(BASE, "../backend/output/embedding_comparison")
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare embedding models")
    parser.add_argument(
        "--openai-models",
        nargs="+",
        default=[
            "text-embedding-3-small",
        ],
        help="OpenAI embedding models to test",
    )
    parser.add_argument(
        "--st-models",
        nargs="+",
        default=[
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
        ],
        help="SentenceTransformers models to test",
    )
    parser.add_argument(
        "--index-type",
        choices=["flat", "hnsw"],
        default="flat",
        help="FAISS index type to use (flat or hnsw)",
    )
    parser.add_argument(
        "--hnsw-m",
        type=int,
        default=32,
        help="HNSW M parameter (used only with hnsw index type)",
    )
    parser.add_argument(
        "--ef-construction",
        type=int,
        default=40,
        help="HNSW ef_construction parameter (used only with hnsw index type)",
    )
    parser.add_argument(
        "--ef-search",
        type=int,
        default=100,
        help="HNSW ef_search parameter (used only with hnsw index type)",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20],
        help="K values for evaluation",
    )
    return parser.parse_args()


def test_embedding_model(
    backend,
    model,
    index_type,
    hnsw_m=32,
    ef_construction=40,
    k_values=[10],
    ef_search=100,
):
    """Test a single embedding model with specified index configuration"""
    print(f"\n=== Testing {backend}/{model} with {index_type} index ===")
    results = {}

    # 1. Build the index with this embedding model
    build_cmd = [
        "python",
        "../backend/src/build_index.py",
        "--backend",
        backend,
        "--model",
        model,
        "--index-type",
        index_type,
    ]

    # Add HNSW-specific parameters only if using HNSW
    if index_type == "hnsw":
        build_cmd.extend(
            [
                "--hnsw-m",
                str(hnsw_m),
                "--ef-construction",
                str(ef_construction),
            ]
        )

    print(f"Building index: {' '.join(build_cmd)}")
    start_time = time.time()
    build_result = subprocess.run(build_cmd, capture_output=True, text=True)
    build_time = time.time() - start_time

    if build_result.returncode != 0:
        print(f"Error building index for {backend}/{model}:")
        print(build_result.stderr)
        return None

    print(f"Index built in {build_time:.2f} seconds")

    # Record embedding time from the output if available
    embedding_time = None
    for line in build_result.stdout.split("\n"):
        if "Generated" in line and "embeddings in" in line:
            try:
                embedding_time = float(line.split("in")[1].split("s")[0].strip())
            except:
                pass

    # 2. Run evaluation
    index_filename = f"faiss_index_{index_type}.idx"
    index_path = f"../backend/output/{index_filename}"

    # Only modify HNSW-specific parameters if using HNSW index
    if index_type == "hnsw":
        # Create a temp script to set efSearch before evaluation
        temp_script = """
import sys
import faiss

# Load the index
index_path = sys.argv[1]
ef_search = int(sys.argv[2])

# Load and modify the index
index = faiss.read_index(index_path)
index.hnsw.efSearch = ef_search
faiss.write_index(index, index_path)
print(f"Set efSearch = {ef_search} for {index_path}")
        """

        with open("../backend/output/set_ef_search.py", "w") as f:
            f.write(temp_script)

        # Set ef_search before evaluation
        subprocess.run(
            ["python", "../backend/output/set_ef_search.py", index_path, str(ef_search)]
        )

    # Run evaluation for different k values
    eval_results = {}

    for k in k_values:
        eval_cmd = [
            "python",
            "../backend/src/evaluate.py",
            "--backend",
            backend,
            "--model",
            model,
            "--index",
            index_filename,
            "--k",
            str(k),
        ]

        print(f"Evaluating with k={k}")
        eval_result = subprocess.run(eval_cmd, capture_output=True, text=True)

        if eval_result.returncode != 0:
            print(f"Error evaluating with k={k}:")
            print(eval_result.stderr)
            continue

        # Try to extract metrics from the evaluation output
        try:
            # Look for the metrics in the output directory
            eval_results_path = "../backend/output/eval_results.json"
            if os.path.exists(eval_results_path):
                with open(eval_results_path, "r") as f:
                    metrics = json.load(f)
                    eval_results[k] = metrics.get(str(k), {})
        except Exception as e:
            print(f"Error extracting metrics: {e}")

    # 3. Get model info
    model_info = {
        "backend": backend,
        "model": model,
        "build_time": build_time,
        "embedding_time": embedding_time,
        "evaluation": eval_results,
    }

    # Save model info
    model_dir = os.path.join(OUTPUT_DIR, f"{backend}_{model.replace('/', '-')}")
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(model_dir, "results.json"), "w") as f:
        json.dump(model_info, f, indent=2)

    # Save stdout and stderr for debugging
    with open(os.path.join(model_dir, "build_output.txt"), "w") as f:
        f.write(build_result.stdout)
        f.write("\n\nSTDERR:\n")
        f.write(build_result.stderr)

    return model_info


def summarize_results(all_results, k=10):
    """Summarize and compare results from all models"""
    summaries = []

    for model_info in all_results:
        if not model_info:
            continue

        backend = model_info["backend"]
        model = model_info["model"]

        # Get metrics for the specific k value
        k_metrics = model_info["evaluation"][k]

        summaries.append(
            {
                "backend": backend,
                "model": model,
                "precision": k_metrics.get("precision", 0),
                "recall": k_metrics.get("recall", 0),
                "mrr": k_metrics.get("mrr", 0),
                "ndcg": k_metrics.get("ndcg", 0),
                "build_time": model_info.get("build_time", 0),
                "embedding_time": model_info.get("embedding_time", 0),
            }
        )

    return pd.DataFrame(summaries)


def visualize_comparison(summary_df):
    """Create visualizations comparing the embedding models"""
    # Create plots directory
    plots_dir = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Add a combined name column for easier plotting
    print(summary_df)
    if summary_df.empty:
        print("No results to visualize.")
        return

    summary_df["model_name"] = summary_df.apply(
        lambda row: f"{row['backend']}/{row['model']}", axis=1
    )

    # 1. Comparison of all metrics
    metrics = ["precision", "recall", "mrr", "ndcg"]

    plt.figure(figsize=(14, 8))

    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i + 1)
        sns.barplot(x="model_name", y=metric, data=summary_df)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{metric.upper()} Comparison")
        plt.tight_layout()

    plt.savefig(os.path.join(plots_dir, "metrics_comparison.png"))

    # 2. Build time comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(x="model_name", y="build_time", data=summary_df)
    plt.xticks(rotation=45, ha="right")
    plt.title("Build Time Comparison")
    plt.ylabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "build_time_comparison.png"))

    # 3. Radar chart of metrics
    if len(summary_df) > 0:
        try:
            plt.figure(figsize=(10, 10))

            # Prepare data for radar chart
            categories = metrics
            N = len(categories)

            # Create angles for each metric
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop

            # Initialize the plot
            ax = plt.subplot(111, polar=True)

            # Draw one axis per variable and add labels
            plt.xticks(angles[:-1], categories, size=12)

            # Draw ylabels (values)
            ax.set_rlabel_position(0)
            plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], size=10)
            plt.ylim(0, 1)

            # Plot each model
            for i, row in summary_df.iterrows():
                values = [row[metric] for metric in metrics]
                values += values[:1]  # Close the loop

                # Plot values
                ax.plot(
                    angles,
                    values,
                    linewidth=2,
                    linestyle="solid",
                    label=row["model_name"],
                )
                ax.fill(angles, values, alpha=0.1)

            # Add legend
            plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
            plt.title("Metric Comparison Across Models", size=15)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "radar_comparison.png"))
        except Exception as e:
            print(f"Error creating radar chart: {e}")

    # 4. Backend comparison
    if len(summary_df["backend"].unique()) > 1:
        # Define numeric columns to average
        numeric_cols = [
            "precision",
            "recall",
            "mrr",
            "ndcg",
            "build_time",
            "embedding_time",
        ]

        # Ensure numeric columns are actually numeric, coercing errors to NaN
        for col in numeric_cols:
            summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce")

        # Group by backend and calculate mean only for numeric columns
        backend_summary = (
            summary_df.groupby("backend")[numeric_cols].mean().reset_index()
        )

        for metric in metrics:
            plt.figure(figsize=(8, 5))
            sns.barplot(x="backend", y=metric, data=backend_summary)
            plt.title(f"{metric.upper()} by Backend")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{metric}_by_backend.png"))
            plt.close()


def find_best_model(summary_df, metric="ndcg"):
    """Find the best performing model based on the given metric"""
    if summary_df.empty:
        return None

    # Sort by the given metric
    sorted_df = summary_df.sort_values(by=metric, ascending=False)

    # Get the best model
    best_model = sorted_df.iloc[0]

    return best_model


def main():
    args = parse_args()
    all_results = []

    # 1. Test OpenAI models
    for model in args.openai_models:
        try:
            print(f"\n=== Testing OpenAI model {model} ===")
            result = test_embedding_model(
                "openai",
                model,
                args.index_type,
                args.hnsw_m,
                args.ef_construction,
                args.k_values,
                args.ef_search,
            )
            if result:
                all_results.append(result)
            else:
                print(f"Failed to get results for {model}")
        except Exception as e:
            print(f"Error testing OpenAI model {model}: {e}")

    # 2. Test SentenceTransformers models
    for model in args.st_models:
        try:
            print(f"=== Testing SentenceTransformer model: {model} ===")
            result = test_embedding_model(
                "sentence-transformers",
                model,
                args.index_type,
                args.hnsw_m,
                args.ef_construction,
                args.k_values,
                args.ef_search,
            )

            if result:
                all_results.append(result)
            else:
                print(f"Failed to get results for {model}")
        except Exception as e:
            print(f"Error testing SentenceTransformer model {model}: {e}")

    # 3. Summarize results
    print("\nSummarizing results...")
    print("All results:" + str(all_results))
    for k in args.k_values:
        summary_df = summarize_results(all_results, k=k)

        # Save summary as CSV
        summary_path = os.path.join(OUTPUT_DIR, f"model_comparison_k{k}.csv")
        summary_df.to_csv(summary_path, index=False)

        print(f"\nModel Comparison at k={k}:")
        print(summary_df.to_string())

    # Focus on k=10 for visualizations and final recommendation
    summary_df = summarize_results(all_results, k=10)

    # 4. Visualize comparison
    print("\nGenerating visualizations...")
    visualize_comparison(summary_df)

    # 5. Find the best model
    best_model = find_best_model(summary_df, metric="ndcg")

    if best_model is not None:
        print("\n=== BEST EMBEDDING MODEL ===")
        print(f"Backend: {best_model['backend']}")
        print(f"Model: {best_model['model']}")
        print(f"NDCG@10: {best_model['ndcg']:.4f}")
        print(f"Precision@10: {best_model['precision']:.4f}")
        print(f"Recall@10: {best_model['recall']:.4f}")
        print(f"MRR@10: {best_model['mrr']:.4f}")

        # Save as JSON for future reference
        with open(os.path.join(OUTPUT_DIR, "best_embedding_model.json"), "w") as f:
            best_model_dict = best_model.to_dict()
            json.dump(best_model_dict, f, indent=2)

        # Print command to use the best model
        print("\nTo use this model for your production system, run:")
        print(
            f"python ../backend/src/build_index.py "
            + f"--backend {best_model['backend']} "
            + f"--model {best_model['model']} "
            + f"--index-type {args.index_type} "
            + (f"--hnsw-m {args.hnsw_m}" if args.index_type == "hnsw" else "")
        )
    else:
        print("\nCould not determine the best model.")

    print(f"\nAll results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
