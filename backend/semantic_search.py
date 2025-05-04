import os
import json
import numpy as np
import argparse
from sentence_transformers import SentenceTransformer
import faiss


def parse_args():
    parser = argparse.ArgumentParser(description="Semantic search using FAISS")
    parser.add_argument(
        "--index", default="faiss_index_flat.idx", help="FAISS index filename"
    )
    parser.add_argument(
        "--id-map", default="id_map.json", help="ID mapping JSON filename"
    )
    parser.add_argument(
        "--products", default="products.json", help="Products JSON filename"
    )
    parser.add_argument(
        "--model", default="text-embedding-3-small", help="Model name for embedding"
    )
    parser.add_argument(
        "--backend",
        choices=["sentence-transformers", "openai"],
        default="openai",
        help="Embedding backend to use",
    )
    return parser.parse_args()


BASE = os.path.dirname(__file__)
DATA = os.path.abspath(os.path.join(BASE, "./data"))
OUTPUT = os.path.abspath(os.path.join(BASE, "./output"))

args = parse_args()

# Load resources
index = faiss.read_index(os.path.join(OUTPUT, args.index))
ids = json.load(open(os.path.join(OUTPUT, args.id_map), "r", encoding="utf-8"))
products = {
    item["id"]: item
    for item in json.load(
        open(os.path.join(DATA, args.products), "r", encoding="utf-8")
    )
}

# Load the appropriate embedding model
if args.backend == "sentence-transformers":
    model = SentenceTransformer(args.model)
elif args.backend == "openai":
    try:
        import openai
        from tenacity import retry, stop_after_attempt, wait_random_exponential

        @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
        def get_embedding(text):
            return (
                openai.embeddings.create(input=text, model=args.model).data[0].embedding
            )

    except ImportError:
        raise ImportError("Please install openai package: pip install openai")


def semantic_search(query: str, k: int = 5):
    # Generate embeddings using the selected backend
    if args.backend == "sentence-transformers":
        q_emb = model.encode([query])
    elif args.backend == "openai":
        q_emb = np.array([get_embedding(query)])

    q_emb = np.array(q_emb, dtype="float32")

    # Check dimensions match
    if q_emb.shape[1] != index.d:
        raise ValueError(
            f"Embedding dimension mismatch: Model produces {q_emb.shape[1]}D vectors but index expects {index.d}D vectors. "
            f"Make sure you're using the same model that was used to build the index."
        )

    distances, indices = index.search(q_emb, k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx >= len(ids) or idx < 0:
            print(f"Warning: Invalid index {idx}, skipping")
            continue
        pid = ids[idx]
        item = products[pid]
        results.append(
            {
                "id": pid,
                "title": item["title"],
                "category": item.get("category"),
                "price": item.get("price"),
                "url": item.get("url"),
                "distance": float(dist),
            }
        )
    return results
