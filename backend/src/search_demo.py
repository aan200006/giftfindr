import os
import json
import numpy as np
import argparse
from sentence_transformers import SentenceTransformer
import faiss


def parse_args():
    parser = argparse.ArgumentParser(description="Semantic search using FAISS")
    parser.add_argument(
        "--index", default="faiss_index.idx", help="FAISS index filename"
    )
    parser.add_argument(
        "--id-map", default="id_map.json", help="ID mapping JSON filename"
    )
    parser.add_argument(
        "--products", default="products.json", help="Products JSON filename"
    )
    return parser.parse_args()


BASE = os.path.dirname(__file__)
DATA = os.path.abspath(os.path.join(BASE, "../data"))
OUTPUT = os.path.abspath(os.path.join(BASE, "../output"))

args = parse_args()

index = faiss.read_index(os.path.join(OUTPUT, args.index))
ids = json.load(open(os.path.join(OUTPUT, args.id_map), "r", encoding="utf-8"))
products = {
    item["id"]: item
    for item in json.load(
        open(os.path.join(DATA, args.products), "r", encoding="utf-8")
    )
}
model = SentenceTransformer("all-mpnet-base-v2")


def semantic_search(query: str, k: int = 5):
    q_emb = model.encode([query])
    q_emb = np.array(q_emb, dtype="float32")
    distances, indices = index.search(q_emb, k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
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


if __name__ == "__main__":
    print(
        f"Using index: {args.index}, id map: {args.id_map}, products: {args.products}"
    )
    while True:
        q = input("Enter search query (or 'quit'): ")
        if q.lower() == "quit":
            break
        hits = semantic_search(q, k=5)
        print("Top results:")
        for h in hits:
            print(
                f"- {h['title']}  ({h['url']}) (${h['price']}) [{h['category']}] (dist={h['distance']:.3f})"
            )
        print()
