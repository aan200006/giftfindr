import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Load index and mappings
index = faiss.read_index("faiss_index.idx")
ids = json.load(open("id_map.json", "r", encoding="utf-8"))
products = {
    item["id"]: item for item in json.load(open("products.json", "r", encoding="utf-8"))
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
                "distance": float(dist),
            }
        )
    return results


if __name__ == "__main__":
    while True:
        q = input("Enter search query (or 'quit'): ")
        if q.lower() == "quit":
            break
        hits = semantic_search(q, k=5)
        print("Top results:")
        for h in hits:
            print(
                f"- {h['title']} (${h['price']}) [{h['category']}] (dist={h['distance']:.3f})"
            )
        print()
