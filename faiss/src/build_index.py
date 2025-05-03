import os
import json
import numpy as np
import argparse
from sentence_transformers import SentenceTransformer
import faiss
import time

BASE = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(BASE, "../data"))
OUT_DIR = os.path.abspath(os.path.join(BASE, "../output"))
os.makedirs(OUT_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build FAISS index from product catalog"
    )
    parser.add_argument(
        "--backend",
        choices=["sentence-transformers", "openai"],
        default="sentence-transformers",
        help="Embedding backend",
    )
    parser.add_argument(
        "--model", default="all-mpnet-base-v2", help="Model name for selected backend"
    )
    parser.add_argument(
        "--index-type",
        choices=["flat", "ivfpq", "hnsw"],
        default="flat",
        help="FAISS index type",
    )
    parser.add_argument(
        "--nlist", type=int, default=100, help="Number of clusters for IVF index"
    )
    parser.add_argument(
        "--m", type=int, default=8, help="Number of subquantizers for PQ"
    )
    parser.add_argument(
        "--nbits", type=int, default=8, help="Number of bits per subquantizer for PQ"
    )
    parser.add_argument(
        "--hnsw-m", type=int, default=32, help="Number of neighbors for HNSW index"
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Don't use cached embeddings"
    )
    return parser.parse_args()


def get_embeddings(texts, args):
    cache_path = os.path.join(
        OUT_DIR, f"embeddings_{args.backend}_{args.model.replace('/', '-')}.npy"
    )

    # Try to load from cache unless --no-cache is specified
    if not args.no_cache and os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    print(f"Generating embeddings using {args.backend}/{args.model}...")
    start_time = time.time()

    if args.backend == "sentence-transformers":
        model = SentenceTransformer(args.model)
        embeddings = model.encode(texts, show_progress_bar=True)
    elif args.backend == "openai":
        try:
            import openai
            from tenacity import retry, stop_after_attempt, wait_random_exponential

            @retry(
                wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6)
            )
            def get_embedding(text):
                return (
                    openai.embeddings.create(input=text, model=args.model)
                    .data[0]
                    .embedding
                )

            embeddings = []
            for i, text in enumerate(texts):
                if i > 0 and i % 100 == 0:
                    print(f"Processed {i}/{len(texts)} texts")
                embeddings.append(get_embedding(text))
            embeddings = np.array(embeddings)
        except ImportError:
            raise ImportError("Please install openai package: pip install openai")

    elapsed = time.time() - start_time
    print(f"Generated {len(embeddings)} embeddings in {elapsed:.2f}s")

    # Cache embeddings for future use
    np.save(cache_path, embeddings)
    print(f"Saved embeddings to {cache_path}")

    return embeddings


def build_index(embeddings, args):
    dim = embeddings.shape[1]

    if args.index_type == "flat":
        print("Building FLAT index (exact search)")
        index = faiss.IndexFlatL2(dim)
    elif args.index_type == "ivfpq":
        print(
            f"Building IVF-PQ index (nlist={args.nlist}, m={args.m}, nbits={args.nbits})"
        )
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, args.nlist, args.m, args.nbits)
        index.train(embeddings)
    elif args.index_type == "hnsw":
        print(f"Building HNSW index (M={args.hnsw_m})")
        index = faiss.IndexHNSWFlat(dim, args.hnsw_m)

    index.add(embeddings)
    return index


def main():
    args = parse_args()

    # 1. Load product data
    with open(os.path.join(DATA_DIR, "products.json"), "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. Prepare texts and ID mapping
    texts = []
    ids = []
    for item in data:
        title = item.get("title", "")
        desc = item.get("description", "")
        category = item.get("category", "")
        price = item.get("price", "")
        # Include more context for better semantic matching
        text = f"Title: {title}. Description: {desc}. Category: {category}. Price: ${price}"
        texts.append(text)
        ids.append(item["id"])

    # 3. Generate embeddings
    embeddings = get_embeddings(texts, args)
    embeddings = np.array(embeddings, dtype="float32")  # FAISS requires float32

    # 4. Build FAISS index
    index = build_index(embeddings, args)

    # 5. Persist index and ID map into output/
    index_path = os.path.join(OUT_DIR, f"faiss_index_{args.index_type}.idx")
    faiss.write_index(index, index_path)
    with open(os.path.join(OUT_DIR, "id_map.json"), "w", encoding="utf-8") as f:
        json.dump(ids, f, indent=2)

    print(f"Indexed {len(ids)} products → {index_path}")


if __name__ == "__main__":
    main()
