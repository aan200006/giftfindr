import os
from dotenv import load_dotenv

load_dotenv()
import json
import argparse
import numpy as np
import faiss
import time


def parse_args():
    p = argparse.ArgumentParser(
        description="Build FAISS index with ST or OpenAI embeddings, choose index type"
    )
    p.add_argument(
        "--backend",
        choices=["st", "openai"],
        default="st",
        help="Embedding backend: 'st' = sentence-transformers; 'openai' = OpenAI embeddings",
    )
    p.add_argument(
        "--model",
        type=str,
        default="all-mpnet-base-v2",
        help="ST model name or OpenAI embedding model (e.g. 'text-embedding-ada-002')",
    )
    p.add_argument(
        "--index-type",
        choices=["flat", "ivfpq", "hnsw"],
        default="flat",
        help="flat = exact L2, ivfpq = IVF‑PQ, hnsw = HNSW‑Flat",
    )
    p.add_argument(
        "--nlist", type=int, default=100, help="IVF: number of Voronoi cells"
    )
    p.add_argument("--m", type=int, default=8, help="IVF‑PQ: number of subquantizers")
    p.add_argument("--nbits", type=int, default=8, help="IVF‑PQ: bits per codebook")
    p.add_argument(
        "--hnsw-m", type=int, default=32, help="HNSW: number of neighbors per node"
    )
    return p.parse_args()


def load_data(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_st_embeddings(texts, model_name):
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(model_name)
    return np.array(m.encode(texts, show_progress_bar=True), dtype="float32")


def get_openai_embeddings(texts, model_name):
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("VITE_OPENAI_API_KEY"))

    embs = []
    for chunk_start in range(0, len(texts), 50):
        chunk = texts[chunk_start : chunk_start + 50]
        resp = client.embeddings.create(model=model_name, input=chunk)
        embs.extend(np.array([e.embedding for e in resp.data], dtype="float32"))
    return np.vstack(embs)


def main():
    args = parse_args()
    BASE = os.path.dirname(__file__)
    DATA = os.path.abspath(os.path.join(BASE, "../data/products.json"))
    OUT = os.path.abspath(os.path.join(BASE, "../output"))
    os.makedirs(OUT, exist_ok=True)

    data = load_data(DATA)
    texts, ids = [], []
    for item in data:
        txt = f"{item['title']} — {item.get('description','')}"
        texts.append(txt)
        ids.append(item["id"])

    cache_file = os.path.join(OUT, f"embeddings_{args.backend}.npy")
    if os.path.exists(cache_file):
        embeddings = np.load(cache_file)
        print(f"Loaded cached embeddings ({embeddings.shape})")
    else:
        if args.backend == "st":
            embeddings = get_st_embeddings(texts, args.model)
        else:
            embeddings = get_openai_embeddings(texts, args.model)
        np.save(cache_file, embeddings)
        print(f"Saved embeddings cache → {cache_file}")

    # log stats
    norms = np.linalg.norm(embeddings, axis=1)
    print(
        f"Embedding dims: {embeddings.shape}; norm min/max: {norms.min():.3f}/{norms.max():.3f}"
    )

    # build and save FAISS index
    dim = embeddings.shape[1]

    # build chosen index
    start = time.time()
    if args.index_type == "flat":
        index = faiss.IndexFlatL2(dim)

    elif args.index_type == "ivfpq":
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, args.nlist, args.m, args.nbits)
        index.train(embeddings)  # must train before add()

    else:  # hnsw
        index = faiss.IndexHNSWFlat(dim, args.hnsw_m)

    index.add(embeddings)
    dur = time.time() - start
    print(f"Built {args.index_type} index in {dur:.1f}s")

    # write out
    fname = f"faiss_{args.index_type}.idx"
    path = os.path.join(OUT, fname)
    faiss.write_index(index, path)
    with open(os.path.join(OUT, "id_map.json"), "w", encoding="utf-8") as f:
        json.dump(ids, f, indent=2)

    print(f"Saved index → {path}")


if __name__ == "__main__":
    main()
