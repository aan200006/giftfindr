# Giftfindr: FAISS-based Gift Recommendation

Giftfindr is a prototype that turns natural-language gift queries into personalized recommendations using a vector search (FAISS) over a small product catalog.

---

## Repository Structure

```
faiss/
├── data/
│   ├── products.json        # Sample product catalog
│   ├── gold.json            # Gold-standard queries and relevant item IDs
│   └── history.json         # Labeling history tracking file
├── output/
│   ├── faiss_index.idx      # Serialized FAISS index (built by build_index.py)
│   ├── id_map.json          # Maps index positions back to product IDs
│   ├── eval_results.json    # Evaluation metrics JSON output
│   └── eval_results.csv     # Evaluation metrics CSV output
│   └── embeddings_*.npy     # Cached embeddings from different backends
├── src/
│   ├── build_index.py       # Builds FAISS index from products.json
│   ├── search_demo.py       # REPL for semantic search over the FAISS index
│   ├── evaluate.py          # Computes P@K, R@K, MRR@K, nDCG@K against gold.json
│   └── ingest/
│       ├── generate_products.py  # Fetches products from APIs and merges with existing
│       └── etsy_fetcher.py       # Handles API calls to Etsy's product catalog
├── labeling/
│   ├── index.html           # Web UI for labeling relevant products for queries
│   └── label.js             # JavaScript for the labeling interface
└── requirements.txt         # Python dependencies (faiss-cpu, sentence-transformers, openai, pandas)
```

---

## Setup

1. Clone the repo and navigate into the `faiss/` folder.
2. (Optional) Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Complete Workflow

### 1. Data Ingestion

Generate or update your product catalog:

```bash
python src/ingest/generate_products.py --sources etsy --limit 5 --out data/products.json
```

This will:

- Fetch products from the Etsy API
- Merge with any existing products in the output file
- Deduplicate by product ID
- Save the updated catalog

### 2. Build the FAISS Index

Generate embeddings and create a searchable index:

```bash
python src/build_index.py
```

This will:

- Load `data/products.json` and generate embeddings via SentenceTransformers.
- Build an exact L2 FAISS index and save it to `output/faiss_index.idx`.
- Write the ID mapping to `output/id_map.json`.

We support three index types:

- **flat**: exact L2 (`IndexFlatL2`)
- **ivfpq**: IVF‑PQ (`IndexIVFPQ`), tuned via `--nlist`, `--m`, `--nbits`
- **hnsw**: HNSW (`IndexHNSWFlat`), tuned via `--hnsw-m`

Example:

```bash
# exact
python src/build_index.py --index-type flat

# IVF-PQ
python src/build_index.py --index-type ivfpq --nlist 200 --m 16 --nbits 8

# HNSW
python src/build_index.py --index-type hnsw --hnsw-m 64
```

You can also choose between embeddings from Sentence Transformers or OpenAI:

```bash
# OpenAI embedding model
python src/build_index.py --backend openai --model text-embedding-3-small
```

### 3. Create and Label Gold Standard Data

Use the labeling tool to create a gold standard dataset for evaluation:

1. Start a local server in the project folder:

   ```bash
   # Using Python's built-in server
   python -m http.server
   ```

2. Navigate to `http://localhost:8000/labeling/` in your browser
3. Add relevant product IDs for each query
4. Export the labeled data to `gold.json` and `history.json` when done
5. Move these files to the `data/` directory

### 4. Try Semantic Search

Launch the interactive demo:

```bash
python src/search_demo.py
```

Then enter a natural-language query (e.g. "gift for a yoga lover") and see the top-5 results printed with title, category, price, and distance.

### 5. Automated Evaluation

To measure retrieval performance, run:

```bash
python src/evaluate.py --k 1 5 10 20
```

This computes metrics against your `data/gold.json`:

- **Precision@K**: Fraction of top-K results that are relevant.
- **Recall@K**: Fraction of all relevant items that appear in top-K.
- **MRR@K**: Mean Reciprocal Rank (position of first relevant item).
- **nDCG@K**: Normalized Discounted Cumulative Gain (quality of ranking).

### Sample Output

```
Retrieval Evaluation Metrics:
    Precision  Recall     MRR    nDCG
K
1      0.7000  0.6833  0.7000  0.7000
5      0.2067  0.9333  0.8167  0.8572
10     0.1100  0.9556  0.8222  0.8704
```

#### What This Means

- **At K=1**:

  - **Precision = 0.70**: 70% of the time, the very top result was relevant.
  - **Recall = 0.68**: On average, 68% of all relevant items appear within the first slot.
  - **MRR = 0.70**: The first relevant item typically appears at position ~1.4 (1/0.7).
  - **nDCG = 0.70**: Ranking quality is 70% of ideal when considering position discounts.

- **At K=5**:

  - **Recall = 0.93**: Nearly all relevant items are found in your top 5.
  - **Precision = 0.21**: Only ~1 in 5 of those top 5 are relevant, indicating room to improve ranking or filtering.
  - **MRR = 0.82**, **nDCG = 0.86**: Relevant items tend to appear near the top of the list.

- **At K=10**:
  - **Recall = 0.96**: Almost all relevant items are within top 10.
  - **Precision = 0.11**: Over half of the suggestions are non-relevant.
  - **MRR = 0.82**, **nDCG = 0.87**: Similar ranking quality as K=5.

**Key takeaway**: High recall shows the index covers relevant items, but low precision signals the need for better ranking, additional filters, or improved embedding quality.

---

## Next Steps & Improvements

- **Tune FAISS**: Try approximate indices (IVF, HNSW) and adjust `nlist`/`nprobe` for a precision-recall trade-off.
- **Enhance embeddings**: Fine-tune SentenceTransformer on your product descriptions or use OpenAI embeddings.
- **Filter by metadata**: Post-process to enforce budget or category constraints.
- **Error analysis**: Slice queries by slots (budget, recipient) to target specific failure modes.
- **Add more sources**: Integrate additional product APIs beyond Etsy.

---

## License

MIT © James Han, Anna Ngo
