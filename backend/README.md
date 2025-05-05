# Giftfindr: FAISS-based Gift Recommendation

Giftfindr is a prototype that turns natural-language gift queries into personalized recommendations using a vector search (FAISS) over a small product catalog, exposed via a simple API and interactive tools.

---

## Repository Structure

```
backend/
├── data/
│   ├── products.json        # Sample product
│   └── gold.json            # Gold-standard queries and relevant item IDs
├── output/
│   ├── faiss_index_*.idx    # Serialized FAISS index (built by build_index.py)
│   ├── id_map.json          # Maps index positions back to product IDs
│   ├── eval_results.json    # Evaluation metrics JSON output
│   └── eval_results.csv     # Evaluation metrics CSV output
│   └── embeddings_*.npy     # Cached embeddings from different backends
│   └── faiss_tuning/        # Results from FAISS parameter tuning
├── src/
│   ├── build_index.py       # Builds FAISS index from products.json
│   ├── search_demo.py       # REPL for semantic search over the FAISS index
│   ├── evaluate.py          # Computes P@K, R@K, MRR@K, nDCG@K against gold.json
│   ├── server.py            # Flask server providing API endpoints
│   └── ingest/
│       ├── generate_products.py  # Fetches products from APIs and merges with existing
│       └── etsy_fetcher.py       # Handles API calls to Etsy's product catalog
├── labeling/
│   ├── index.html           # Web UI for labeling and searching
│   └── label.js             # JavaScript for the labeling/search interface
├── requirements.txt         # Python dependencies
└── .env                     # API keys and environment variables (not committed)
```

---

## Setup

1.  Clone the repo and navigate into the `backend/` folder.
2.  (Optional) Create and activate a Python virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Create a `.env` file in the `backend/` directory and add your OpenAI API key:
    ```env
    OPENAI_API_KEY=your_openai_api_key_here
    # Add ETSY_API_KEY if using Etsy ingestion
    # ETSY_API_KEY=your_etsy_api_key_here
    ```

---

## Running the Server

The Flask server provides the API endpoints for search and chatbot functionality.

```bash
python src/server.py
```

The server will run on `http://localhost:5001` by default.

### API Endpoints

- `POST /api/search`: Performs semantic search. Expects JSON body with `recipient`, `occasion`, `interests`, and optional `price`, `k`. Returns a list of product results.
- `POST /api/chatbot`: Basic chatbot interaction using OpenAI. Expects `messages` array.
- `POST /api/chatbot/json`: Chatbot interaction that attempts to extract structured data (recipient, age, interests, budget, occasion) from the conversation. Expects `messages` and optional `previousStructuredData`.

---

## Complete Workflow

### 1. Data Ingestion (Optional)

Generate or update your product catalog using external APIs (e.g., Etsy):

```bash
# Make sure ETSY_API_KEY is in your .env file
python src/ingest/generate_products.py --sources etsy --items-per-query 5 --num-queries 20 --out data/products.json
```

This will:

- Fetch products from the Etsy API
- Merge with any existing products in the output file
- Deduplicate by product ID
- Save the updated catalog

### 2. Build the FAISS Index

Generate embeddings and create a searchable index:

```bash
python src/build_index.py --index-type flat --model text-embedding-3-small --backend openai
```

This will:

- Load `data/products.json` and generate embeddings via OpenAI API.
- Build an exact L2 FAISS index and save it to `output/faiss_index_flat.idx`.
- Write the ID mapping to `output/id_map.json`.

We support three index types:

- **flat**: exact L2 (`IndexFlatL2`)
- **ivfpq**: IVF‑PQ (`IndexIVFPQ`), tuned via `--nlist`, `--m`, `--nbits`
- **hnsw**: HNSW (`IndexHNSWFlat`), tuned via `--hnsw-m`

Example:

```bash
# exact with OpenAI embeddings (default)
python src/build_index.py --index-type flat

# Using Sentence Transformers instead
python src/build_index.py --backend sentence-transformers --model all-mpnet-base-v2

# IVF-PQ with OpenAI
python src/build_index.py --index-type ivfpq --nlist 200 --m 16 --nbits 8

# HNSW with OpenAI
python src/build_index.py --index-type hnsw --hnsw-m 64
```

You can also choose between embeddings from Sentence Transformers or OpenAI:

```bash
# OpenAI embedding model
python src/build_index.py --backend openai --model text-embedding-3-small
```

### 3. Run the Backend Server

Start the Flask server to enable API access for the labeling tool and other clients:

```bash
python src/server.py
```

### 4. Create and Label Gold Standard Data

Use the integrated labeling and search tool to create/refine a gold standard dataset for evaluation:

1.  With the server running, start a local HTTP server in the `backend/` directory:
    ```bash
    # Using Python's built-in server (run in a separate terminal)
    python -m http.server 8000
    ```
2.  Navigate to `http://localhost:8000/labeling/` in your browser.
3.  **Gold.json Labeler Tab:**
    - Review existing queries from `data/gold.json`.
    - Check/uncheck products to mark them as relevant/irrelevant for the current query.
    - Use filters (text, price) to narrow down the product list.
    - Navigate with Prev/Next buttons or ←/→ arrow keys.
    - Use "Remove Query" or Del key to discard a query.
    - Click "Export gold.json" to download the modified data. **You must manually move the downloaded `gold.json` file to the `backend/data/` directory, overwriting the old one.**
4.  **FAISS Search Tab:**
    - Enter search criteria (recipient, occasion, interests, max price).
    - Click "Search" to query the `/api/search` endpoint.
    - Results are displayed with checkboxes. Select the products relevant to your query.
    - Click "Save Query to Gold" to add the current query and selected product IDs to the dataset being built in the browser. This new entry will be included when you export using the "Export gold.json" button on the "Gold.json Labeler" tab.

### 5. Try Semantic Search (CLI Demo)

Launch the interactive command-line demo:

```bash
python src/search_demo.py --index faiss_index_flat.idx --model text-embedding-3-small --backend openai
```

Then enter a natural-language query (e.g. "gift for a yoga lover") and see the top-5 results printed with title, category, price, and distance.

### 6. Automated Evaluation

To measure retrieval performance against your curated `data/gold.json`:

```bash
python src/evaluate.py --index faiss_index_flat.idx --k 1 5 10 20 --model text-embedding-3-small --backend openai --products products.json --id-map id_map.json --gold gold.json
```

**Important**: Make sure to use the same model and backend that were used to build the index!

This computes metrics against your `data/gold.json`:

- **Precision@K**: Fraction of top-K results that are relevant.
- **Recall@K**: Fraction of all relevant items that appear in top-K.
- **MRR@K**: Mean Reciprocal Rank (position of first relevant item).
- **nDCG@K**: Normalized Discounted Cumulative Gain (quality of ranking).

The results are saved to `output/eval_results.json` and `output/eval_results.csv`.

### Sample Output

```
Retrieval Evaluation Metrics:
    Precision  Recall     MRR    nDCG
K
1      0.7000  0.6833  0.7000  0.7000
5      0.2067  0.9333  0.8167  0.8572
10     0.1100  0.9556  0.8222  0.8704
20     0.0550  0.9667  0.8222  0.8761
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
