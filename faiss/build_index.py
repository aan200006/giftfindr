import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# 1. Load product data from JSON
with open("./products.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. Prepare texts and ID mapping
texts = []
ids = []
for item in data:
    title = item.get("title", "")
    desc = item.get("description", "")
    # You can also include category or price in the embedded text if desired:
    # meta = f"Category: {item.get('category', '')} | Price: ${item.get('price', '')}"
    text = f"{title} — {desc}"
    texts.append(text)
    ids.append(item["id"])

# 3. Generate embeddings
model = SentenceTransformer("all-mpnet-base-v2")
embeddings = model.encode(texts, show_progress_bar=True)
embeddings = np.array(embeddings, dtype="float32")  # FAISS requires float32

# 4. Build FAISS index (exact L2 for prototype)
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# 5. Persist index and ID map
faiss.write_index(index, "faiss_index.idx")
with open("id_map.json", "w", encoding="utf-8") as f:
    json.dump(ids, f, indent=2)

print(f"Indexed {len(ids)} products into 'faiss_index.idx'")
