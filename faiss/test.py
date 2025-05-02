from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Sample gift data
gift_data = [
    "Bluetooth headphones for music lovers",
    "Romantic candlelight dinner package",
    "Educational STEM toy for kids",
    "Luxury leather wallet for men",
    "Custom photo album for couples",
]

# Step 1: Load model and encode descriptions
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(gift_data).astype("float32")

# Step 2: Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Step 3: Search with a query
query = "gift for someone who likes music"
query_vec = model.encode([query]).astype("float32")

D, I = index.search(query_vec, k=3)

# Step 4: Print results
print("Query:", query)
print("Top gift matches:")
for idx in I[0]:
    print("-", gift_data[idx])
