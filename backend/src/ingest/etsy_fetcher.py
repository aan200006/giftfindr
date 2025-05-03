import requests
import json, os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = Path(__file__).parent / "base_queries.json"
with open(CONFIG_PATH, encoding="utf-8") as f:
    query_config = json.load(f)

# flatten into one list (and dedupe if you like)
base_queries = set(query for lst in query_config.values() for query in lst)

ETSY_KEY = os.getenv("ETSY_API_KEY")  # put your key in .env

if not ETSY_KEY:
    raise ValueError(
        "Etsy API key not found. Please set the ETSY_API_KEY environment variable."
    )

BASE_URL = "https://api.etsy.com/v3/application/listings/active"


def fetch_etsy(query, limit=50):
    r = requests.get(
        BASE_URL,
        headers={"x-api-key": ETSY_KEY},
        params={"keywords": query, "limit": limit},
    )
    print(f"Fetching {query} from Etsy. Status Code: {r.status_code}")
    return r.json().get("results", [])


def fetch_etsy_items(limit=50):
    all_items = {}

    # Try to load existing products
    products_path = Path(__file__).parent.parent.parent / "data" / "products.json"
    existing_products = {}
    if products_path.exists():
        try:
            with open(products_path, encoding="utf-8") as f:
                existing_products = {p["id"]: p for p in json.load(f)}
            print(f"Loaded {len(existing_products)} existing products")
        except Exception as e:
            print(f"Warning: Could not load existing products: {e}")

    # Add existing products to our collection
    all_items.update(existing_products)

    # Fetch new items
    for q in base_queries:
        items = fetch_etsy(q, limit=limit)
        for it in items:
            all_items[it["listing_id"]] = {
                "id": it["listing_id"],
                "title": it["title"],
                "description": it["description"],
                "category": (it.get("tags", [])[-1] if it.get("tags") else ""),
                "price": float(it["price"]["amount"]) / 100.0,
                "url": it["url"],
            }

    print(
        f"Collected total of {len(all_items)} items ({len(all_items) - len(existing_products)} new)"
    )
    return list(all_items.values())
