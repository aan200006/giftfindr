import requests
import json, os
import random
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = Path(__file__).parent / "base_queries.json"
with open(CONFIG_PATH, encoding="utf-8") as f:
    query_config = json.load(f)

# Instead of flattening, we'll use the template for structured queries
query_template = query_config.get(
    "query_template", "{occasion} gift, {recipient}, {interest}"
)
recipients = query_config.get("recipient", [])
occasions = query_config.get("occasion", [])
interests = query_config.get("interest", [])

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
    data = r.json()
    results = data.get("results", [])
    count = data.get("count", 0)
    print(f"Found {len(results)} results out of {count} total for query: {query}")
    return r.json().get("results", [])


def generate_random_queries(n=50):
    """Generate n random queries using the query template"""
    queries = []
    for _ in range(n):
        recipient = random.choice(recipients)
        occasion = random.choice(occasions)
        interest = random.choice(interests)
        query = query_template.format(
            recipient=recipient, occasion=occasion, interest=interest
        )
        queries.append(query)
    return queries


def fetch_etsy_items(num_queries=50, items_per_query=5):
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

    # Load existing gold data
    gold_path = Path(__file__).parent.parent.parent / "data" / "gold.json"
    gold_data = []
    if gold_path.exists():
        try:
            with open(gold_path, encoding="utf-8") as f:
                gold_data = json.load(f)
            print(f"Loaded {len(gold_data)} existing gold entries")
        except Exception as e:
            print(f"Warning: Could not load existing gold data: {e}")

    # Track queries by a dictionary for faster lookups
    gold_dict = {entry["query"]: entry for entry in gold_data}

    # Generate random queries based on the template
    queries = generate_random_queries(n=num_queries)

    # Fetch new items using randomly generated queries
    for q in queries:
        items = fetch_etsy(q, limit=items_per_query)

        # Collect item IDs for this query
        relevant_ids = []

        for it in items:
            listing_id = it["listing_id"]
            relevant_ids.append(listing_id)

            all_items[listing_id] = {
                "id": listing_id,
                "title": it["title"],
                "description": it["description"],
                "category": ", ".join(it.get("tags", [])),
                "price": float(it["price"]["amount"]) / 100.0,
                "url": it["url"],
            }

        # Simply update the gold data with this query's results
        # Each query gets its own entry with associated product IDs
        gold_dict[q] = {"query": q, "relevant_ids": relevant_ids}
        print(f"Added/updated query '{q}' with {len(relevant_ids)} relevant items")

    # Save updated gold data
    updated_gold_data = list(gold_dict.values())
    try:
        with open(gold_path, "w", encoding="utf-8") as f:
            json.dump(updated_gold_data, f, indent=2)
        print(f"Updated gold.json with {len(updated_gold_data)} entries")
    except Exception as e:
        print(f"Error saving gold data: {e}")

    print(
        f"Collected total of {len(all_items)} items ({len(all_items) - len(existing_products)} new)"
    )
    return list(all_items.values())
