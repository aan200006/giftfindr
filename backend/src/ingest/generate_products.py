#!/usr/bin/env python3
import argparse
import json
import os
from etsy_fetcher import fetch_etsy_items

# from ingest.ebay_fetcher import fetch_ebay_items
from utils.common import dedupe_by_id, ensure_dir


def main(sources, out_path, per_source_limit):
    # Load existing products if available
    existing_products = []
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                existing_products = json.load(f)
            print(
                f"🔄 Loaded {len(existing_products)} existing products from {out_path}"
            )
        except Exception as e:
            print(f"⚠️ Warning: Could not load existing products: {e}")

    # Fetch new items
    catalog = existing_products.copy()

    if "etsy" in sources:
        print(f"🛠️  Fetching up to {per_source_limit} items from Etsy…")
        new_items = fetch_etsy_items(limit=per_source_limit)
        # Only add items not already in catalog
        existing_ids = {item["id"] for item in existing_products}
        new_items_to_add = [
            item for item in new_items if item["id"] not in existing_ids
        ]
        catalog.extend(new_items_to_add)
        print(f"➕ Adding {len(new_items_to_add)} new items from Etsy")

    # if "ebay" in sources:
    #     print(f"🛠️  Fetching up to {per_source_limit} items from eBay…")
    #     catalog += fetch_ebay_items(limit=per_source_limit)

    # Final deduplication just to be safe
    unique = dedupe_by_id(catalog, key="id")
    ensure_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(unique, f, indent=2)

    print(
        f"✅ Saved {len(unique)} unique products to {out_path} ({len(unique) - len(existing_products)} new)"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate products.json from APIs")
    p.add_argument(
        "--sources",
        nargs="+",
        choices=["etsy", "ebay"],
        default=["etsy", "ebay"],
        help="Which catalogs to pull",
    )
    p.add_argument("--limit", type=int, default=3, help="Max items to fetch per source")
    p.add_argument(
        "--out", type=str, default="../products.json", help="Output JSON file path"
    )
    args = p.parse_args()
    main(args.sources, args.out, args.limit)
