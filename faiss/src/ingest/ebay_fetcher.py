import os
import requests
from dotenv import load_dotenv

# Load .env variables (e.g., EBAY_OAUTH_TOKEN)
load_dotenv()

EBAY_SEARCH_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"


def fetch_ebay_items(query: str, limit: int = 50) -> list[dict]:
    """
    Fetches items from the eBay Browse API matching the given free-text query.
    Requires an environment variable EBAY_OAUTH_TOKEN to be set.
    :param query: Search keywords string
    :param limit: Maximum number of items to return
    :return: List of product dicts with fields: id, title, description, category, price, url
    """
    token = os.getenv("EBAY_OAUTH_TOKEN")
    if not token:
        raise RuntimeError("EBAY_OAUTH_TOKEN not set in environment")

    headers = {"Authorization": f"Bearer {token}"}
    params = {"q": query, "limit": limit}

    response = requests.get(EBAY_SEARCH_URL, headers=headers, params=params)
    response.raise_for_status()
    data = response.json().get("itemSummaries", [])

    items = []
    for it in data:
        items.append(
            {
                "id": it.get("itemId"),
                "title": it.get("title", ""),
                "description": it.get("shortDescription", ""),
                "category": (
                    it.get("categoryPath", "").split(">")[-1]
                    if it.get("categoryPath")
                    else ""
                ),
                "price": (
                    float(it["price"]["value"])
                    if it.get("price") and it["price"].get("value")
                    else 0.0
                ),
                "url": it.get("itemWebUrl", ""),
            }
        )
    return items
