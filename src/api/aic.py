import json
import re
from datetime import datetime, timezone
from pathlib import Path

import requests


BASE_URL = "https://api.artic.edu/api/v1/artworks"


def _sanitize_query_for_filename(query):
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", query.strip().lower())
    cleaned = cleaned.strip("_")
    return cleaned or "query"


def _deduplicate_artworks(results_by_query, search_queries=None):
    """Return a deduplicated list of artwork records, with a per-artwork
    matched_queries list computed from the queries that surfaced each artwork.
    """
    artworks_by_id = {}

    for query in search_queries or list(results_by_query.keys()):
        query_results = results_by_query.get(query, [])
        for artwork in query_results:
            artwork_id = artwork.get("id")
            if artwork_id not in artworks_by_id:
                artworks_by_id[artwork_id] = {
                    "artwork": artwork,
                    "matched_queries": [],
                }
            if query not in artworks_by_id[artwork_id]["matched_queries"]:
                artworks_by_id[artwork_id]["matched_queries"].append(query)

    return list(artworks_by_id.values())


def _normalize_artwork_records(unique_artworks, search_queries=None):
    """Normalize an existing unique_artworks collection into the same
    dedicated shape used in the multi-query API: each record has an
    `artwork` payload and a `matched_queries` list.
    """
    normalized = []
    for item in unique_artworks:
        if isinstance(item, dict) and "artwork" in item:
            normalized.append(item)
        elif isinstance(item, dict) and "id" in item:
            normalized.append({
                "artwork": item,
                "matched_queries": list(search_queries or []),
            })
        else:
            normalized.append(item)
    return normalized


def save_artworks_response(payload, query=None, limit=10, page=1, output_dir=None, search_queries=None):
    """Persist an AIC artworks search payload into a single JSON file.

    The helper accepts either:
    - an already-shaped envelope containing search_queries and unique_artworks,
    - or a plain results_by_query mapping that needs to be converted.
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[2] / "data" / "raw" / "aic"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if isinstance(payload, dict) and "unique_artworks" in payload:
        stored_queries = search_queries if search_queries is not None else payload.get("search_queries")
        stored_queries = list(stored_queries or [])
        file_payload = {
            "search_queries": stored_queries,
            "unique_artworks": _normalize_artwork_records(payload["unique_artworks"], stored_queries),
        }
    elif search_queries is not None:
        stored_queries = list(search_queries)
        file_payload = {
            "search_queries": stored_queries,
            "unique_artworks": _deduplicate_artworks(payload, stored_queries),
        }
    else:
        stored_queries = [query] if query else []
        file_payload = {
            "search_queries": stored_queries,
            "unique_artworks": _deduplicate_artworks({query or "query": payload}, stored_queries),
        }

    safe_query = _sanitize_query_for_filename("_".join(stored_queries or ["all_artworks"]))
    filename = f"search_multiple_queries_{safe_query}_{timestamp}.json"
    output_path = output_dir / filename

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(file_payload, file, indent=2, ensure_ascii=False)

    return output_path


def search_artworks(query=None, page=1, limit=10, output_dir=None):

    params = {
        "limit": limit,
        "page": page,
        "fields": (
            "id,"
            "title,"
            "alt_text,"
            "description,"
            "short_description,"
            "subject_titles,"
            "theme_titles,"
            "style_titles,"
            "image_id,"
            "is_public_domain"
        )
    }

    url = BASE_URL

    # If the query is not empty or None, add it to the parameters.
    if query is not None and query != "":
        params["q"] = query
        url = f"{BASE_URL}/search"

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()

    payload = response.json()

    return payload


def search_all_artworks(query=None, max_pages=5, limit=10):
    first_page = search_artworks(query, page=1, limit=limit)
    all_data = list(first_page["data"])
    total_pages = first_page["pagination"]["total_pages"]


    total_pages = min(total_pages, max_pages)

    for page in range(2, total_pages + 1):
        page_data = search_artworks(query, page=page, limit=limit)
        all_data.extend(page_data["data"])

    return all_data


# search_multiple_queries allows searching for multiple queries and returns
# a deduplicated mapping of artwork IDs to an artwork payload and its
# matched_queries metadata.
def search_multiple_queries(queries, max_pages=None, limit=10, output_dir=None):
    all_results = {}

    for query in queries:
        results = search_all_artworks(query, max_pages=max_pages, limit=limit)
        all_results[query] = results

    artworks_by_id = {}

    for query in queries:
        for artwork in all_results[query]:
            artwork_id = artwork.get("id")
            if artwork_id not in artworks_by_id:
                artworks_by_id[artwork_id] = {
                    "artwork": artwork,
                    "matched_queries": [],
                }
            if query not in artworks_by_id[artwork_id]["matched_queries"]:
                artworks_by_id[artwork_id]["matched_queries"].append(query)

    save_artworks_response(
        {
            "search_queries": list(queries),
            "unique_artworks": list(artworks_by_id.values()),
        },
        search_queries=queries,
        output_dir=output_dir,
    )

    return artworks_by_id