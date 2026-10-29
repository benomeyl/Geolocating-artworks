"""
Art Institute of Chicago (AIC) data access: public API, local dump and storage.

Corpora are stored as {"search_queries": [...], "unique_artworks": [...]}
where each record is {"artwork": {...}, "matched_queries": [...]}.
`enrich_records_from_dump` adds the context fields used by the place model
from the local dump (data/raw/artic-api-data), without network calls.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import requests


BASE_URL = "https://api.artic.edu/api/v1/artworks"
DEFAULT_DUMP_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "artic-api-data" / "json" / "artworks"
# Fields requested from the search API.
SEARCH_FIELDS = (
    "id",
    "title",
    "alt_text",
    "description",
    "short_description",
    "subject_titles",
    "theme_titles",
    "style_titles",
    "alt_titles",
    "artwork_type_title",
    "place_of_origin",
    "artist_display",
    "exhibition_history",
    "provenance_text",
    "credit_line",
    "image_id",
    "is_public_domain",
)


def _sanitize_query_for_filename(query):
    """Turn a query into a safe file-name fragment ("query" if nothing is left)."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", query.strip().lower())
    cleaned = cleaned.strip("_")
    return cleaned or "query"


def _group_by_artwork_id(results_by_query, search_queries=None):
    """Map artwork id to {"artwork", "matched_queries"} from per-query results.

    The first payload seen for an artwork is kept; matched_queries lists the
    queries that returned it, in query order.
    """
    artworks_by_id = {}
    for query in search_queries or list(results_by_query.keys()):
        for artwork in results_by_query.get(query, []):
            record = artworks_by_id.setdefault(artwork.get("id"), {"artwork": artwork, "matched_queries": []})
            if query not in record["matched_queries"]:
                record["matched_queries"].append(query)
    return artworks_by_id


def _deduplicate_artworks(results_by_query, search_queries=None):
    """Return a deduplicated list of artwork records with their matched_queries."""
    return list(_group_by_artwork_id(results_by_query, search_queries).values())


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


def _matches_query(artwork, query):
    """Return True when the query matches the title or the main textual fields."""
    if not query:
        return True
    needle = str(query).strip().lower()
    if not needle:
        return True

    haystack = " ".join(
        str(value or "")
        for value in (
            artwork.get("title"),
            artwork.get("alt_titles"),
            artwork.get("place_of_origin"),
            artwork.get("description"),
            artwork.get("short_description"),
            artwork.get("artist_title"),
            artwork.get("artist_display"),
            " ".join(artwork.get("subject_titles") or []),
            " ".join(artwork.get("theme_titles") or []),
            " ".join(artwork.get("style_titles") or []),
        )
    ).lower()
    return needle in haystack


# Context fields added to AIC records for the place model. Only alt_titles is
# a source of candidates; the others are clues about a candidate found
# elsewhere (object type, place of origin, artist biography, provenance and
# exhibitions). latitude, longitude and gallery_title locate the museum room
# in Chicago: they are deliberately excluded.
CONTEXT_FIELDS = (
    "alt_titles",
    "artwork_type_title",
    "place_of_origin",
    "artist_display",
    "exhibition_history",
    "provenance_text",
    "credit_line",
)


def enrich_records_from_dump(records, dump_dir=None, fields=CONTEXT_FIELDS):
    """Fill missing context fields of AIC records from the local dump, in place.

    Records whose id is not an AIC integer id (Wikidata, CSV imports) are
    left untouched. Returns the number of records found in the dump.
    """
    dump_dir = Path(dump_dir) if dump_dir is not None else DEFAULT_DUMP_DIR
    if not dump_dir.is_dir():
        return 0
    enriched = 0
    for record in records:
        artwork = record.get("artwork", record)
        artwork_id = str(artwork.get("id", ""))
        file_path = dump_dir / f"{artwork_id}.json"
        if not artwork_id.isdigit() or not file_path.is_file():
            continue
        with file_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        for field in fields:
            if artwork.get(field) in (None, "", []) and payload.get(field) not in (None, "", []):
                artwork[field] = payload[field]
        enriched += 1
    return enriched


def load_artworks_from_dump(dump_dir=None, limit=None, queries=None):
    """Load artwork records directly from the local AIC dump directory.

    The dump is a folder containing one JSON file per artwork, usually under
    data/raw/artic-api-data/json/artworks. This avoids making network calls.
    """
    dump_dir = Path(dump_dir) if dump_dir is not None else DEFAULT_DUMP_DIR
    if not dump_dir.exists():
        raise FileNotFoundError(f"AIC dump directory not found: {dump_dir}")

    artworks = []
    for file_path in sorted(dump_dir.glob("*.json")):
        with file_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        if not isinstance(payload, dict):
            continue
        if "id" not in payload:
            continue

        if queries and not any(_matches_query(payload, query) for query in queries):
            continue
        artworks.append(payload)

        if limit is not None and len(artworks) >= limit:
            break

    return artworks


def get_corpus_from_dump(queries=None, output_dir=None, dump_dir=None, limit=None):
    """Build the same unique_artworks structure as get_corpus() using a local dump."""
    artworks = load_artworks_from_dump(dump_dir=dump_dir, limit=limit, queries=queries)
    unique_artworks = {}

    for artwork in artworks:
        artwork_id = artwork.get("id")
        if artwork_id is None:
            continue
        if artwork_id not in unique_artworks:
            unique_artworks[artwork_id] = {
                "artwork": artwork,
                "matched_queries": [],
            }

        if queries:
            matched = [query for query in queries if _matches_query(artwork, query)]
            for query in matched:
                if query not in unique_artworks[artwork_id]["matched_queries"]:
                    unique_artworks[artwork_id]["matched_queries"].append(query)

    if output_dir is not None:
        save_artworks_response(
            {
                "search_queries": list(queries or []),
                "unique_artworks": list(unique_artworks.values()),
            },
            search_queries=list(queries or []),
            output_dir=output_dir,
        )

    return unique_artworks


def search_artworks(query=None, page=1, limit=10, output_dir=None):
    """Fetch one page of artworks from the AIC API (full-text search when `query` is set).

    `output_dir` is accepted for backwards compatibility and ignored.
    """
    params = {"limit": limit, "page": page, "fields": ",".join(SEARCH_FIELDS)}
    url = BASE_URL
    if query is not None and query != "":
        params["q"] = query
        url = f"{BASE_URL}/search"

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def search_all_artworks(query=None, max_pages=5, limit=10):
    """Fetch up to `max_pages` pages of results and return the artwork list."""
    first_page = search_artworks(query, page=1, limit=limit)
    all_data = list(first_page["data"])
    total_pages = min(first_page["pagination"]["total_pages"], max_pages)
    for page in range(2, total_pages + 1):
        all_data.extend(search_artworks(query, page=page, limit=limit)["data"])
    return all_data


def search_multiple_queries(queries, max_pages=None, limit=10, output_dir=None):
    """Run several searches, save one deduplicated JSON file and return {id: record}.

    Each record holds the artwork payload and the queries that returned it.
    """
    all_results = {
        query: search_all_artworks(query, max_pages=max_pages, limit=limit)
        for query in queries
    }
    artworks_by_id = _group_by_artwork_id(all_results, queries)
    save_artworks_response(
        {
            "search_queries": list(queries),
            "unique_artworks": list(artworks_by_id.values()),
        },
        search_queries=queries,
        output_dir=output_dir,
    )

    return artworks_by_id


def get_corpus(queries=None, max_pages=5, limit=10, output_dir="data/raw/aic"):
    """Compatibility wrapper around the API-based corpus extraction.

    When local dump data is available, prefer get_corpus_from_dump().
    """
    if queries is None:
        artworks = search_all_artworks(max_pages=max_pages, limit=limit)
        unique_artworks = {
            artwork["id"]: {
                "artwork": artwork,
                "matched_queries": [],
            }
            for artwork in artworks
        }
        search_queries = None
    else:
        search_queries = list(queries)
        unique_artworks = search_multiple_queries(
            search_queries,
            max_pages=max_pages,
            limit=limit,
            output_dir=output_dir,
        )

    if queries is None:
        save_artworks_response(
            {
                "search_queries": None,
                "unique_artworks": list(unique_artworks.values()),
            },
            search_queries=None,
            output_dir=output_dir,
        )

    return unique_artworks

def keep_useful_fields(artwork):
    """Reduce an AIC artwork to the fields used by the project."""
    return {
        "id": artwork.get("id"),
        "title": artwork.get("title"),
        "description": artwork.get("description"),
        "short_description": artwork.get("short_description"),
        "subject_titles": artwork.get("subject_titles") or [],
        "theme_titles": artwork.get("theme_titles") or [],
        "style_titles": artwork.get("style_titles") or [],
        "artist_title": artwork.get("artist_title"),
        **{field: artwork.get(field) for field in CONTEXT_FIELDS},
        "date_display": artwork.get("date_display"),
        "image_id": artwork.get("image_id"),
        "is_public_domain": artwork.get("is_public_domain"),
    }


def export_useful_artworks(unique_artworks, output_path):
    """Write the records reduced by `keep_useful_fields` to a corpus JSON file."""
    payload = {
        "search_queries": [],
        "unique_artworks": [
            {
                "artwork": keep_useful_fields(record.get("artwork", record)),
                "matched_queries": record.get("matched_queries", []),
            }
            for record in unique_artworks
        ],
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return output_path