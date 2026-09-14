import json
import re
from datetime import datetime, timezone
from pathlib import Path

import requests


BASE_URL = "https://api.artic.edu/api/v1/artworks/search"


def _sanitize_query_for_filename(query):
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", query.strip().lower())
    cleaned = cleaned.strip("_")
    return cleaned or "query"


def save_artworks_response(payload, query, limit=10, page=1, output_dir=None):
    """Persist an AIC artworks search payload into a JSON file in data/raw/aic.

    Parameters
    ----------
    payload:
        The parsed JSON response returned by request.get(...).json().
    query:
        Original free-text query used for the search.
    limit:
        Number of artworks requested by the query.
    output_dir:
        Optional destination directory for the output file. Defaults to the
        repository's data/raw/aic directory.
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[2] / "data" / "raw" / "aic"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_query = _sanitize_query_for_filename(query)
    filename = f"search_{safe_query}_limit_{limit}_page_{page}_{timestamp}.json"
    output_path = output_dir / filename

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    return output_path


def search_artworks(query, page=1, limit=10, output_dir=None):
    params = {
        "q": query,
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

    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()

    payload = response.json()
    
    return payload


def search_all_artworks(query, max_pages=None, limit=10):

    first_page = search_artworks(query, page=1, limit=limit)
    save_artworks_response(
                first_page,
                query=query,
                limit=limit,
                page=1,
                output_dir=None,
            )

    all_data = first_page["data"]
    total_pages = first_page["pagination"]["total_pages"]

    if max_pages is not None:
        total_pages = min(total_pages, max_pages)

    for page in range(2, total_pages + 1):
        page_data = search_artworks(query, page=page, limit=limit)

        save_artworks_response(
            page_data,
            query=query,
            limit=limit,
            page=page,
            output_dir=None,
        )

        all_data.extend(page_data["data"])

    return all_data