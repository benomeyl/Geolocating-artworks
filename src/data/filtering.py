import re
import pandas as pd
from collections import defaultdict

POSITIVES_LABELS = [
    "view of", "depicting", "depicts", "shows", "showing", "landscape of", "painting of", "painting depicting", "painting showing", "vue sur",
    "vue de", "vue du", "représente", "représentant", "paysage de", "scène à", "peinture de", "peinture représentant", "peinture montrant",
    "photo de", "photographie de", "photographie représentant", "photographie montrant",
]
NEGATIVES_LABELS = [
    "born in", "died in", "studied in", "trained in", "exhibited in", "acquired",
    "né à", "mort à", "formé à", "exposé à", "collection", "purchased", "donated",
    "moved to", "returned to", "during his stay", "painted in", "peint à",
]

FIELDS_WEIGHTS = {
    "title": 0.4,
    "themes": 0.3,
    "subjects": 0.2,
    "description": 0.0,
    "short_description": 0.0,
    "styles": -0.1,  # "Dutch School" and "French Impressionism" rarely indicate depicted places.
}



def _flatten_keywords(keywords):
    if isinstance(keywords, dict):
        flattened = []
        for value in keywords.values():
            flattened.extend(_flatten_keywords(value))
        return flattened
    if isinstance(keywords, (list, tuple, set)):
        return [str(keyword) for keyword in keywords]
    return [str(keywords)]


def _keyword_pattern(keyword):
    escaped_keyword = re.escape(keyword.strip())
    return re.compile(r"(?<!\w)" + escaped_keyword + r"(?!\w)", re.IGNORECASE)


def filter_by_keywords(combined_texts, keywords):
    """
    Filters artworks based on the presence of keywords in the combined text of the artwork.

    Parameters:
    combined_texts (dict): A dictionary mapping artwork IDs to their combined text.
    keywords (dict or list): The nested keyword structure from keywords.json,
        or a flat list of keywords.

    Returns:
    dict: A dictionary containing the filtered artworks and their matching
        keywords, for example:
        {"filtered_artworks": [{"id": 1, "matched_keywords": ["street"]}]}
    """
    keyword_patterns = [
        (keyword, _keyword_pattern(keyword))
        for keyword in _flatten_keywords(keywords)
        if keyword.strip()
    ]
    artworks_df = pd.DataFrame(
        combined_texts.items(),
        columns=["id", "combined_text"],
    )

    artworks_df["matched_keywords"] = artworks_df["combined_text"].fillna("").apply(
        lambda text: [
            keyword
            for keyword, pattern in keyword_patterns
            if pattern.search(text)
        ]
    )

    filtered_artworks = artworks_df.loc[
        artworks_df["matched_keywords"].str.len() > 0,
        ["id", "matched_keywords"],
    ].to_dict(orient="records")

    return {"filtered_artworks": filtered_artworks}

def extract_candidate_artwork(artwork, ner, threshold=0.85):
    """
    Extract detected LOC candidates, including their source field and position
    within that field (required for context scoring).
    """

    fields = {
        "title": artwork.get("title") or "",
        "description": artwork.get("description") or "",
        "short_description": artwork.get("short_description") or "",
        "themes": " ".join(artwork.get("theme_titles") or []),
        "subjects": " ".join(artwork.get("subject_titles") or []),
        "styles": " ".join(artwork.get("style_titles") or []),
    }

    candidates = []
    for field, text in fields.items():
        if not text.strip():
            continue
        results = ner(text)
        for result in results:
            if result["entity_group"] == "LOC" and result["score"] >= threshold:
                candidates.append({
                    "place": result["word"],
                    "score_ner": round(float(result["score"]), 3),
                    "field": field,
                    "start": result["start"],
                    "end": result["end"],
                    "source_text": text,
                })
    return candidates


def local_context(text, start, end, window=60):
    """
    Get the local context around a slice of text.
    """
    return text[max(0, start - window):end + window].lower()

def score_candidate(candidate):
    score = FIELDS_WEIGHTS.get(candidate["field"], 0.0)

    context = local_context(candidate["source_text"], candidate["start"], candidate["end"])
    if any(marker in context for marker in POSITIVES_LABELS):
        score += 0.3
    if any(marker in context for marker in NEGATIVES_LABELS):
        score -= 0.4

    return score

def merge_candidate_places(candidates):
    """
    Merge candidates that refer to the same place (case-insensitive) and
    compute a heuristic score for each unique place.
    """
    groups = defaultdict(list)
    for candidate in candidates:
        place_key = candidate["place"].lower().strip()
        groups[place_key].append(candidate)

    merged_places = []
    for place_key, occurrences in groups.items():
        heuristic_score = max(score_candidate(candidate) for candidate in occurrences)
        found_fields = {candidate["field"] for candidate in occurrences}

        # Add a bonus when the place appears in multiple independent fields.
        if len(found_fields) > 1:
            heuristic_score += 0.15

        merged_places.append({
            "place": occurrences[0]["place"],  # Preserve the original spelling and capitalization.
            "score_ner_max": max(candidate["score_ner"] for candidate in occurrences),
            "score_heuristique": round(heuristic_score, 3),
            "champs": sorted(found_fields),
            "occurrences": occurrences,  # Kept for debugging and traceability.
        })

    return sorted(merged_places, key=lambda place: place["score_heuristique"], reverse=True)


def evaluate_artwork(artwork, ner, threshold=0.85):
    candidates = extract_candidate_artwork(artwork, ner, threshold=threshold)
    merged_places = merge_candidate_places(candidates)
    return merged_places


def compare_heuristic_places_with_annotations(records, heuristic_results, threshold=0.4):
    """Compare heuristic acceptance with singular depicted-place annotations."""
    real_depicted_artworks = []
    heuristic_accepted_artworks = []
    artworks_with_discrepancy = []
    real_depicted_places = 0
    heuristic_accepted_places = 0

    for record in records:
        artwork = record.get("artwork", record)
        artwork_id = artwork.get("id")
        annotations = record.get("annotations", {})
        is_real_depicted = (
            annotations.get("place_status") == "depicted"
            and annotations.get("depicted_place") is not None
        )

        accepted_places = [
            place for place in heuristic_results.get(artwork_id, [])
            if place.get("score_heuristique", 0) >= threshold
        ]

        if is_real_depicted:
            real_depicted_artworks.append(artwork_id)
            real_depicted_places += 1
        if accepted_places:
            heuristic_accepted_artworks.append(artwork_id)
            heuristic_accepted_places += len(accepted_places)

        if is_real_depicted != bool(accepted_places):
            artworks_with_discrepancy.append({
                "id": artwork_id,
                "real_depicted": is_real_depicted,
                "accepted": bool(accepted_places),
            })

    return {
        "real_depicted_artworks": len(real_depicted_artworks),
        "heuristic_accepted_artworks": len(heuristic_accepted_artworks),
        "real_depicted_places": real_depicted_places,
        "heuristic_accepted_places": heuristic_accepted_places,
        "artworks_with_discrepancy": artworks_with_discrepancy,
    }