
import re
import pandas as pd

POSITIVES_LABELS = [
    "view of", "depicting", "depicts", "shows", "showing", "landscape of",
    "vue de", "vue du", "représente", "représentant", "paysage de", "scène à",
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
    "styles": -0.1,  # "Dutch School", "French Impressionism" = rarement un lieu représenté
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

def extract_candidate_artwork(artwork, ner, seuil=0.85):
    """
    item : un objet au format {"artwork": {...}, "matched_queries": [...]}
    Retourne les candidats LOC détectés, avec le champ d'origine et la
    position dans ce champ (nécessaire pour le calcul du contexte).
    """
    

    fields = {
        "title": artwork.get("title") or "",
        "description": artwork.get("description") or "",
        "short_description": artwork.get("short_description") or "",
        "themes": " ".join(artwork.get("theme_titles") or []),
        "subjects": " ".join(artwork.get("subject_titles") or []),
        "styles": " ".join(artwork.get("style_titles") or []),
    }

    candidats = []
    for field, text in fields.items():
        if not text.strip():
            continue
        resultats = ner(text)
        for r in resultats:
            if r["entity_group"] == "LOC" and r["score"] >= seuil:
                candidats.append({
                    "lieu": r["word"],
                    "score_ner": round(float(r["score"]), 3),
                    "field": field,
                    "start": r["start"],
                    "end": r["end"],
                    "source_text": text,
                })
    return candidats


def local_context(text, start, end, window=60):
    """
    get the local context around a slice of the text
    """
    return text[max(0, start - window):end + window].lower()

def score_candidat(candidat):
    score = FIELDS_WEIGHTS.get(candidat["field"], 0.0)

    ctx = local_context(candidat["source_text"], candidat["start"], candidat["end"])
    if any(m in ctx for m in POSITIVES_LABELS):
        score += 0.3
    if any(m in ctx for m in NEGATIVES_LABELS):
        score -= 0.4

    return score