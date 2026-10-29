"""
Candidate extraction, features, labels, training and prediction of depicted places.

Pipeline (see docs/pipeline.md for the full picture):

1. `extract_candidate_artwork` runs the NER on the text fields of an artwork
   and keeps cleaned LOC mentions ("candidates").
2. Occurrences of the same name are grouped; `candidate_features`,
   `artwork_context_features` and `_add_geographic_features` describe each
   group with textual, contextual and geocoding features.
3. `build_geographic_label_table` geocodes the candidates of annotated
   artworks and labels each one at the most specific matching level
   (`geographic_candidate_label`).
4. `train_hierarchical_model` fits one logistic regression per geographic
   level (exact_place ... country) and picks a threshold that reaches the
   level's target precision; `evaluate_hierarchical_grouped_cv` re-runs the
   whole procedure on outer folds to decide which levels can be deployed.
5. `predict_with_hierarchical_model` returns, for each artwork, the most
   specific deployable level whose threshold is crossed, or abstains.

Everything is text-only: no visual data enters the features.

The module also keeps the older name-based model (`build_training_table`,
`train_logistic_model`, `predict_with_model` and the evaluation helpers that
go with it), still used by `src/main.py` and `run_pipeline.py train-legacy`.
"""

import difflib
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

try:
    from src.geolocation.gazetter import haversine_distance_km, resolve_artwork_places, resolve_location
except ModuleNotFoundError:
    from geolocation.gazetter import haversine_distance_km, resolve_artwork_places, resolve_location


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GENERIC_WORDS = {
    # Articles and prepositions.
    "the", "of", "de", "du", "des", "la", "le", "les", "l", "d", "di", "del",
    "der", "die", "das", "van", "von", "al", "el",
    # Common place types (en, fr, de, it, es, nl).
    "cathedral", "cathedrale", "church", "eglise", "chapel",
    "chapelle", "abbey", "abbaye", "castle", "chateau", "palace", "palais",
    "bridge", "pont", "gate", "porte", "tower", "tour", "square", "place",
    "street", "rue", "avenue", "boulevard", "station", "gare", "market",
    "marche", "harbour", "harbor", "port", "basilica", "basilique",
    "kathedrale", "kirche", "schloss", "burg", "dom", "duomo", "chiesa",
    "iglesia", "catedral", "kerk", "poort", "brug", "toren",
}

# Phrases near a mention that suggest it is (or is not) the depicted place.
POSITIVES_LABELS = [
    "view of", "depicting", "depicts", "shows", "showing", "landscape of", "painting of",
    "painting depicting", "painting showing", "vue sur", "vue de", "vue du", "représente",
    "représentant", "paysage de", "scène à", "peinture de", "peinture représentant",
    "peinture montrant", "photo de", "photographie de", "photographie représentant",
    "photographie montrant",
]
NEGATIVES_LABELS = [
    "born in", "died in", "studied in", "trained in", "exhibited in", "acquired",
    "né à", "mort à", "formé à", "exposé à", "collection", "purchased", "donated",
    "moved to", "returned to", "during his stay", "painted in", "peint à",
]

# Text fields read by the NER, in a fixed order. Each one becomes a binary
# feature ("field_title", ...) whose weight is learned by the model.
FIELD_NAMES = [
    "title", "alt_titles", "themes", "subjects", "description", "short_description", "styles",
]

# Minimum NER score for a candidate. Deliberately low: it only removes noise,
# the score itself is a feature (score_ner_max) weighted by the model.
DEFAULT_NER_FLOOR = 0.2

# Features of the legacy name-based model. The order is fixed so that
# training and prediction agree.
FEATURE_COLUMNS = [f"field_{field}" for field in FIELD_NAMES] + [
    "n_fields",
    "has_positive_marker",
    "has_negative_marker",
    "score_ner_max",
]

# Geographic levels, from the most to the least specific.
GEOGRAPHIC_LEVELS = [
    "exact_place",
    "neighborhood",
    "city",
    "region",
    "country",
]
_SPECIFICITY = {level: index for index, level in enumerate(GEOGRAPHIC_LEVELS)}
_GEOGRAPHIC_LABEL_LEVELS = set(GEOGRAPHIC_LEVELS)
GEOGRAPHIC_TARGET_PRECISION = {
    "exact_place": 0.95,
    "neighborhood": 0.93,
    "city": 0.90,
    "region": 0.90,
    "country": 0.90,
}
GEOGRAPHIC_FEATURE_COLUMNS = FEATURE_COLUMNS + [
    f"geo_level_{level}" for level in GEOGRAPHIC_LEVELS
] + [
    "geo_resolved",
    "geo_source_geonames",
    "geo_source_nominatim",
    "geo_has_neighborhood",
    "geo_has_city",
    "geo_has_region",
    "geo_has_country",
    "geo_importance",
    "geo_place_rank",
] + [
    "object_non_pictorial",
    "in_place_of_origin",
    "in_artist_display",
    "in_provenance_or_exhibition",
] + [
    "geo_name_similarity",
    "geo_context_conflict",
]

# AIC object types (artwork_type_title) that do not depict a scene: a place
# in their title is where they come from ("Kantharos, Attica"). Pictorial
# types (Print, Photograph, Painting...) and unknown types are not listed.
NON_PICTORIAL_ARTWORK_TYPES = {
    "vessel", "coin", "glass", "costume and accessories", "sculpture", "ceramics",
    "decorative arts", "metalwork", "furniture", "furnishings", "religious/ritual object",
    "arms", "armor", "architectural fragment", "textile", "coverings and hangings",
    "mask", "funerary object", "basketry", "design",
}

# Words that are never a place on their own (NER artefacts such as "of").
_ENTITY_STOPWORDS = _GENERIC_WORDS | {"a", "an", "and", "with", "et", "y", "e", "und"}
# Lowercase connectors that reveal a truncated entity start
# ("of Control" for "Line of Control").
_LEADING_CONNECTORS = {
    "of", "the", "de", "du", "des", "del", "della", "di", "da", "von", "van", "der",
}
# Single characters allowed between two pieces of one entity ("N" "." "Y").
# No space: entities separated by a space or ". " stay distinct
# ("Dresden. Albert Square").
_ENTITY_JOINERS = set(".-–'’")
_STRIPPED_PUNCTUATION = " \t\n,;:!?()[]{}\"«»“”‘’'"

# Nominatim `addresstype` / `type` values with an unambiguous level.
_NOMINATIM_TYPE_LEVELS = {
    "country": "country",
    "state": "region",
    "region": "region",
    "province": "region",
    "state_district": "region",
    "county": "region",
    "city": "city",
    "town": "city",
    "village": "city",
    "municipality": "city",
    "hamlet": "city",
    "neighbourhood": "neighborhood",
    "neighborhood": "neighborhood",
    "suburb": "neighborhood",
    "quarter": "neighborhood",
    "borough": "neighborhood",
    "city_district": "neighborhood",
    "city_block": "neighborhood",
}
# GeoNames codes of large areas (natural region, archipelago, mountains...).
_GEONAMES_REGION_CODES = {"RGN", "RGNE", "RGNH", "AREA", "ISLS", "PEN", "MTS", "VAL"}

# Hierarchy fields a candidate must have for a non-match to count as a real
# negative, by annotated precision (otherwise the label is unevaluable).
_REQUIRED_HIERARCHY = {
    "exact_place": ("city", "region", "country"),
    "neighborhood": ("city", "region", "country"),
    "city": ("city", "region", "country"),
    "region": ("region", "country"),
    "country": ("country",),
}


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def _artwork(record):
    """Return the artwork payload of a record (records may also be bare artworks)."""
    return record.get("artwork", record)


def _place_status(record):
    """Return the annotated place_status of a record, or None."""
    return record.get("annotations", {}).get("place_status")


def _artwork_statuses(records):
    """Map artwork id to annotated place_status."""
    return {_artwork(record).get("id"): _place_status(record) for record in records}


def _ratio(numerator, denominator):
    """Return numerator / denominator, or 0.0 when the denominator is zero."""
    return numerator / denominator if denominator else 0.0


def _f1(precision, recall):
    """Harmonic mean of precision and recall (0.0 when both are zero)."""
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _valid_coordinates(place):
    """Return (latitude, longitude) when `place` holds valid coordinates, else None."""
    if not isinstance(place, dict):
        return None
    try:
        latitude = float(place["latitude"])
        longitude = float(place["longitude"])
    except (KeyError, TypeError, ValueError):
        return None
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        return None
    return latitude, longitude


def _levels_up_to(level):
    """Levels at least as specific as `level` (cumulative positives of its model)."""
    return {candidate for candidate in GEOGRAPHIC_LEVELS if _SPECIFICITY[candidate] <= _SPECIFICITY[level]}


# ---------------------------------------------------------------------------
# Keyword pre-filter
# ---------------------------------------------------------------------------

def _flatten_keywords(keywords):
    """Flatten the nested keyword structure of keywords.json into a list of strings."""
    if isinstance(keywords, dict):
        flattened = []
        for value in keywords.values():
            flattened.extend(_flatten_keywords(value))
        return flattened
    if isinstance(keywords, (list, tuple, set)):
        return [str(keyword) for keyword in keywords]
    return [str(keywords)]


def _keyword_pattern(keyword):
    """Case-insensitive regex matching `keyword` as a whole word."""
    escaped_keyword = re.escape(keyword.strip())
    return re.compile(r"(?<!\w)" + escaped_keyword + r"(?!\w)", re.IGNORECASE)


def filter_by_keywords(combined_texts, keywords):
    """
    Keep the artworks whose combined text contains at least one keyword.

    Parameters:
    combined_texts (dict): Artwork id -> combined text of the artwork.
    keywords (dict or list): The nested keyword structure from keywords.json,
        or a flat list of keywords.

    Returns:
    dict: {"filtered_artworks": [{"id": 1, "matched_keywords": ["street"]}, ...]}
    """
    keyword_patterns = [
        (keyword, _keyword_pattern(keyword))
        for keyword in _flatten_keywords(keywords)
        if keyword.strip()
    ]
    artworks_df = pd.DataFrame(combined_texts.items(), columns=["id", "combined_text"])
    artworks_df["matched_keywords"] = artworks_df["combined_text"].fillna("").apply(
        lambda text: [keyword for keyword, pattern in keyword_patterns if pattern.search(text)]
    )
    filtered_artworks = artworks_df.loc[
        artworks_df["matched_keywords"].str.len() > 0,
        ["id", "matched_keywords"],
    ].to_dict(orient="records")
    return {"filtered_artworks": filtered_artworks}


# ---------------------------------------------------------------------------
# NER candidate extraction and cleaning
# ---------------------------------------------------------------------------

def _is_word_char(character):
    """True for letters, digits and combining accents."""
    return character.isalnum() or unicodedata.category(character) == "Mn"


def _in_word(text, index):
    """True if text[index] belongs to a word; an apostrophe between letters does ("Akbar's")."""
    if not 0 <= index < len(text):
        return False
    if _is_word_char(text[index]):
        return True
    return (
        text[index] in "'’"
        and 0 < index < len(text) - 1
        and _is_word_char(text[index - 1])
        and _is_word_char(text[index + 1])
    )


def _merge_entity_pieces(text, entities):
    """Extend NER spans to whole words and merge pieces of the same entity.

    Two spans are merged when they overlap or are separated by a single
    joiner character ("N.Y", "Pirna-Posta"); the merged score is the maximum.
    """
    spans = []
    for entity in sorted(entities, key=lambda item: item["start"]):
        start, end = entity["start"], entity["end"]
        while start > 0 and _in_word(text, start - 1) and _in_word(text, start):
            start -= 1
        while end < len(text) and _in_word(text, end - 1) and _in_word(text, end):
            end += 1
        score = float(entity["score"])
        if spans:
            previous = spans[-1]
            gap = text[previous["end"]:start]
            if start <= previous["end"] or (len(gap) == 1 and gap in _ENTITY_JOINERS):
                previous["end"] = max(previous["end"], end)
                previous["score"] = max(previous["score"], score)
                continue
        spans.append({"start": start, "end": end, "score": score})
    return spans


def _trim_entity_span(text, start, end):
    """Remove punctuation, a "d'" elision and repair a leading connector.

    Returns the corrected (start, end).
    """
    while start < end and text[start] in _STRIPPED_PUNCTUATION:
        start += 1
    while end > start and text[end - 1] in _STRIPPED_PUNCTUATION:
        end -= 1
    # A final dot is kept only in abbreviations ("N.Y.").
    if end > start and text[end - 1] == "." and "." not in text[start:end - 1]:
        end -= 1

    elision = re.match(r"d['’]", text[start:end])
    if elision:
        start += elision.end()

    first_word = re.match(r"\w+", text[start:end])
    if first_word and first_word.group(0) in _LEADING_CONNECTORS and first_word.end() < end - start:
        previous_word = re.search(r"(\w+)\s+$", text[:start])
        if previous_word and previous_word.group(1)[:1].isupper():
            # "of Control" -> "Line of Control".
            start = previous_word.start(1)
        else:
            start += first_word.end()
            while start < end and text[start].isspace():
                start += 1
    return start, end


def _clean_entity_spans(text, entities):
    """
    Repair raw NER LOC spans using the source text.

    Names are read back from `text` through the offsets (not from the
    tokenizer's decoded `word`, which inserts spaces: "Saint - Lazare").
    Spans cut inside a word are extended to the whole word, adjacent pieces
    of one entity are merged, and spans that cannot be a place name (a single
    letter, a function word) are dropped.

    Returns a list of {"start", "end", "score", "surface"} dicts.
    """
    cleaned = []
    for span in _merge_entity_pieces(text, entities):
        start, end = _trim_entity_span(text, span["start"], span["end"])
        surface = text[start:end]
        letter_count = sum(character.isalpha() for character in surface)
        if letter_count < 2 or _normalize_place_name(surface) in _ENTITY_STOPWORDS:
            continue
        cleaned.append({**span, "start": start, "end": end, "surface": surface})
    return cleaned


def _artwork_text_fields(artwork):
    """Return the text of each field in FIELD_NAMES (lists are joined)."""
    return {
        "title": artwork.get("title") or "",
        "alt_titles": " / ".join(artwork.get("alt_titles") or []),
        "description": artwork.get("description") or "",
        "short_description": artwork.get("short_description") or "",
        "themes": " ".join(artwork.get("theme_titles") or []),
        "subjects": " ".join(artwork.get("subject_titles") or []),
        "styles": " ".join(artwork.get("style_titles") or []),
    }


def extract_candidate_artwork(artwork, ner, threshold=DEFAULT_NER_FLOOR):
    """
    Extract cleaned LOC candidates from the text fields of an artwork.

    Each candidate keeps its source field and its position in that field
    (needed for the context markers). `threshold` is a low noise floor, not
    the acceptance threshold: it is applied after cleaning, to the best score
    of the merged pieces, and the score itself stays a model feature.
    """
    candidates = []
    for field, text in _artwork_text_fields(artwork).items():
        if not text.strip():
            continue
        locations = [result for result in ner(text) if result["entity_group"] == "LOC"]
        for span in _clean_entity_spans(text, locations):
            if span["score"] >= threshold:
                candidates.append({
                    "place": span["surface"],
                    "score_ner": round(span["score"], 3),
                    "field": field,
                    "start": span["start"],
                    "end": span["end"],
                    "source_text": text,
                })
    return candidates


def extract_all_candidates(records, ner, threshold=DEFAULT_NER_FLOOR):
    """
    Run `extract_candidate_artwork` on every record, keyed by artwork id.

    The NER runs once; the raw candidates are reused both to build the
    training table and to predict.
    """
    return {
        _artwork(record).get("id"): extract_candidate_artwork(_artwork(record), ner, threshold=threshold)
        for record in records
    }


# ---------------------------------------------------------------------------
# Candidate features
# ---------------------------------------------------------------------------

def local_context(text, start, end, window=60):
    """Return the lowercased text around text[start:end], `window` characters each side."""
    return text[max(0, start - window):end + window].lower()


def group_candidates_by_place(candidates):
    """Group candidate occurrences by place name (case-insensitive)."""
    groups = defaultdict(list)
    for candidate in candidates:
        groups[candidate["place"].lower().strip()].append(candidate)
    return groups


def _has_marker(occurrences, markers):
    """True if any marker appears in the local context of any occurrence."""
    return any(
        any(marker in local_context(c["source_text"], c["start"], c["end"]) for marker in markers)
        for c in occurrences
    )


def candidate_features(occurrences):
    """
    Raw textual features of one grouped candidate (all its occurrences).

    The logistic regression learns how to weight them: fields where the
    place appears, number of fields, positive/negative context markers and
    best NER score.
    """
    found_fields = {c["field"] for c in occurrences}
    features = {f"field_{field}": int(field in found_fields) for field in FIELD_NAMES}
    features["n_fields"] = len(found_fields)
    features["has_positive_marker"] = int(_has_marker(occurrences, POSITIVES_LABELS))
    features["has_negative_marker"] = int(_has_marker(occurrences, NEGATIVES_LABELS))
    features["score_ner_max"] = max(c["score_ner"] for c in occurrences)
    return features


def _mentions(text, place_name):
    """True if the normalized place name appears as whole words in `text`."""
    place = _normalize_place_name(place_name)
    return bool(place) and f" {place} " in f" {_normalize_place_name(text)} "


def artwork_context_features(artwork, place_name):
    """
    AIC context clues for a candidate (see aic.CONTEXT_FIELDS).

    These fields never produce candidates. They only tell whether the place
    found in the title/description is also the artwork's place of origin,
    appears in the artist's biography ("born Leiden") or in the provenance /
    exhibition history, and whether the object is non-pictorial (a vessel, a
    coin...). The model learns the weight of each clue.

    `has_context_metadata` is computed for models trained before it was
    removed from GEOGRAPHIC_FEATURE_COLUMNS: it only flagged AIC records and
    the model used it as a source shortcut (21 % depicted places among AIC
    gold artworks versus 70 % elsewhere).
    """
    history = " ".join(
        str(artwork.get(field) or "")
        for field in ("exhibition_history", "provenance_text", "credit_line")
    )
    origin = artwork.get("place_of_origin") or ""
    artist = artwork.get("artist_display") or ""
    artwork_type = str(artwork.get("artwork_type_title") or "").casefold()
    return {
        "has_context_metadata": int(bool(origin or artist or history.strip())),
        "object_non_pictorial": int(artwork_type in NON_PICTORIAL_ARTWORK_TYPES),
        "in_place_of_origin": int(_mentions(origin, place_name)),
        "in_artist_display": int(_mentions(artist, place_name)),
        "in_provenance_or_exhibition": int(_mentions(history, place_name)),
    }


def _add_geographic_features(features, location):
    """Return a copy of `features` with the geocoding features of `location`."""
    location = location or {}
    candidate_level = _candidate_geographic_level(location)
    hierarchy = location.get("hierarchy") or {}
    result = dict(features)
    for level in GEOGRAPHIC_LEVELS:
        result[f"geo_level_{level}"] = int(candidate_level == level)
    result.update({
        "geo_resolved": int(bool(location)),
        "geo_source_geonames": int(location.get("source") == "GeoNames"),
        "geo_source_nominatim": int(location.get("source") == "Nominatim"),
        "geo_has_neighborhood": int(bool(hierarchy.get("neighborhood"))),
        "geo_has_city": int(bool(hierarchy.get("city"))),
        "geo_has_region": int(bool(hierarchy.get("region"))),
        "geo_has_country": int(bool(hierarchy.get("country") or location.get("country_code"))),
        "geo_importance": float(location.get("importance") or 0.0),
        "geo_place_rank": float(location.get("place_rank") or 0.0),
    })
    return result


def _name_similarity(place_name, location):
    """Similarity (0-1) between a candidate and the name of its geocoded location.

    1.0 when the significant words of one name are contained in the other's
    ("Rouen Cathedral" / "Cathédrale de Rouen"), otherwise the character
    similarity. Low values flag geocodes unrelated to the mention ("Louvre"
    -> "Γουβιά"), but also legitimate translations ("Venice" -> "Venezia").
    """
    resolved_name = (location or {}).get("name")
    candidate, resolved = _normalize_place_name(place_name), _normalize_place_name(resolved_name)
    if not candidate or not resolved:
        return 0.0
    candidate_tokens, resolved_tokens = _core_tokens(place_name), _core_tokens(resolved_name)
    if candidate_tokens and resolved_tokens and (
        candidate_tokens <= resolved_tokens or resolved_tokens <= candidate_tokens
    ):
        return 1.0
    return difflib.SequenceMatcher(None, candidate, resolved).ratio()


def _hierarchical_candidate_features(occurrences, location, artwork):
    """All features of a grouped candidate used by the hierarchical model."""
    place_name = occurrences[0]["place"]
    features = _add_geographic_features(candidate_features(occurrences), location)
    features.update(artwork_context_features(artwork, place_name))
    features["geo_name_similarity"] = _name_similarity(place_name, location)
    features["geo_context_conflict"] = int(bool((location or {}).get("context_conflict")))
    return features


# ---------------------------------------------------------------------------
# Place-name matching
# ---------------------------------------------------------------------------

def _normalize_place_name(name):
    """Lowercase, strip accents and punctuation, collapse spaces."""
    text = unicodedata.normalize("NFKD", (name or "").lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _core_tokens(name):
    """Normalized tokens of a name, without articles and generic place words."""
    return {t for t in _normalize_place_name(name).split() if t not in _GENERIC_WORDS}


def fuzzy_match(name_a, name_b, threshold=0.85):
    """
    Lenient name match ignoring accents, articles and generic words
    ("cathédrale de Rouen" == "Rouen Cathedral").
    """
    a, b = _normalize_place_name(name_a), _normalize_place_name(name_b)
    if not a or not b:
        return False
    if a == b:
        return True
    ta, tb = _core_tokens(name_a), _core_tokens(name_b)
    if ta and tb and (ta <= tb or tb <= ta):
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= threshold


# ---------------------------------------------------------------------------
# Geographic level of a geocoded candidate
# ---------------------------------------------------------------------------

def _geonames_level(feature_code):
    """Geographic level of a GeoNames feature code."""
    if feature_code.startswith("PCL"):
        return "country"
    # ADM1/ADM2 (state, département, US county) are regions; ADM3+ is most
    # often a municipality (comune, Gemeinde).
    if feature_code.startswith(("ADM1", "ADM2", "ADMD")) or feature_code in _GEONAMES_REGION_CODES:
        return "region"
    if feature_code == "PPLX":
        return "neighborhood"
    if feature_code.startswith(("PPL", "ADM")):
        return "city"
    return "exact_place"


def _place_rank(location):
    """Nominatim place_rank as an int (0 when missing or invalid)."""
    try:
        return int(location.get("place_rank") or 0)
    except (TypeError, ValueError):
        return 0


def _nominatim_level(location):
    """Geographic level of a Nominatim result, or None if it cannot be told."""
    for key in ("addresstype", "feature_code"):
        level = _NOMINATIM_TYPE_LEVELS.get(str(location.get(key) or "").casefold())
        if level:
            return level
    place_rank = _place_rank(location)
    place_type = str(location.get("feature_code") or "").casefold()
    place_class = str(location.get("place_class") or "").casefold()
    # Administrative boundaries have no telling type: their level comes from
    # place_rank (4 country, 8 state, 12 county, 16 city, 20 suburb).
    if place_rank and (place_type == "administrative" or place_class == "boundary"):
        if place_rank <= 4:
            return "country"
        if place_rank <= 12:
            return "region"
        if place_rank <= 19:
            return "city"
        if place_rank <= 24:
            return "neighborhood"
    if place_rank >= 25:
        return "exact_place"
    if place_class and place_class not in {"place", "boundary"}:
        return "exact_place"
    return None


def _candidate_geographic_level(location):
    """Geographic level of a geocoded location (GeoNames or Nominatim), or None."""
    if not location:
        return None
    if location.get("source") == "GeoNames":
        return _geonames_level(str(location.get("feature_code") or "").upper())
    return _nominatim_level(location)


def _geocode_artwork(grouped_candidates, alternatives_cache, gazetteer_conn, nominatim_client):
    """Geocode the grouped candidates of one artwork together (see gazetter.resolve_artwork_places).

    Returns one location (or None) per group, in order.
    """
    if not grouped_candidates or (gazetteer_conn is None and nominatim_client is None):
        return [None] * len(grouped_candidates)
    mentions = [(occurrences[0]["place"], occurrences) for occurrences in grouped_candidates]
    resolved = resolve_artwork_places(mentions, gazetteer_conn, nominatim_client, alternatives_cache)
    return [resolved[name] for name, _ in mentions]


# ---------------------------------------------------------------------------
# Hierarchical labels
# ---------------------------------------------------------------------------

def _unevaluable(distance_km=None, **details):
    """Label for a candidate that can be neither positive nor negative."""
    return {"level": "unevaluable", "status": "unevaluable", "distance_km": distance_km, **details}


def _hierarchy_matches(target, candidate_location, candidate_level):
    """Tell whether the candidate shares the target's country, region, city and neighborhood.

    Names are compared both to the annotation and to the target hierarchy
    obtained by reverse geocoding (`resolved_hierarchy`), which is in the
    geocoder's language ("Sachsen") while the annotation may be in English
    ("Saxony"). Countries also match by ISO code.
    """
    hierarchy = candidate_location.get("hierarchy") or {}
    target_hierarchy = target.get("resolved_hierarchy") or {}
    candidate_name = _normalize_place_name(candidate_location.get("name"))

    def value_matches(field):
        """True if the candidate's value for `field` equals the target's (annotated or reverse-geocoded)."""
        target_values = {
            value
            for value in (
                _normalize_place_name(target.get(field)),
                _normalize_place_name(target_hierarchy.get(field)),
            )
            if value
        }
        resolved_value = _normalize_place_name(hierarchy.get(field))
        if not resolved_value and candidate_level == field:
            resolved_value = candidate_name
        return bool(resolved_value and resolved_value in target_values)

    target_country_code = (
        target.get("country_code") or target_hierarchy.get("country_code") or ""
    ).upper()
    candidate_country_code = (
        hierarchy.get("country_code") or candidate_location.get("country_code") or ""
    ).upper()
    target_name = _normalize_place_name(target.get("name"))
    return {
        "country": value_matches("country") or bool(
            target_country_code and target_country_code == candidate_country_code
        ),
        "region": value_matches("region"),
        "city": value_matches("city"),
        "neighborhood": bool(
            target_name
            and (
                _normalize_place_name(hierarchy.get("neighborhood")) == target_name
                or (candidate_level == "neighborhood" and candidate_name == target_name)
            )
        ),
    }


def _matched_level(precision, candidate_level, distance, matches, tolerances):
    """Most specific level at which the candidate matches a target of `precision`."""
    exact_tolerance_km, neighborhood_tolerance_km, city_tolerance_km = tolerances
    if precision == "exact_place":
        if candidate_level == "exact_place" and distance <= exact_tolerance_km:
            return "exact_place"
        if matches["city"]:
            return "city"
    elif precision == "neighborhood":
        if matches["neighborhood"] or (
            candidate_level == "exact_place" and distance <= neighborhood_tolerance_km
        ):
            return "neighborhood"
        if matches["city"]:
            return "city"
    elif precision == "city":
        if matches["city"] or (
            candidate_level in {"exact_place", "neighborhood", "city"}
            and distance <= city_tolerance_km
        ):
            return "city"
    elif precision == "region" and matches["region"]:
        return "region"
    elif precision == "country" and matches["country"]:
        return "country"

    # Fall back to a coarser level shared with the target.
    if precision not in {"region", "country"} and matches["region"]:
        return "region"
    if matches["country"]:
        return "country"
    return None


def _label_against_target(target, candidate_location, tolerances):
    """Label a geocoded candidate against one annotated depicted place."""
    precision = target.get("precision")
    if precision not in _GEOGRAPHIC_LABEL_LEVELS:
        return _unevaluable()
    target_coordinates = _valid_coordinates(target)
    candidate_coordinates = _valid_coordinates(candidate_location)
    if target_coordinates is None or candidate_coordinates is None:
        return _unevaluable()

    distance = haversine_distance_km(*target_coordinates, *candidate_coordinates)
    candidate_level = _candidate_geographic_level(candidate_location)
    details = {"candidate_level": candidate_level, "geocoder_source": candidate_location.get("source")}
    matches = _hierarchy_matches(target, candidate_location, candidate_level)
    matched_level = _matched_level(precision, candidate_level, distance, matches, tolerances)

    hierarchy = candidate_location.get("hierarchy") or {}
    if matched_level is None and not any(hierarchy.get(field) for field in _REQUIRED_HIERARCHY[precision]):
        # Without a hierarchy, a non-match proves nothing.
        return _unevaluable(distance, **details)
    return {
        "level": matched_level or "no_match",
        "status": "match" if matched_level else "negative",
        "distance_km": distance,
        **details,
    }


def _label_against_several_targets(depicted_places, candidate_location, tolerances):
    """Label against each target and keep the most specific match."""
    labels = [
        _label_against_target(depicted_place, candidate_location, tolerances)
        for depicted_place in depicted_places
    ]
    matching_labels = [label for label in labels if label["status"] == "match"]
    if matching_labels:
        return min(matching_labels, key=lambda label: _SPECIFICITY[label["level"]])
    if any(label["status"] == "unevaluable" for label in labels):
        return _unevaluable(
            next((label.get("distance_km") for label in labels if label.get("distance_km") is not None), None),
            candidate_level=labels[0].get("candidate_level"),
            geocoder_source=labels[0].get("geocoder_source"),
        )
    return labels[0]


def geographic_candidate_label(
    record,
    candidate_location,
    *,
    exact_tolerance_km=0.25,
    neighborhood_tolerance_km=2.0,
    city_tolerance_km=25.0,
):
    """
    Label a candidate at the most specific level supported by its location.

    Returns a dict with `level` (a geographic level, "no_match", "none" or
    "unevaluable"), `status` ("match", "negative" or "unevaluable") and
    `distance_km`, plus `candidate_level` and `geocoder_source` when the
    candidate was compared. Artworks that are not "depicted" make every
    candidate negative; unresolved cases are never forced to negative.
    """
    if min(exact_tolerance_km, neighborhood_tolerance_km, city_tolerance_km) < 0:
        raise ValueError("Geographic label tolerances must be non-negative.")
    annotations = record.get("annotations", {})
    place_status = annotations.get("place_status")
    if place_status == "undeterminable":
        return _unevaluable()
    if place_status != "depicted":
        # The candidate level is still needed: without it the evaluation
        # never scores these candidates and cannot count their false positives.
        return {
            "level": "none",
            "status": "negative",
            "distance_km": None,
            "candidate_level": _candidate_geographic_level(candidate_location),
            "geocoder_source": (candidate_location or {}).get("source"),
        }

    tolerances = (exact_tolerance_km, neighborhood_tolerance_km, city_tolerance_km)
    depicted_places = annotations.get("depicted_places") or []
    if len(depicted_places) > 1:
        return _label_against_several_targets(depicted_places, candidate_location, tolerances)
    return _label_against_target(annotations.get("depicted_place") or {}, candidate_location, tolerances)


def _with_target_hierarchies(record, nominatim_client, reverse_cache):
    """Return a copy of `record` whose depicted places carry `resolved_hierarchy`.

    The hierarchy comes from reverse-geocoding the annotated coordinates, so
    that region/country names are compared in the geocoder's language.
    """
    if nominatim_client is None or not hasattr(nominatim_client, "reverse"):
        return record
    annotations = record.get("annotations") or {}

    def resolve(place):
        """Add the reverse-geocoded hierarchy to one depicted place."""
        coordinates = _valid_coordinates(place or {})
        if coordinates is None or place.get("resolved_hierarchy"):
            return place
        if coordinates not in reverse_cache:
            reverse_cache[coordinates] = nominatim_client.reverse(*coordinates)
        location = reverse_cache[coordinates]
        if not location:
            return place
        return {**place, "resolved_hierarchy": location.get("hierarchy") or {}}

    updated = dict(annotations)
    if annotations.get("depicted_place"):
        updated["depicted_place"] = resolve(annotations["depicted_place"])
    if annotations.get("depicted_places"):
        updated["depicted_places"] = [resolve(place) for place in annotations["depicted_places"]]
    return {**record, "annotations": updated}


# ---------------------------------------------------------------------------
# Cross-validation groups
# ---------------------------------------------------------------------------

def _target_group_key(place):
    """Stable key of an annotated target (Wikidata, GeoNames, coordinates or name)."""
    wikidata_id = place.get("wikidata_id")
    if wikidata_id:
        return f"wikidata:{wikidata_id}"
    geoname_id = place.get("geonameid")
    if geoname_id:
        return f"geonames:{geoname_id}"
    coordinates = _valid_coordinates(place)
    if coordinates is not None:
        return (
            f"coordinates:{coordinates[0]:.4f},{coordinates[1]:.4f}:"
            f"{place.get('precision', 'unknown')}"
        )
    name = _normalize_place_name(place.get("name"))
    return f"name:{name}:{place.get('precision', 'unknown')}" if name else None


def _artwork_target_cv_groups(records):
    """Map each artwork id to a CV group joining artworks that share a target.

    Groups are the connected components of the artwork-target graph (a
    union-find), so that several views of the same monument never end up in
    both the training and the test fold.
    """
    parents = {}

    def find(item):
        """Root of `item`, with path compression."""
        parents.setdefault(item, item)
        if parents[item] != item:
            parents[item] = find(parents[item])
        return parents[item]

    def union(left, right):
        """Join the components of `left` and `right`."""
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    artwork_nodes = []
    for record in records:
        artwork_id = _artwork(record).get("id")
        artwork_node = f"artwork:{artwork_id}"
        find(artwork_node)
        artwork_nodes.append((artwork_id, artwork_node))
        annotations = record.get("annotations", {})
        targets = annotations.get("depicted_places") or [annotations.get("depicted_place")]
        for target in targets:
            target_key = _target_group_key(target) if target else None
            if target_key:
                union(artwork_node, f"target:{target_key}")

    return {artwork_id: find(artwork_node) for artwork_id, artwork_node in artwork_nodes}


# ---------------------------------------------------------------------------
# Training tables
# ---------------------------------------------------------------------------

def build_geographic_label_table(
    records,
    candidates_by_artwork,
    *,
    gazetteer_conn=None,
    nominatim_client=None,
    exact_tolerance_km=0.25,
    neighborhood_tolerance_km=2.0,
    city_tolerance_km=25.0,
):
    """
    Build one row per (artwork, grouped candidate) with features and hierarchical label.

    The candidates of an artwork are geocoded jointly (they disambiguate
    each other); unresolved cases are labelled
    "unevaluable" instead of negative. `table.attrs` keeps the place_status
    and the CV group of every artwork, including artworks without candidates.
    """
    rows = []
    alternatives_cache = {}
    reverse_cache = {}
    cv_groups = _artwork_target_cv_groups(records)
    for record in records:
        record = _with_target_hierarchies(record, nominatim_client, reverse_cache)
        artwork = _artwork(record)
        artwork_id = artwork.get("id")
        groups = list(group_candidates_by_place(candidates_by_artwork.get(artwork_id, [])).values())
        locations = _geocode_artwork(groups, alternatives_cache, gazetteer_conn, nominatim_client)
        for occurrences, location in zip(groups, locations):
            place_name = occurrences[0]["place"]
            label = geographic_candidate_label(
                record,
                location,
                exact_tolerance_km=exact_tolerance_km,
                neighborhood_tolerance_km=neighborhood_tolerance_km,
                city_tolerance_km=city_tolerance_km,
            )
            row = _hierarchical_candidate_features(occurrences, location, artwork)
            row.update({
                "artwork_id": artwork_id,
                "cv_group": cv_groups.get(artwork_id, f"artwork:{artwork_id}"),
                "place_status": _place_status(record),
                "place": place_name,
                "label_level": label["level"],
                "label_status": label["status"],
                "distance_km": label.get("distance_km"),
                "candidate_level": label.get("candidate_level"),
                "geocoder_source": label.get("geocoder_source"),
                "candidate_geolocation": location,
            })
            rows.append(row)
    table = pd.DataFrame(rows)
    table.attrs["artwork_statuses"] = _artwork_statuses(records)
    table.attrs["artwork_cv_groups"] = cv_groups
    return table


def label_candidate(record, place_name, fuzzy_threshold=0.85):
    """
    Legacy binary label: 1 if `place_name` fuzzy-matches an annotated depicted place.

    Artworks that are not "depicted" (including "mentioned_only" and "none")
    make all their candidates negative, which provides hard negatives such as
    a creation place cited next to the real subject. "undeterminable"
    artworks are excluded upstream by `build_training_table`.
    """
    annotations = record.get("annotations", {})
    if annotations.get("place_status") != "depicted":
        return 0
    depicted_places = annotations.get("depicted_places") or [annotations.get("depicted_place") or {}]
    return int(any(
        fuzzy_match(place_name, depicted.get("name"), fuzzy_threshold)
        for depicted in depicted_places
    ))


def build_training_table(records, candidates_by_artwork, fuzzy_threshold=0.85):
    """Legacy table: one row per (artwork, grouped candidate) with name-based label."""
    rows = []
    for record in records:
        if _place_status(record) == "undeterminable":
            continue
        artwork_id = _artwork(record).get("id")
        groups = group_candidates_by_place(candidates_by_artwork.get(artwork_id, []))
        for occurrences in groups.values():
            place_name = occurrences[0]["place"]  # Original spelling and case.
            row = candidate_features(occurrences)
            row["artwork_id"] = artwork_id
            row["place"] = place_name
            row["label"] = label_candidate(record, place_name, fuzzy_threshold)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Legacy name-based model
# ---------------------------------------------------------------------------

def _best_threshold(labels, probabilities, minimum_recall=0.8):
    """
    Most precise threshold reaching `minimum_recall` on the precision-recall
    curve; if none reaches it, the threshold with the best F1.
    """
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    # The curve has one more point than thresholds (precision=1, recall=0).
    precision, recall = precision[:-1], recall[:-1]

    meets_target = recall >= minimum_recall
    if meets_target.any():
        candidate_idx = precision[meets_target].argmax()
        return float(thresholds[meets_target][candidate_idx])

    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    return float(thresholds[f1_scores.argmax()])


def train_logistic_model(
    table,
    n_splits=5,
    random_state=42,
    regularization_values=(0.01, 0.1, 1.0, 10.0),
    minimum_recall=0.8,
):
    """
    Train the legacy name-based logistic regression.

    Cross-validation is grouped by artwork (StratifiedGroupKFold) so that
    candidates of one artwork never sit in both train and test. C is chosen
    by out-of-fold F1, the threshold on the out-of-fold probabilities, and
    the final model is refitted on all rows.

    Returns the model, chosen C and threshold, coefficients and out-of-fold
    metrics.
    """
    X = table[FEATURE_COLUMNS].values
    y = table["label"].values
    groups = table["artwork_id"].astype(str).values

    effective_splits = min(n_splits, len(set(groups)))
    if effective_splits < 2:
        raise ValueError("Au moins 2 œuvres distinctes sont nécessaires pour la validation croisée.")
    cv = StratifiedGroupKFold(n_splits=effective_splits, shuffle=True, random_state=random_state)

    best_C = None
    best_f1 = -1.0
    best_oof_probabilities = None
    for C in regularization_values:
        oof_probabilities = pd.Series(index=table.index, dtype=float)
        for train_idx, test_idx in cv.split(X, y, groups):
            clf = LogisticRegression(C=C, class_weight="balanced", max_iter=1000)
            clf.fit(X[train_idx], y[train_idx])
            oof_probabilities.iloc[test_idx] = clf.predict_proba(X[test_idx])[:, 1]

        precision, recall, _ = precision_recall_curve(y, oof_probabilities.values)
        f1 = float((2 * precision * recall / (precision + recall + 1e-9)).max())
        if f1 > best_f1:
            best_f1, best_C = f1, C
            best_oof_probabilities = oof_probabilities.values.copy()

    # Threshold chosen on out-of-fold probabilities, not on a refit, to avoid
    # an optimistic threshold.
    threshold = _best_threshold(y, best_oof_probabilities, minimum_recall=minimum_recall)
    oof_predictions = best_oof_probabilities >= threshold
    true_positives = int(((oof_predictions == 1) & (y == 1)).sum())
    false_positives = int(((oof_predictions == 1) & (y == 0)).sum())
    false_negatives = int(((oof_predictions == 0) & (y == 1)).sum())
    true_negatives = int(((oof_predictions == 0) & (y == 0)).sum())
    precision = _ratio(true_positives, true_positives + false_positives)
    recall = _ratio(true_positives, true_positives + false_negatives)

    final_clf = LogisticRegression(C=best_C, class_weight="balanced", max_iter=1000)
    final_clf.fit(X, y)
    return {
        "model": final_clf,
        "feature_columns": FEATURE_COLUMNS,
        "C": best_C,
        "threshold": threshold,
        "coefficients": dict(zip(FEATURE_COLUMNS, final_clf.coef_[0].tolist())),
        "intercept": float(final_clf.intercept_[0]),
        "cv_metrics": {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
        },
    }


def predict_with_model(records, candidates_by_artwork, model):
    """
    Apply the legacy model: for each artwork id, candidates sorted by probability.

    The output keeps the historical key "score_heuristique" so that
    `evaluate_binary_predictions` works unchanged.
    """
    clf = model["model"]
    feature_columns = model["feature_columns"]
    results = {}
    for record in records:
        artwork_id = _artwork(record).get("id")
        places = []
        for occurrences in group_candidates_by_place(candidates_by_artwork.get(artwork_id, [])).values():
            row = candidate_features(occurrences)
            probability = float(clf.predict_proba([[row[col] for col in feature_columns]])[0, 1])
            places.append({
                "place": occurrences[0]["place"],
                "score_heuristique": probability,
                "champs": sorted({c["field"] for c in occurrences}),
            })
        results[artwork_id] = sorted(places, key=lambda p: p["score_heuristique"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Hierarchical model: training
# ---------------------------------------------------------------------------

def _threshold_for_precision(
    labels,
    probabilities,
    groups,
    target_precision,
    minimum_predictions,
    minimum_prediction_groups,
):
    """Threshold with the best recall whose precision reaches the target.

    Only thresholds accepting at least `minimum_predictions` rows from
    `minimum_prediction_groups` CV groups are considered. Returns the
    threshold with its statistics, or None if no threshold qualifies.
    """
    best = None
    for threshold in sorted(set(float(value) for value in probabilities), reverse=True):
        accepted = probabilities >= threshold
        prediction_count = int(accepted.sum())
        accepted_groups = len(set(groups[accepted]))
        if prediction_count < minimum_predictions or accepted_groups < minimum_prediction_groups:
            continue
        true_positives = int(((labels == 1) & accepted).sum())
        precision = true_positives / prediction_count
        if precision < target_precision:
            continue
        candidate = {
            "threshold": threshold,
            "precision": precision,
            "recall": true_positives / int((labels == 1).sum()),
            "support": prediction_count,
            "prediction_groups": accepted_groups,
            "true_positives": true_positives,
            "false_positives": prediction_count - true_positives,
        }
        rank = (candidate["recall"], candidate["precision"], candidate["threshold"])
        if best is None or rank > (best["recall"], best["precision"], best["threshold"]):
            best = candidate
    return best


def _abstaining_level(status, target_precision, **details):
    """Per-level entry of a level that does not predict."""
    return {"status": status, "target_precision": target_precision, **details, "model": None, "threshold": None}


def _best_regularization(X, labels, groups, n_splits, random_state, regularization_values, threshold_options):
    """Try each C with grouped CV and keep the one whose precision-constrained threshold has the best recall."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best = None
    for C in regularization_values:
        oof = np.full(len(labels), np.nan)
        try:
            for train_idx, test_idx in cv.split(X, labels, groups):
                classifier = LogisticRegression(C=C, class_weight="balanced", max_iter=1000)
                classifier.fit(X[train_idx], labels[train_idx])
                oof[test_idx] = classifier.predict_proba(X[test_idx])[:, 1]
        except ValueError:
            continue
        if not np.isfinite(oof).all():
            continue
        selected = _threshold_for_precision(labels, oof, groups, **threshold_options)
        if selected is None:
            continue
        candidate = {**selected, "C": C, "oof_probabilities": oof}
        rank = (candidate["recall"], candidate["precision"], -candidate["false_positives"])
        if best is None or rank > (best["recall"], best["precision"], -best["false_positives"]):
            best = candidate
    return best


def _fit_level(level, matrix, statuses, label_levels, groups, target_precision, options):
    """Train the classifier of one level, or explain why the level abstains."""
    positive_levels = _levels_up_to(level)
    row_indices = np.flatnonzero((statuses == "negative") | (statuses == "match"))
    level_labels = np.asarray([
        int(statuses[index] == "match" and label_levels[index] in positive_levels)
        for index in row_indices
    ])
    level_groups = groups[row_indices]
    positive_group_count = len(set(level_groups[level_labels == 1]))
    negative_group_count = len(set(level_groups[level_labels == 0]))

    if positive_group_count < options["minimum_positive_groups"]:
        return _abstaining_level(
            "insufficient_positive_groups",
            target_precision,
            positive_groups=positive_group_count,
            minimum_positive_groups=options["minimum_positive_groups"],
        )
    if negative_group_count < 2:
        return _abstaining_level("insufficient_negative_groups", target_precision, negative_groups=negative_group_count)
    n_level_splits = min(options["n_splits"], positive_group_count, negative_group_count)
    if n_level_splits < 2:
        return _abstaining_level("insufficient_cv_groups", target_precision)

    X = matrix[row_indices]
    best = _best_regularization(
        X,
        level_labels,
        level_groups,
        n_level_splits,
        options["random_state"],
        options["regularization_values"],
        {
            "target_precision": target_precision,
            "minimum_predictions": options["minimum_predictions"],
            "minimum_prediction_groups": options["minimum_prediction_groups"],
        },
    )
    if best is None:
        return _abstaining_level(
            "precision_target_unmet",
            target_precision,
            positive_groups=positive_group_count,
            negative_groups=negative_group_count,
        )

    final_classifier = LogisticRegression(C=best["C"], class_weight="balanced", max_iter=1000)
    final_classifier.fit(X, level_labels)
    return {
        "status": "active",
        "model": final_classifier,
        "C": best["C"],
        "threshold": best["threshold"],
        "target_precision": target_precision,
        "oof_precision": best["precision"],
        "oof_recall": best["recall"],
        "oof_support": best["support"],
        "oof_prediction_groups": best["prediction_groups"],
        "positive_groups": positive_group_count,
        "negative_groups": negative_group_count,
        "cv_splits": n_level_splits,
        "false_positives": best["false_positives"],
        "true_positives": best["true_positives"],
        # Used below for the work-level metrics, then removed.
        "oof_row_indices": row_indices.tolist(),
        "oof_probabilities": best["oof_probabilities"].tolist(),
    }


def train_hierarchical_model(
    table,
    *,
    n_splits=5,
    random_state=42,
    regularization_values=(0.1, 1.0, 10.0),
    target_precisions=None,
    minimum_recall=0.5,
    minimum_positive_groups=5,
    minimum_prediction_groups=5,
    minimum_predictions=5,
):
    """Fit one grouped-CV classifier per geographic precision level.

    The classifier of a level treats as positive every candidate matching at
    that level or a more specific one. Thresholds are chosen only when
    out-of-fold decisions meet the requested empirical precision and support
    floors. Under-supported levels abstain.
    """
    if table.empty:
        raise ValueError("Geographic label table is empty.")
    target_precisions = target_precisions or GEOGRAPHIC_TARGET_PRECISION
    feature_columns = [column for column in GEOGRAPHIC_FEATURE_COLUMNS if column in table.columns]
    if len(feature_columns) != len(GEOGRAPHIC_FEATURE_COLUMNS):
        missing = sorted(set(GEOGRAPHIC_FEATURE_COLUMNS) - set(feature_columns))
        raise ValueError(f"Missing geographic model features: {missing}")

    matrix = table[feature_columns].fillna(0).to_numpy(dtype=float)
    statuses = table["label_status"].to_numpy()
    label_levels = table["label_level"].to_numpy()
    groups = table["cv_group"].astype(str).to_numpy()
    options = {
        "n_splits": n_splits,
        "random_state": random_state,
        "regularization_values": regularization_values,
        "minimum_positive_groups": minimum_positive_groups,
        "minimum_prediction_groups": minimum_prediction_groups,
        "minimum_predictions": minimum_predictions,
    }
    per_level = {
        level: _fit_level(
            level, matrix, statuses, label_levels, groups, float(target_precisions[level]), options
        )
        for level in GEOGRAPHIC_LEVELS
    }

    active_levels = [level for level in GEOGRAPHIC_LEVELS if per_level[level]["status"] == "active"]
    model_bundle = {
        "model_type": "hierarchical_geographic_logistic_regression",
        "model_version": 1,
        "feature_columns": feature_columns,
        "levels": per_level,
        "specificity_order": GEOGRAPHIC_LEVELS,
        "target_precisions": dict(target_precisions),
        "minimum_work_recall": minimum_recall,
        "minimum_positive_groups": minimum_positive_groups,
        "minimum_prediction_groups": minimum_prediction_groups,
        "active_levels": active_levels,
        "cv_grouping": "connected_artwork_and_target_entities",
        "cv_metrics_are_threshold_tuning_estimates": True,
    }
    oof_scores_by_row = defaultdict(dict)
    for level in active_levels:
        level_model = per_level[level]
        for row_index, probability in zip(level_model["oof_row_indices"], level_model["oof_probabilities"]):
            oof_scores_by_row[row_index][level] = probability
    model_bundle["threshold_tuning_work_metrics"] = evaluate_hierarchical_work_predictions(
        table,
        oof_scores_by_row,
        per_level,
        _SPECIFICITY,
        records=table.attrs.get("artwork_statuses"),
    )
    model_bundle["minimum_work_recall_met"] = (
        model_bundle["threshold_tuning_work_metrics"]["recall"] >= minimum_recall
    )
    for level_model in per_level.values():
        level_model.pop("oof_row_indices", None)
        level_model.pop("oof_probabilities", None)
    return model_bundle


# ---------------------------------------------------------------------------
# Hierarchical model: evaluation
# ---------------------------------------------------------------------------

def _select_hierarchical_level(candidate_level, level_scores, levels, specificity):
    """Most specific active level whose threshold is crossed for this candidate, or None.

    A level coarser than the candidate's own level can be chosen (a building
    can be predicted at city level), never a finer one.
    """
    candidate_level_rank = specificity.get(candidate_level)
    if candidate_level_rank is None:
        return None
    decisions = []
    for level, probability in level_scores.items():
        level_model = levels[level]
        if level_model["status"] != "active" or candidate_level_rank > specificity[level]:
            continue
        if probability >= level_model["threshold"]:
            decisions.append((specificity[level], -probability, level))
    return min(decisions)[2] if decisions else None


def evaluate_hierarchical_work_predictions(
    table,
    scores_by_row,
    levels,
    specificity,
    records=None,
):
    """Compute correct-location precision/recall for one hierarchical choice per artwork.

    `records` may be a list of records, a precomputed {artwork_id: place_status}
    dict or None (statuses then come from the table). A depicted artwork
    predicted at the wrong place counts as both a false positive and a false
    negative ("mislocated"). "undeterminable" artworks are left out: no
    decision about them can be judged right or wrong.
    """
    rows_by_artwork = defaultdict(list)
    for row_index, row in table.iterrows():
        rows_by_artwork[row["artwork_id"]].append((row_index, row))

    if isinstance(records, dict):
        artwork_statuses = records
    elif records is None:
        artwork_statuses = {row["artwork_id"]: row.get("place_status") for _, row in table.iterrows()}
    else:
        artwork_statuses = _artwork_statuses(records)

    true_positives = false_positives = true_negatives = false_negatives = 0
    mislocated_count = 0
    evaluated_artworks = 0
    for artwork_id, actual_status in artwork_statuses.items():
        if actual_status == "undeterminable":
            continue
        evaluated_artworks += 1
        decisions = []
        for row_index, row in rows_by_artwork.get(artwork_id, []):
            row_scores = scores_by_row.get(row_index, {})
            selected_level = _select_hierarchical_level(row.get("candidate_level"), row_scores, levels, specificity)
            if selected_level is None:
                continue
            actual_level = row.get("label_level")
            correct_at_level = (
                row.get("label_status") == "match"
                and actual_level in specificity
                and specificity[actual_level] <= specificity[selected_level]
            )
            decisions.append((specificity[selected_level], -row_scores[selected_level], correct_at_level))

        selected = min(decisions) if decisions else None
        actual_depicted = actual_status == "depicted"
        predicted = selected is not None
        if actual_depicted and predicted and selected[2]:
            true_positives += 1
        elif actual_depicted:
            false_negatives += 1
            if predicted:
                false_positives += 1
                mislocated_count += 1
        elif predicted:
            false_positives += 1
        else:
            true_negatives += 1

    precision = _ratio(true_positives, true_positives + false_positives)
    recall = _ratio(true_positives, true_positives + false_negatives)
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "mislocated_count": mislocated_count,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "accuracy": _ratio(true_positives + true_negatives, evaluated_artworks),
        "artwork_count": evaluated_artworks,
        "depicted_artwork_count": true_positives + false_negatives,
        "metrics_are_threshold_tuning_estimates": True,
    }


def _score_rows(model, table):
    """Score every row of `table` with each active level compatible with its candidate level."""
    scores_by_row = defaultdict(dict)
    if table.empty:
        return scores_by_row
    feature_columns = model.get("feature_columns", GEOGRAPHIC_FEATURE_COLUMNS)
    matrix = table[feature_columns].fillna(0).to_numpy(dtype=float)
    for row_position, (_, row) in enumerate(table.iterrows()):
        candidate_level = row.get("candidate_level")
        if candidate_level not in _SPECIFICITY:
            continue
        for level, level_model in model["levels"].items():
            if level_model["status"] != "active" or _SPECIFICITY[candidate_level] > _SPECIFICITY[level]:
                continue
            scores_by_row[row_position][level] = float(
                level_model["model"].predict_proba(matrix[row_position:row_position + 1])[0][1]
            )
    return scores_by_row


def _count_level_decisions(level_totals, level_prediction_groups, model, table, scores_by_row):
    """Add the candidate-level confusion counts of one test fold, per level."""
    for row_position, (_, row) in enumerate(table.iterrows()):
        if row.get("label_status") == "unevaluable":
            continue
        for level, probability in scores_by_row.get(row_position, {}).items():
            counts = level_totals[level]
            actual_positive = (
                row.get("label_status") == "match" and row.get("label_level") in _levels_up_to(level)
            )
            predicted_positive = probability >= model["levels"][level]["threshold"]
            counts["evaluated"] += 1
            if actual_positive:
                counts["actual_positives"] += 1
            if predicted_positive:
                counts["predictions"] += 1
                level_prediction_groups[level].add(str(row.get("cv_group")))
                counts["true_positives" if actual_positive else "false_positives"] += 1
            elif actual_positive:
                counts["false_negatives"] += 1


def evaluate_hierarchical_grouped_cv(
    records,
    table,
    *,
    n_splits=5,
    random_state=42,
    training_options=None,
):
    """Evaluate a full level/threshold selection process on outer grouped folds.

    For each outer fold, `train_hierarchical_model` (thresholds included) is
    fitted on the training artworks only and applied to the test artworks.
    Returns work-level metrics and, per level, the candidate-level precision
    used to decide which levels are deployable.
    """
    artwork_statuses = _artwork_statuses(records)
    artwork_groups = table.attrs.get("artwork_cv_groups") or _artwork_target_cv_groups(records)
    artwork_ids = list(artwork_statuses)
    group_values = [str(artwork_groups.get(artwork_id, f"artwork:{artwork_id}")) for artwork_id in artwork_ids]
    effective_splits = min(n_splits, len(set(group_values)))
    if effective_splits < 2:
        raise ValueError("At least two geographic CV groups are required.")

    y = np.asarray([int(artwork_statuses[artwork_id] == "depicted") for artwork_id in artwork_ids])
    splitter = GroupKFold(n_splits=effective_splits)
    totals = Counter()
    level_totals = {level: Counter() for level in GEOGRAPHIC_LEVELS}
    level_prediction_groups = {level: set() for level in GEOGRAPHIC_LEVELS}
    folds = []
    training_options = training_options or {}

    for fold_number, (train_work_idx, test_work_idx) in enumerate(
        splitter.split(np.zeros(len(artwork_ids)), y, group_values),
        start=1,
    ):
        train_ids = {artwork_ids[index] for index in train_work_idx}
        test_ids = {artwork_ids[index] for index in test_work_idx}
        train_table = table[table["artwork_id"].isin(train_ids)].reset_index(drop=True)
        test_table = table[table["artwork_id"].isin(test_ids)].reset_index(drop=True)
        train_table.attrs["artwork_statuses"] = {
            artwork_id: artwork_statuses[artwork_id] for artwork_id in train_ids
        }
        train_table.attrs["artwork_cv_groups"] = {
            artwork_id: artwork_groups.get(artwork_id, f"artwork:{artwork_id}")
            for artwork_id in train_ids
        }

        if train_table.empty:
            model = {
                "levels": {
                    level: {"status": "no_training_rows", "threshold": None}
                    for level in GEOGRAPHIC_LEVELS
                },
                "specificity_order": GEOGRAPHIC_LEVELS,
            }
        else:
            model = train_hierarchical_model(train_table, **training_options)

        scores_by_row = _score_rows(model, test_table)
        metrics = evaluate_hierarchical_work_predictions(
            test_table,
            scores_by_row,
            model["levels"],
            _SPECIFICITY,
            records={artwork_id: artwork_statuses[artwork_id] for artwork_id in test_ids},
        )
        _count_level_decisions(level_totals, level_prediction_groups, model, test_table, scores_by_row)
        for metric in (
            "true_positives",
            "false_positives",
            "true_negatives",
            "false_negatives",
            "mislocated_count",
            "artwork_count",
            "depicted_artwork_count",
        ):
            totals[metric] += metrics[metric]
        folds.append({
            "fold": fold_number,
            "test_artworks": len(test_ids),
            "active_levels": [
                level for level in GEOGRAPHIC_LEVELS if model["levels"][level]["status"] == "active"
            ],
            "metrics": metrics,
        })

    tp = totals["true_positives"]
    fp = totals["false_positives"]
    fn = totals["false_negatives"]
    tn = totals["true_negatives"]
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    per_level = {}
    for level, counts in level_totals.items():
        predictions = counts["predictions"]
        actual_positives = counts["actual_positives"]
        true_positives = counts["true_positives"]
        per_level[level] = {
            "precision": true_positives / predictions if predictions else None,
            "recall": true_positives / actual_positives if actual_positives else None,
            "true_positives": true_positives,
            "false_positives": counts["false_positives"],
            "false_negatives": counts["false_negatives"],
            "predictions": predictions,
            "prediction_groups": len(level_prediction_groups[level]),
            "evaluated_candidates": counts["evaluated"],
        }
    return {
        "fold_count": effective_splits,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "mislocated_count": totals["mislocated_count"],
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "accuracy": _ratio(tp + tn, totals["artwork_count"]),
        "artwork_count": totals["artwork_count"],
        "depicted_artwork_count": totals["depicted_artwork_count"],
        "metrics_by_level": per_level,
        "grouping": "connected_artwork_and_target_entities",
        "thresholds_tuned_inside_outer_training_folds": True,
        "folds": folds,
    }


# ---------------------------------------------------------------------------
# Hierarchical model: prediction
# ---------------------------------------------------------------------------

def _level_decisions(feature_row, candidate_level, model, specificity):
    """Score a candidate with every usable level.

    Returns (level_scores, decisions): the probability of each deployable
    level compatible with the candidate's level, and the levels whose
    threshold is crossed, most specific first.
    """
    level_scores = {}
    decisions = []
    candidate_level_rank = specificity.get(candidate_level)
    if candidate_level_rank is None:
        return level_scores, decisions
    deployable_levels = model.get("deployable_levels")
    for level in model["specificity_order"]:
        level_model = model["levels"][level]
        if level_model["status"] != "active":
            continue
        if deployable_levels is not None and level not in deployable_levels:
            continue
        if candidate_level_rank > specificity[level]:
            continue
        probability = float(level_model["model"].predict_proba(feature_row)[0][1])
        level_scores[level] = probability
        if probability >= level_model["threshold"]:
            decisions.append({
                "level": level,
                "model_score": probability,
                "confidence": level_model.get("outer_cv_precision", level_model.get("oof_precision")),
                "confidence_basis": "outer_grouped_cv_empirical_precision",
                "confidence_support": level_model.get(
                    "outer_cv_prediction_groups",
                    level_model.get("oof_prediction_groups"),
                ),
                "target_precision": level_model["target_precision"],
                "threshold": level_model["threshold"],
            })
    decisions.sort(key=lambda decision: (specificity[decision["level"]], -decision["model_score"]))
    return level_scores, decisions


def predict_with_hierarchical_model(
    records,
    candidates_by_artwork,
    model,
    *,
    gazetteer_conn=None,
    nominatim_client=None,
    geocode_top_k=3,
):
    """Predict the most specific sufficiently precise geographic level.

    Only the `geocode_top_k` candidates with the best NER score are geocoded,
    jointly so that they disambiguate each other.
    For each artwork, returns `place_status_predicted`, the accepted
    `prediction` (or None) and every scored candidate.
    """
    if geocode_top_k < 1:
        raise ValueError("geocode_top_k must be at least 1.")

    feature_columns = model["feature_columns"]
    specificity = {level: index for index, level in enumerate(model["specificity_order"])}
    alternatives_cache = {}
    results = {}

    for record in records:
        artwork = _artwork(record)
        artwork_id = artwork.get("id")
        grouped_candidates = sorted(
            group_candidates_by_place(candidates_by_artwork.get(artwork_id, [])).values(),
            key=lambda occurrences: max(occurrence.get("score_ner", 0.0) for occurrence in occurrences),
            reverse=True,
        )

        geocoded = grouped_candidates[:geocode_top_k]
        locations = _geocode_artwork(geocoded, alternatives_cache, gazetteer_conn, nominatim_client)
        locations += [None] * (len(grouped_candidates) - len(geocoded))

        scored_candidates = []
        for occurrences, location in zip(grouped_candidates, locations):
            place_name = occurrences[0]["place"]
            features = _hierarchical_candidate_features(occurrences, location, artwork)
            feature_row = np.asarray([[features[column] for column in feature_columns]], dtype=float)
            candidate_level = _candidate_geographic_level(location)
            level_scores, decisions = _level_decisions(feature_row, candidate_level, model, specificity)
            scored_candidates.append({
                "place": place_name,
                "score_ner_max": max(occurrence.get("score_ner", 0.0) for occurrence in occurrences),
                "geolocation": location,
                "candidate_level": candidate_level,
                "level_scores": level_scores,
                "decision": decisions[0] if decisions else None,
                "champs": sorted({occurrence["field"] for occurrence in occurrences}),
            })

        accepted = sorted(
            (candidate for candidate in scored_candidates if candidate["decision"]),
            key=lambda candidate: (
                specificity[candidate["decision"]["level"]],
                -candidate["decision"]["model_score"],
                -candidate["score_ner_max"],
            ),
        )
        results[artwork_id] = {
            "place_status_predicted": bool(accepted),
            "prediction": accepted[0] if accepted else None,
            "candidates": scored_candidates,
        }
    return results


# ---------------------------------------------------------------------------
# Legacy evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_binary_predictions(records, heuristic_results, threshold=0.5, review_margin=0.1):
    """Evaluate whether an artwork depicts a place, regardless of place identity.

    An artwork is predicted "depicted" when at least one candidate reaches
    `threshold`. Candidates within `review_margin` below the threshold are
    flagged for review. See `evaluate_place_identity` for the identity check.
    """
    if review_margin < 0:
        raise ValueError("review_margin must be non-negative.")

    true_positives = false_positives = true_negatives = false_negatives = 0
    false_positive_ids = []
    false_negative_ids = []
    flagged_predictions = []
    errors = []

    for record in records:
        artwork_id = _artwork(record).get("id")
        actual = _place_status(record) == "depicted"
        artwork_places = heuristic_results.get(artwork_id, [])
        accepted_places = [
            place for place in artwork_places if place.get("score_heuristique", 0) >= threshold
        ]
        predicted = bool(accepted_places)
        rejected_places = [
            {"place": place.get("place"), "score_heuristique": place.get("score_heuristique", 0)}
            for place in artwork_places
            if threshold - review_margin <= place.get("score_heuristique", 0) < threshold
        ]
        if not predicted and rejected_places:
            flagged_predictions.append({"id": artwork_id, "rejected_places": rejected_places})

        if actual and predicted:
            true_positives += 1
        elif predicted:
            false_positives += 1
            false_positive_ids.append(artwork_id)
        elif actual:
            false_negatives += 1
            false_negative_ids.append(artwork_id)
        else:
            true_negatives += 1

        if actual != predicted:
            errors.append({
                "id": artwork_id,
                "actual": actual,
                "predicted": predicted,
                "place_status": _place_status(record),
                "predicted_places": [
                    {"place": place.get("place"), "score_heuristique": place.get("score_heuristique", 0)}
                    for place in accepted_places
                ],
            })

    total = true_positives + false_positives + true_negatives + false_negatives
    precision = _ratio(true_positives, true_positives + false_positives)
    recall = _ratio(true_positives, true_positives + false_negatives)
    return {
        "total": total,
        "positive_count": true_positives + false_negatives,
        "negative_count": true_negatives + false_positives,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "false_positive_ids": false_positive_ids,
        "false_negative_ids": false_negative_ids,
        "flagged_ids": [item["id"] for item in flagged_predictions],
        "flagged_predictions": flagged_predictions,
        "accuracy": _ratio(true_positives + true_negatives, total),
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "errors": errors,
    }


def evaluate_place_identity(
    records,
    heuristic_results,
    fuzzy_threshold=0.85,
    *,
    gazetteer_conn=None,
    nominatim_client=None,
    coordinate_tolerance_km=5.0,
    city_tolerance_km=30.0,
):
    """
    Evaluate the top-1 candidate by name and, when possible, by distance.

    Distance to the annotated point uses `coordinate_tolerance_km`. The city
    metric compares the candidate with the GeoNames centre of the annotated
    city within `city_tolerance_km`; it approximates, rather than tests,
    membership of an administrative boundary.
    """
    if coordinate_tolerance_km < 0 or city_tolerance_km < 0:
        raise ValueError("Coordinate tolerances must be non-negative.")
    can_geocode = gazetteer_conn is not None or nominatim_client is not None

    total_depicted = 0
    correct_identity = 0
    wrong_identity = []
    no_candidate = []
    coordinate_matches = 0
    coordinate_evaluable = 0
    coordinate_resolved_candidates = 0
    coordinate_errors = []
    city_matches = 0
    city_evaluable = 0
    city_resolved_candidates = 0
    city_errors = []

    for record in records:
        annotations = record.get("annotations", {})
        if annotations.get("place_status") != "depicted":
            continue
        artwork_id = _artwork(record).get("id")
        total_depicted += 1

        places = heuristic_results.get(artwork_id, [])
        if not places:
            no_candidate.append(artwork_id)
        depicted_place = annotations.get("depicted_place") or {}
        real_place = depicted_place.get("name")
        top_candidate = places[0] if places else None
        top_place = top_candidate.get("place") if top_candidate else None

        identity_matches = top_place is not None and fuzzy_match(top_place, real_place, fuzzy_threshold)
        if identity_matches:
            correct_identity += 1

        # Distance to the annotated point.
        actual_coordinates = _valid_coordinates(depicted_place)
        predicted_coordinates = _valid_coordinates(top_candidate)
        if predicted_coordinates is None and top_place and can_geocode:
            resolved_candidate = resolve_location(top_place, gazetteer_conn, nominatim_client=nominatim_client)
            if resolved_candidate:
                predicted_coordinates = _valid_coordinates(resolved_candidate)

        distance_to_actual_km = None
        if actual_coordinates is not None:
            coordinate_evaluable += 1
            if predicted_coordinates is not None:
                coordinate_resolved_candidates += 1
                distance_to_actual_km = haversine_distance_km(*actual_coordinates, *predicted_coordinates)
                if distance_to_actual_km <= coordinate_tolerance_km:
                    coordinate_matches += 1
            if distance_to_actual_km is None or distance_to_actual_km > coordinate_tolerance_km:
                coordinate_errors.append({
                    "id": artwork_id,
                    "predicted": top_place,
                    "actual": real_place,
                    "distance_km": distance_to_actual_km,
                })

        if top_candidate and not identity_matches:
            wrong_identity.append({
                "id": artwork_id,
                "predicted": top_place,
                "actual": real_place,
                "distance_km": distance_to_actual_km,
            })

        # Distance to the centre of the annotated city.
        city_name = depicted_place.get("city")
        if city_name and can_geocode and actual_coordinates is not None:
            city_location = resolve_location(
                city_name,
                gazetteer_conn,
                reference_coordinates=actual_coordinates,
                nominatim_client=nominatim_client,
            )
            if city_location is not None:
                city_evaluable += 1
                city_coordinates = _valid_coordinates(city_location)
                distance = None
                if predicted_coordinates is not None:
                    city_resolved_candidates += 1
                    distance = haversine_distance_km(*city_coordinates, *predicted_coordinates)
                    if distance <= city_tolerance_km:
                        city_matches += 1
                if distance is None or distance > city_tolerance_km:
                    city_errors.append({
                        "id": artwork_id,
                        "predicted": top_place,
                        "city": city_name,
                        "distance_km": distance,
                    })

    def optional_ratio(numerator, denominator):
        """numerator / denominator, or None when nothing could be evaluated."""
        return numerator / denominator if denominator else None

    return {
        "total_depicted": total_depicted,
        "correct_identity": correct_identity,
        "wrong_identity_count": len(wrong_identity),
        "no_candidate_count": len(no_candidate),
        "identity_accuracy": _ratio(correct_identity, total_depicted),
        "wrong_identity": wrong_identity,
        "no_candidate_ids": no_candidate,
        "coordinate_matches": coordinate_matches,
        "coordinate_evaluable": coordinate_evaluable,
        "coordinate_resolved_candidates": coordinate_resolved_candidates,
        "coordinate_candidate_coverage": optional_ratio(coordinate_resolved_candidates, coordinate_evaluable),
        "coordinate_tolerance_km": coordinate_tolerance_km,
        "coordinate_accuracy": optional_ratio(coordinate_matches, coordinate_evaluable),
        "coordinate_accuracy_when_geocoded": optional_ratio(coordinate_matches, coordinate_resolved_candidates),
        "coordinate_errors": coordinate_errors,
        "city_matches": city_matches,
        "city_evaluable": city_evaluable,
        "city_resolved_candidates": city_resolved_candidates,
        "city_candidate_coverage": optional_ratio(city_resolved_candidates, city_evaluable),
        "city_tolerance_km": city_tolerance_km,
        "city_proximity_accuracy": optional_ratio(city_matches, city_evaluable),
        "city_proximity_accuracy_when_geocoded": optional_ratio(city_matches, city_resolved_candidates),
        "city_errors": city_errors,
    }


def compare_heuristic_places_with_annotations(records, heuristic_results, threshold=0.4):
    """Compare heuristic acceptance with singular depicted-place annotations."""
    real_depicted_artworks = []
    heuristic_accepted_artworks = []
    artworks_with_discrepancy = []
    real_depicted_places = 0
    heuristic_accepted_places = 0

    for record in records:
        artwork_id = _artwork(record).get("id")
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
