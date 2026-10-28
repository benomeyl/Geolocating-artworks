"""
Data filtering, candidate extraction and place-identification scoring.

Ce module remplace le score heuristique à poids fixés à la main (ancien
`score_candidate` + `FIELDS_WEIGHTS`/bonus/malus + recherche en grille
`optimize_heuristic_parameters`) par une régression logistique apprise sur
le jeu annoté. Tout reste strictement textuel (champ d'origine, marqueurs
lexicaux, score du NER) — aucune donnée visuelle n'entre dans les features.

Changement de grain important par rapport à la version précédente :
l'ancienne évaluation (`evaluate_binary_predictions`) ne vérifiait que si
UNE PLACE QUELCONQUE dépassait le seuil pour une œuvre annotée "depicted" —
jamais si c'était la BONNE. Deux œuvres pouvaient donc être comptées comme
des succès alors que le lieu retenu était en réalité le lieu de création,
pas le sujet représenté. En passant à une supervision PAR CANDIDAT (un
label par lieu détecté, pas par œuvre), le modèle apprend aussi à préférer
le bon candidat face à des distracteurs de la même œuvre (ex: "Rome"
mentionné en contexte de création à côté du vrai sujet "Paris") — chose
que l'ancien score, qui évaluait chaque candidat indépendamment contre un
seuil global, ne pouvait pas faire. `evaluate_place_identity` (en bas de
ce fichier) rend cette distinction mesurable.

Usage typique :

    candidates_by_artwork = extract_all_candidates(records, ner)
    table = build_training_table(records, candidates_by_artwork)
    model = train_logistic_model(table)

    results = predict_with_model(records, candidates_by_artwork, model)
    detection_metrics = evaluate_binary_predictions(records, results, threshold=model["threshold"])
    identity_metrics = evaluate_place_identity(records, results)
"""

import difflib
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
import unicodedata

try:
    from src.geolocation.gazetter import haversine_distance_km, resolve_location
except ModuleNotFoundError:
    from geolocation.gazetter import haversine_distance_km, resolve_location

_GENERIC_WORDS = {
    # articles / prépositions
    "the", "of", "de", "du", "des", "la", "le", "les", "l", "d", "di", "del",
    "der", "die", "das", "van", "von", "al", "el",
    # types de lieux courants (en, fr, de, it, es, nl)
    "cathedral", "cathedrale", "cathedrale", "church", "eglise", "chapel",
    "chapelle", "abbey", "abbaye", "castle", "chateau", "palace", "palais",
    "bridge", "pont", "gate", "porte", "tower", "tour", "square", "place",
    "street", "rue", "avenue", "boulevard", "station", "gare", "market",
    "marche", "harbour", "harbor", "port", "basilica", "basilique",
    "kathedrale", "kirche", "schloss", "burg", "dom", "duomo", "chiesa",
    "iglesia", "catedral", "kerk", "poort", "brug", "toren",
}

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

# Ordre fixe des champs texte considérés. Remplace FIELDS_WEIGHTS : au lieu
# d'un poids choisi à la main par champ, chaque champ devient une colonne
# binaire ("field_title", "field_themes", ...) dont le modèle apprend le
# coefficient lui-même.
FIELD_NAMES = [
    "title", "alt_titles", "themes", "subjects", "description", "short_description", "styles",
]

# Seuil plancher appliqué à l'extraction NER (voir extract_candidate_artwork).
# Volontairement bas : le score NER devient une FEATURE continue
# (score_ner_max) que le modèle pondère lui-même, plutôt qu'une coupure
# dure choisie à la main (l'ancien paramètre `ner_threshold` de la grille).
DEFAULT_NER_FLOOR = 0.2

# Colonnes de features utilisées par le modèle, dans un ordre fixe
# (nécessaire pour que l'entraînement et la prédiction s'accordent).
FEATURE_COLUMNS = [f"field_{field}" for field in FIELD_NAMES] + [
    "n_fields",
    "has_positive_marker",
    "has_negative_marker",
    "score_ner_max",
]

GEOGRAPHIC_LEVELS = [
    "exact_place",
    "neighborhood",
    "city",
    "region",
    "country",
]
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
]


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


# Mots qui ne sont jamais un lieu à eux seuls (artefacts NER : "of", "La").
_ENTITY_STOPWORDS = _GENERIC_WORDS | {"a", "an", "and", "with", "et", "y", "e", "und"}
# Connecteurs qui, en tête d'entité, signalent un début tronqué
# ("of Control" pour "Line of Control").
_LEADING_CONNECTORS = {
    "of", "the", "de", "du", "des", "del", "della", "di", "da", "von", "van", "der",
}
# Séparateurs autorisés entre deux morceaux d'une même entité ("N" "." "Y").
# Pas d'espace : deux entités séparées par un espace ou ". " sont distinctes
# (« Dresden. Albert Square »).
_ENTITY_JOINERS = set(".-–'’")
_STRIPPED_PUNCTUATION = " \t\n,;:!?()[]{}\"«»“”‘’'"


def _is_word_char(character):
    return character.isalnum() or unicodedata.category(character) == "Mn"


def _in_word(text, index):
    """True if text[index] belongs to a word; an apostrophe between letters does (« Akbar's »)."""
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


def _clean_entity_spans(text, entities):
    """
    Corrige les spans LOC bruts du NER à partir du texte source.

    Les noms sont relus dans `text` via les offsets (et non depuis le
    `word` décodé par le tokenizer, qui insère des espaces : « Saint - Lazare »).
    Les spans coupés au milieu d'un mot sont étendus au mot entier, les
    morceaux adjacents d'une même entité fusionnés, et les spans qui ne
    peuvent pas être un nom de lieu (une lettre, un mot-outil) écartés.
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

    cleaned = []
    for span in spans:
        start, end = span["start"], span["end"]
        while start < end and text[start] in _STRIPPED_PUNCTUATION:
            start += 1
        while end > start and text[end - 1] in _STRIPPED_PUNCTUATION:
            end -= 1
        # Un point final n'est gardé que dans une abréviation (« N.Y. »).
        if end > start and text[end - 1] == "." and "." not in text[start:end - 1]:
            end -= 1

        elision = re.match(r"d['’]", text[start:end])
        if elision:
            start += elision.end()

        first_word = re.match(r"\w+", text[start:end])
        if (
            first_word
            and first_word.group(0) in _LEADING_CONNECTORS
            and first_word.end() < end - start
        ):
            previous_word = re.search(r"(\w+)\s+$", text[:start])
            if previous_word and previous_word.group(1)[:1].isupper():
                start = previous_word.start(1)
            else:
                start += first_word.end()
                while start < end and text[start].isspace():
                    start += 1

        surface = text[start:end]
        letter_count = sum(character.isalpha() for character in surface)
        if letter_count < 2 or _normalize_place_name(surface) in _ENTITY_STOPWORDS:
            continue
        cleaned.append({**span, "start": start, "end": end, "surface": surface})
    return cleaned


def extract_candidate_artwork(artwork, ner, threshold=DEFAULT_NER_FLOOR):
    """
    Extract detected LOC candidates, including their source field and position
    within that field (required for context scoring).

    `threshold` reste volontairement bas par défaut (DEFAULT_NER_FLOOR) :
    c'est un filtre anti-bruit, pas le seuil d'acceptation final. Le score
    NER complet est conservé comme feature (score_ner_max) pour le modèle.
    Il est appliqué après nettoyage des spans (_clean_entity_spans), sur le
    score maximal des morceaux fusionnés.
    """
    fields = {
        "title": artwork.get("title") or "",
        "alt_titles": " / ".join(artwork.get("alt_titles") or []),
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
    Lance extract_candidate_artwork sur tous les enregistrements et met les
    résultats en cache par artwork_id, pour ne lancer le NER qu'une seule
    fois (les candidats bruts sont ensuite réutilisés par la construction
    de la table d'entraînement ET par la prédiction).
    """
    candidates_by_artwork = {}
    for record in records:
        artwork = record.get("artwork", record)
        artwork_id = artwork.get("id")
        candidates_by_artwork[artwork_id] = extract_candidate_artwork(
            artwork, ner, threshold=threshold
        )
    return candidates_by_artwork


def local_context(text, start, end, window=60):
    """
    Get the local context around a slice of text.
    """
    return text[max(0, start - window):end + window].lower()


def group_candidates_by_place(candidates):
    """
    Regroupe les candidats détectés par lieu (insensible à la casse).
    Remplace la partie "groupby" de l'ancien merge_candidate_places — la
    partie "scoring" en est désormais séparée (voir candidate_features).
    """
    groups = defaultdict(list)
    for candidate in candidates:
        place_key = candidate["place"].lower().strip()
        groups[place_key].append(candidate)
    return groups


def candidate_features(occurrences):
    """
    Features BRUTES pour un lieu fusionné (un groupe d'occurrences du même
    lieu dans une œuvre). Remplace l'ancien score_candidate : au lieu de
    combiner ces signaux en un score avec des poids fixés à la main, on les
    expose comme colonnes et c'est la régression logistique qui apprend
    leurs poids.
    """
    found_fields = {c["field"] for c in occurrences}
    has_positive = any(
        any(
            marker in local_context(c["source_text"], c["start"], c["end"])
            for marker in POSITIVES_LABELS
        )
        for c in occurrences
    )
    has_negative = any(
        any(
            marker in local_context(c["source_text"], c["start"], c["end"])
            for marker in NEGATIVES_LABELS
        )
        for c in occurrences
    )

    features = {f"field_{field}": int(field in found_fields) for field in FIELD_NAMES}
    features["n_fields"] = len(found_fields)
    features["has_positive_marker"] = int(has_positive)
    features["has_negative_marker"] = int(has_negative)
    features["score_ner_max"] = max(c["score_ner"] for c in occurrences)
    return features



# Types d'objets AIC (artwork_type_title) qui ne représentent pas de scène :
# le lieu cité dans leur titre est une origine (« Kantharos, Attica »).
# Les types figuratifs (Print, Photograph, Painting...) et les types
# inconnus valent 0.
NON_PICTORIAL_ARTWORK_TYPES = {
    "vessel", "coin", "glass", "costume and accessories", "sculpture", "ceramics",
    "decorative arts", "metalwork", "furniture", "furnishings", "religious/ritual object",
    "arms", "armor", "architectural fragment", "textile", "coverings and hangings",
    "mask", "funerary object", "basketry", "design",
}


def _mentions(text, place_name):
    place = _normalize_place_name(place_name)
    return bool(place) and f" {place} " in f" {_normalize_place_name(text)} "


def artwork_context_features(artwork, place_name):
    """
    Indices de contexte AIC pour un candidat (voir aic.CONTEXT_FIELDS).

    Ces champs ne produisent jamais de candidat : ils indiquent seulement
    si le lieu détecté dans le titre/la description est aussi le lieu
    d'origine de l'œuvre, apparaît dans la biographie de l'artiste
    (« born Leiden ») ou dans la provenance / l'historique d'expositions.
    Le modèle apprend le poids de chaque indice. Pas d'indicateur « champs
    présents » : il ne marquerait que les œuvres AIC, et le modèle l'utilisait
    comme raccourci de source (21 % de lieux représentés dans les œuvres AIC
    du gold contre 70 % ailleurs). Il reste calculé, hors
    GEOGRAPHIC_FEATURE_COLUMNS, pour les modèles entraînés avant ce retrait.
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


def _normalize_place_name(name):
    text = unicodedata.normalize("NFKD", (name or "").lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _core_tokens(name):
    return {t for t in _normalize_place_name(name).split() if t not in _GENERIC_WORDS}


def fuzzy_match(name_a, name_b, threshold=0.85):
    """
    Match tolérant : ignore accents, articles et mots génériques
    (« cathédrale de Rouen » == « Rouen Cathedral »).
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



def label_candidate(record, place_name, fuzzy_threshold=0.85):
    """
    Label binaire pour un candidat : 1 si place_name correspond au
    depicted_place annoté de cette œuvre, 0 sinon.

    - place_status != "depicted" (y compris "mentioned_only" et "none") :
      TOUS les candidats de l'œuvre sont négatifs. C'est ce qui fournit les
      négatifs "difficiles" (un lieu de création cité à côté du vrai sujet)
      qui manquaient à l'ancienne évaluation au niveau œuvre.
    - place_status == "undeterminable" : à exclure de l'entraînement en
      amont (voir build_training_table), ni positif ni négatif fiable.
    """
    annotations = record.get("annotations", {})
    if annotations.get("place_status") != "depicted":
        return 0
    depicted_places = annotations.get("depicted_places") or [
        annotations.get("depicted_place") or {}
    ]
    return int(any(
        fuzzy_match(place_name, depicted.get("name"), fuzzy_threshold)
        for depicted in depicted_places
    ))


_GEOGRAPHIC_LABEL_LEVELS = {
    "exact_place",
    "neighborhood",
    "city",
    "region",
    "country",
}


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
# Codes GeoNames de zones étendues (région naturelle, archipel, massif...).
_GEONAMES_REGION_CODES = {"RGN", "RGNE", "RGNH", "AREA", "ISLS", "PEN", "MTS", "VAL"}


def _geonames_level(feature_code):
    if feature_code.startswith("PCL"):
        return "country"
    # ADM1/ADM2 (région, département, county US) sont des régions ; ADM3+
    # correspond le plus souvent à une commune (comune, Gemeinde).
    if feature_code.startswith(("ADM1", "ADM2", "ADMD")) or feature_code in _GEONAMES_REGION_CODES:
        return "region"
    if feature_code == "PPLX":
        return "neighborhood"
    if feature_code.startswith(("PPL", "ADM")):
        return "city"
    return "exact_place"


def _nominatim_level(location):
    for key in ("addresstype", "feature_code"):
        level = _NOMINATIM_TYPE_LEVELS.get(str(location.get(key) or "").casefold())
        if level:
            return level
    try:
        place_rank = int(location.get("place_rank") or 0)
    except (TypeError, ValueError):
        place_rank = 0
    place_type = str(location.get("feature_code") or "").casefold()
    place_class = str(location.get("place_class") or "").casefold()
    # Les limites administratives (type "administrative") n'ont pas de type
    # parlant : leur niveau se lit dans place_rank (4 pays, 8 état,
    # 12 county, 16 ville, 20 quartier).
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
    if not location:
        return None
    if location.get("source") == "GeoNames":
        return _geonames_level(str(location.get("feature_code") or "").upper())
    return _nominatim_level(location)


def _add_geographic_features(features, location):
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


def _target_group_key(place):
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
    parents = {}

    def find(item):
        parents.setdefault(item, item)
        if parents[item] != item:
            parents[item] = find(parents[item])
        return parents[item]

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    artwork_nodes = []
    for record in records:
        artwork_id = (record.get("artwork", record)).get("id")
        artwork_node = f"artwork:{artwork_id}"
        find(artwork_node)
        artwork_nodes.append((artwork_id, artwork_node))
        annotations = record.get("annotations", {})
        targets = annotations.get("depicted_places") or [
            annotations.get("depicted_place")
        ]
        for target in targets:
            if not target:
                continue
            target_key = _target_group_key(target)
            if target_key:
                union(artwork_node, f"target:{target_key}")

    return {
        artwork_id: find(artwork_node)
        for artwork_id, artwork_node in artwork_nodes
    }


def geographic_candidate_label(
    record,
    candidate_location,
    *,
    exact_tolerance_km=0.25,
    neighborhood_tolerance_km=2.0,
    city_tolerance_km=25.0,
):
    """Label a candidate at the most specific level supported by its location."""
    if min(exact_tolerance_km, neighborhood_tolerance_km, city_tolerance_km) < 0:
        raise ValueError("Geographic label tolerances must be non-negative.")
    annotations = record.get("annotations", {})
    place_status = annotations.get("place_status")
    if place_status == "undeterminable":
        return {"level": "unevaluable", "status": "unevaluable", "distance_km": None}
    if place_status != "depicted":
        return {"level": "none", "status": "negative", "distance_km": None}

    depicted_places = annotations.get("depicted_places") or []
    if len(depicted_places) > 1:
        labels = [
            geographic_candidate_label(
                {"annotations": {
                    **annotations,
                    "depicted_place": depicted_place,
                    "depicted_places": [],
                }},
                candidate_location,
                exact_tolerance_km=exact_tolerance_km,
                neighborhood_tolerance_km=neighborhood_tolerance_km,
                city_tolerance_km=city_tolerance_km,
            )
            for depicted_place in depicted_places
        ]
        specificity = {
            "exact_place": 0,
            "neighborhood": 1,
            "city": 2,
            "region": 3,
            "country": 4,
        }
        matching_labels = [label for label in labels if label["status"] == "match"]
        if matching_labels:
            return min(matching_labels, key=lambda label: specificity[label["level"]])
        if any(label["status"] == "unevaluable" for label in labels):
            return {
                "level": "unevaluable",
                "status": "unevaluable",
                "distance_km": next(
                    (label.get("distance_km") for label in labels if label.get("distance_km") is not None),
                    None,
                ),
                "candidate_level": labels[0].get("candidate_level"),
                "geocoder_source": labels[0].get("geocoder_source"),
            }
        return labels[0]

    actual = annotations.get("depicted_place") or {}
    precision = actual.get("precision")
    if precision not in _GEOGRAPHIC_LABEL_LEVELS:
        return {"level": "unevaluable", "status": "unevaluable", "distance_km": None}
    actual_coordinates = _valid_coordinates(actual)
    predicted_coordinates = _valid_coordinates(candidate_location)
    if actual_coordinates is None or predicted_coordinates is None:
        return {"level": "unevaluable", "status": "unevaluable", "distance_km": None}

    distance = haversine_distance_km(*actual_coordinates, *predicted_coordinates)
    candidate_level = _candidate_geographic_level(candidate_location)
    hierarchy = candidate_location.get("hierarchy") or {}

    # Hiérarchie de la cible obtenue par géocodage inverse de ses coordonnées
    # (voir build_geographic_label_table) : elle est dans la même langue que
    # celle des candidats (« Sachsen ») alors que l'annotation peut être en
    # anglais (« Saxony »).
    target_hierarchy = actual.get("resolved_hierarchy") or {}

    def value_matches(field):
        target_values = {
            value
            for value in (
                _normalize_place_name(actual.get(field)),
                _normalize_place_name(target_hierarchy.get(field)),
            )
            if value
        }
        resolved_value = _normalize_place_name(hierarchy.get(field))
        if not resolved_value and candidate_level == field:
            resolved_value = _normalize_place_name(candidate_location.get("name"))
        return bool(resolved_value and resolved_value in target_values)

    target_country_code = (
        actual.get("country_code") or target_hierarchy.get("country_code") or ""
    ).upper()
    candidate_country_code = (
        hierarchy.get("country_code") or candidate_location.get("country_code") or ""
    ).upper()
    country_matches = value_matches("country") or bool(
        target_country_code and target_country_code == candidate_country_code
    )
    region_matches = value_matches("region")
    city_matches = value_matches("city")
    neighborhood_name = _normalize_place_name(actual.get("name"))
    resolved_neighborhood = _normalize_place_name(hierarchy.get("neighborhood"))
    neighborhood_matches = bool(
        neighborhood_name
        and (
            resolved_neighborhood == neighborhood_name
            or (
                candidate_level == "neighborhood"
                and _normalize_place_name(candidate_location.get("name")) == neighborhood_name
            )
        )
    )

    matched_level = None
    if precision == "exact_place":
        if candidate_level == "exact_place" and distance <= exact_tolerance_km:
            matched_level = "exact_place"
        elif city_matches:
            matched_level = "city"
    elif precision == "neighborhood":
        if neighborhood_matches or (
            candidate_level == "exact_place" and distance <= neighborhood_tolerance_km
        ):
            matched_level = "neighborhood"
        elif city_matches:
            matched_level = "city"
    elif precision == "city":
        if city_matches or (
            candidate_level in {"exact_place", "neighborhood", "city"}
            and distance <= city_tolerance_km
        ):
            matched_level = "city"
    elif precision == "region" and region_matches:
        matched_level = "region"
    elif precision == "country" and country_matches:
        matched_level = "country"

    if matched_level is None:
        if precision not in {"region", "country"} and region_matches:
            matched_level = "region"
        elif country_matches:
            matched_level = "country"

    if matched_level is None:
        required_hierarchy = {
            "exact_place": ("city", "region", "country"),
            "neighborhood": ("city", "region", "country"),
            "city": ("city", "region", "country"),
            "region": ("region", "country"),
            "country": ("country",),
        }[precision]
        if not any(hierarchy.get(field) for field in required_hierarchy):
            return {
                "level": "unevaluable",
                "status": "unevaluable",
                "distance_km": distance,
                "candidate_level": candidate_level,
                "geocoder_source": candidate_location.get("source"),
            }

    return {
        "level": matched_level or "no_match",
        "status": "match" if matched_level else "negative",
        "distance_km": distance,
        "candidate_level": candidate_level,
        "geocoder_source": candidate_location.get("source"),
    }


def _with_target_hierarchies(record, nominatim_client, reverse_cache):
    """Return a copy of `record` whose depicted places carry `resolved_hierarchy`.

    The hierarchy comes from reverse-geocoding the annotated coordinates, so
    that region/country names are compared in the geocoder's language.
    """
    if nominatim_client is None or not hasattr(nominatim_client, "reverse"):
        return record
    annotations = record.get("annotations") or {}

    def resolve(place):
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
    """Build inspectable hierarchical labels without forcing unresolved cases negative."""
    rows = []
    location_cache = {}
    reverse_cache = {}
    cv_groups = _artwork_target_cv_groups(records)
    for record in records:
        record = _with_target_hierarchies(record, nominatim_client, reverse_cache)
        artwork = record.get("artwork", record)
        artwork_id = artwork.get("id")
        groups = group_candidates_by_place(candidates_by_artwork.get(artwork_id, []))
        for occurrences in groups.values():
            place_name = occurrences[0]["place"]
            cache_key = place_name.casefold().strip()
            if cache_key not in location_cache:
                if gazetteer_conn is None and nominatim_client is None:
                    location_cache[cache_key] = None
                else:
                    location_cache[cache_key] = resolve_location(
                        place_name,
                        gazetteer_conn,
                        nominatim_client=nominatim_client,
                        include_hierarchy=True,
                    )
            location = location_cache[cache_key]

            label = geographic_candidate_label(
                record,
                location,
                exact_tolerance_km=exact_tolerance_km,
                neighborhood_tolerance_km=neighborhood_tolerance_km,
                city_tolerance_km=city_tolerance_km,
            )
            row = _add_geographic_features(candidate_features(occurrences), location)
            row.update(artwork_context_features(artwork, place_name))
            row.update({
                "artwork_id": artwork_id,
                "cv_group": cv_groups.get(artwork_id, f"artwork:{artwork_id}"),
                "place_status": record.get("annotations", {}).get("place_status"),
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
    table.attrs["artwork_statuses"] = {
        (record.get("artwork", record)).get("id"):
        record.get("annotations", {}).get("place_status")
        for record in records
    }
    table.attrs["artwork_cv_groups"] = cv_groups
    return table


def build_training_table(records, candidates_by_artwork, fuzzy_threshold=0.85):
    """
    Construit une ligne par (œuvre, lieu fusionné), avec ses features
    brutes et son label. C'est la table que train_logistic_model consomme.
    """
    rows = []
    for record in records:
        artwork = record.get("artwork", record)
        artwork_id = artwork.get("id")
        if record.get("annotations", {}).get("place_status") == "undeterminable":
            continue

        groups = group_candidates_by_place(candidates_by_artwork.get(artwork_id, []))
        for occurrences in groups.values():
            row = candidate_features(occurrences)
            row["artwork_id"] = artwork_id
            row["place"] = occurrences[0]["place"]  # orthographe/casse d'origine
            row["label"] = label_candidate(record, occurrences[0]["place"], fuzzy_threshold)
            rows.append(row)

    return pd.DataFrame(rows)


def _best_threshold(labels, probabilities, minimum_recall=0.8):
    """
    Choisit, sur une courbe précision-rappel, le seuil le plus permissif
    qui atteint encore minimum_recall (même logique que l'ancien paramètre
    minimum_recall de optimize_heuristic_parameters) ; à défaut, le seuil
    qui maximise le F1.
    """
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    # precision_recall_curve renvoie un point de plus que thresholds
    # (le point (precision=1, recall=0) sans seuil associé) : on aligne
    # les tableaux en ignorant ce dernier point.
    precision, recall = precision[:-1], recall[:-1]

    meets_target = recall >= minimum_recall
    if meets_target.any():
        # Parmi les seuils qui respectent le rappel minimum, celui qui
        # maximise la précision (donc minimise les faux positifs).
        candidate_idx = precision[meets_target].argmax()
        return float(thresholds[meets_target][candidate_idx])

    # Personne n'atteint minimum_recall : repli sur le meilleur F1,
    # comme l'ancien code repliait sur le meilleur rappel si besoin.
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
    Entraîne la régression logistique avec validation croisée GROUPÉE PAR
    ŒUVRE (StratifiedGroupKFold, groupe = artwork_id) : sans ce
    groupement, des candidats de la MÊME œuvre pourraient se retrouver à
    la fois en train et en test, ce qui fuite de l'information d'une
    manière que la validation croisée simple de l'ancien code
    (_stratified_folds, qui stratifiait au niveau œuvre mais n'existait
    que pour un seul label par œuvre) ne pouvait pas produire — ici
    plusieurs lignes partagent le même artwork_id, donc le risque est réel
    et StratifiedGroupKFold le prévient explicitement.

    `regularization_values` est la SEULE dimension de recherche restante
    (le paramètre C de la régression logistique) — une liste de quelques
    valeurs suffit, plus besoin de grille combinatoire.

    Retourne un dict avec le modèle final, le seuil choisi, les
    coefficients appris (pour comparaison avec les anciens poids à la
    main) et les métriques de validation croisée hors-échantillon.
    """
    X = table[FEATURE_COLUMNS].values
    y = table["label"].values
    groups = table["artwork_id"].astype(str).values

    n_groups = len(set(groups))
    effective_splits = min(n_splits, n_groups)
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
        f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
        f1 = float(f1_scores.max())
        if f1 > best_f1:
            best_f1, best_C = f1, C
            best_oof_probabilities = oof_probabilities.values.copy()

    # Seuil choisi sur les probabilités HORS-ÉCHANTILLON du meilleur C,
    # pas sur un ré-ajustement complet : évite d'être optimiste sur le
    # seuil comme le serait un seuil calé sur les données d'entraînement.
    threshold = _best_threshold(y, best_oof_probabilities, minimum_recall=minimum_recall)

    oof_predictions = best_oof_probabilities >= threshold
    true_positives = int(((oof_predictions == 1) & (y == 1)).sum())
    false_positives = int(((oof_predictions == 1) & (y == 0)).sum())
    false_negatives = int(((oof_predictions == 0) & (y == 1)).sum())
    true_negatives = int(((oof_predictions == 0) & (y == 0)).sum())
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # Modèle final : ré-entraîné sur TOUTES les données avec le C retenu,
    # pour la production / les coefficients à rapporter. Les métriques
    # ci-dessus restent celles, plus honnêtes, de la validation croisée.
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
            "f1": f1,
        },
    }


def _threshold_for_precision(
    labels,
    probabilities,
    groups,
    target_precision,
    minimum_predictions,
    minimum_prediction_groups,
):
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
        recall = true_positives / int((labels == 1).sum())
        candidate = {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "support": prediction_count,
            "prediction_groups": accepted_groups,
            "true_positives": true_positives,
            "false_positives": prediction_count - true_positives,
        }
        if best is None or (
            candidate["recall"],
            candidate["precision"],
            candidate["threshold"],
        ) > (
            best["recall"],
            best["precision"],
            best["threshold"],
        ):
            best = candidate
    return best


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

    Thresholds are chosen only when out-of-fold decisions meet the requested
    empirical precision and support floors. Under-supported levels abstain.
    """
    if table.empty:
        raise ValueError("Geographic label table is empty.")
    target_precisions = target_precisions or GEOGRAPHIC_TARGET_PRECISION
    feature_columns = [
        column for column in GEOGRAPHIC_FEATURE_COLUMNS if column in table.columns
    ]
    if len(feature_columns) != len(GEOGRAPHIC_FEATURE_COLUMNS):
        missing = sorted(set(GEOGRAPHIC_FEATURE_COLUMNS) - set(feature_columns))
        raise ValueError(f"Missing geographic model features: {missing}")

    matrix = table[feature_columns].fillna(0).to_numpy(dtype=float)
    statuses = table["label_status"].to_numpy()
    label_levels = table["label_level"].to_numpy()
    groups = table["cv_group"].astype(str).to_numpy()
    specificity = {level: index for index, level in enumerate(GEOGRAPHIC_LEVELS)}
    per_level = {}

    for level in GEOGRAPHIC_LEVELS:
        positive_levels = {
            candidate_level
            for candidate_level in GEOGRAPHIC_LEVELS
            if specificity[candidate_level] <= specificity[level]
        }
        eligible = (statuses == "negative") | (statuses == "match")
        row_indices = np.flatnonzero(eligible)
        level_labels = np.asarray([
            int(
                statuses[index] == "match"
                and label_levels[index] in positive_levels
            )
            for index in row_indices
        ])
        level_groups = groups[row_indices]
        positive_group_count = len(set(level_groups[level_labels == 1]))
        negative_group_count = len(set(level_groups[level_labels == 0]))
        target_precision = float(target_precisions[level])

        if positive_group_count < minimum_positive_groups:
            per_level[level] = {
                "status": "insufficient_positive_groups",
                "target_precision": target_precision,
                "positive_groups": positive_group_count,
                "minimum_positive_groups": minimum_positive_groups,
                "model": None,
                "threshold": None,
            }
            continue
        if negative_group_count < 2:
            per_level[level] = {
                "status": "insufficient_negative_groups",
                "target_precision": target_precision,
                "negative_groups": negative_group_count,
                "model": None,
                "threshold": None,
            }
            continue

        n_level_splits = min(n_splits, positive_group_count, negative_group_count)
        if n_level_splits < 2:
            per_level[level] = {
                "status": "insufficient_cv_groups",
                "target_precision": target_precision,
                "model": None,
                "threshold": None,
            }
            continue

        X = matrix[row_indices]
        best = None
        cv = StratifiedGroupKFold(
            n_splits=n_level_splits,
            shuffle=True,
            random_state=random_state,
        )
        for C in regularization_values:
            oof = np.full(len(row_indices), np.nan)
            try:
                folds = cv.split(X, level_labels, level_groups)
                for train_idx, test_idx in folds:
                    classifier = LogisticRegression(
                        C=C,
                        class_weight="balanced",
                        max_iter=1000,
                    )
                    classifier.fit(X[train_idx], level_labels[train_idx])
                    oof[test_idx] = classifier.predict_proba(X[test_idx])[:, 1]
            except ValueError:
                continue
            if not np.isfinite(oof).all():
                continue
            selected = _threshold_for_precision(
                level_labels,
                oof,
                level_groups,
                target_precision,
                minimum_predictions,
                minimum_prediction_groups,
            )
            if selected is None:
                continue
            candidate = {**selected, "C": C, "oof_probabilities": oof}
            if best is None or (
                candidate["recall"],
                candidate["precision"],
                -candidate["false_positives"],
            ) > (
                best["recall"],
                best["precision"],
                -best["false_positives"],
            ):
                best = candidate

        if best is None:
            per_level[level] = {
                "status": "precision_target_unmet",
                "target_precision": target_precision,
                "positive_groups": positive_group_count,
                "negative_groups": negative_group_count,
                "model": None,
                "threshold": None,
            }
            continue

        final_classifier = LogisticRegression(
            C=best["C"],
            class_weight="balanced",
            max_iter=1000,
        )
        final_classifier.fit(X, level_labels)
        per_level[level] = {
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
            "oof_row_indices": row_indices.tolist(),
            "oof_probabilities": best["oof_probabilities"].tolist(),
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
        for row_index, probability in zip(
            level_model["oof_row_indices"],
            level_model["oof_probabilities"],
        ):
            oof_scores_by_row[row_index][level] = probability
    model_bundle["threshold_tuning_work_metrics"] = evaluate_hierarchical_work_predictions(
        table,
        oof_scores_by_row,
        per_level,
        specificity,
        records=table.attrs.get("artwork_statuses"),
    )
    model_bundle["minimum_work_recall_met"] = (
        model_bundle["threshold_tuning_work_metrics"]["recall"] >= minimum_recall
    )
    for level_model in per_level.values():
        level_model.pop("oof_row_indices", None)
        level_model.pop("oof_probabilities", None)
    return model_bundle


def _select_hierarchical_level(candidate_level, level_scores, levels, specificity):
    candidate_level_rank = specificity.get(candidate_level)
    if candidate_level_rank is None:
        return None
    decisions = []
    for level, probability in level_scores.items():
        level_model = levels[level]
        if level_model["status"] != "active":
            continue
        if candidate_level_rank > specificity[level]:
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
    """Compute correct-location precision/recall for one hierarchical choice per artwork."""
    rows_by_artwork = defaultdict(list)
    for row_index, row in table.iterrows():
        rows_by_artwork[row["artwork_id"]].append((row_index, row))

    if isinstance(records, dict):
        artwork_statuses = records
    elif records is None:
        artwork_statuses = {
            row["artwork_id"]: row.get("place_status")
            for _, row in table.iterrows()
        }
    else:
        artwork_statuses = {
            (record.get("artwork", record)).get("id"):
            record.get("annotations", {}).get("place_status")
            for record in records
        }

    true_positives = false_positives = true_negatives = false_negatives = 0
    mislocated_count = 0
    for artwork_id, actual_status in artwork_statuses.items():
        candidates = []
        for row_index, row in rows_by_artwork.get(artwork_id, []):
            row_scores = scores_by_row.get(row_index, {})
            selected_level = _select_hierarchical_level(
                row.get("candidate_level"),
                row_scores,
                levels,
                specificity,
            )
            if selected_level is not None:
                actual_level = row.get("label_level")
                actual_is_correct_at_level = (
                    row.get("label_status") == "match"
                    and actual_level in specificity
                    and specificity[actual_level] <= specificity[selected_level]
                )
                candidates.append((
                    specificity[selected_level],
                    -row_scores[selected_level],
                    actual_is_correct_at_level,
                ))
        selected = min(candidates) if candidates else None
        actual_depicted = actual_status == "depicted"
        predicted = selected is not None
        correct_location = predicted and selected[2]
        if actual_depicted and correct_location:
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

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives else 0.0
    )
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "mislocated_count": mislocated_count,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "accuracy": (
            (true_positives + true_negatives) / len(artwork_statuses)
            if artwork_statuses else 0.0
        ),
        "artwork_count": len(artwork_statuses),
        "depicted_artwork_count": true_positives + false_negatives,
        "metrics_are_threshold_tuning_estimates": True,
    }


def evaluate_hierarchical_grouped_cv(
    records,
    table,
    *,
    n_splits=5,
    random_state=42,
    training_options=None,
):
    """Evaluate a full level/threshold selection process on outer grouped folds."""
    artwork_statuses = {
        (record.get("artwork", record)).get("id"):
        record.get("annotations", {}).get("place_status")
        for record in records
    }
    artwork_groups = table.attrs.get("artwork_cv_groups") or _artwork_target_cv_groups(records)
    artwork_ids = list(artwork_statuses)
    group_values = [str(artwork_groups.get(artwork_id, f"artwork:{artwork_id}")) for artwork_id in artwork_ids]
    unique_groups = len(set(group_values))
    effective_splits = min(n_splits, unique_groups)
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

        scores_by_row = defaultdict(dict)
        if not test_table.empty:
            feature_columns = model.get("feature_columns", GEOGRAPHIC_FEATURE_COLUMNS)
            matrix = test_table[feature_columns].fillna(0).to_numpy(dtype=float)
            for row_position, (_, row) in enumerate(test_table.iterrows()):
                candidate_level = row.get("candidate_level")
                candidate_level_rank = GEOGRAPHIC_LEVELS.index(candidate_level) if candidate_level in GEOGRAPHIC_LEVELS else None
                if candidate_level_rank is None:
                    continue
                for level, level_model in model["levels"].items():
                    if level_model["status"] != "active":
                        continue
                    if candidate_level_rank > GEOGRAPHIC_LEVELS.index(level):
                        continue
                    scores_by_row[row_position][level] = float(
                        level_model["model"].predict_proba(matrix[row_position:row_position + 1])[0][1]
                    )

        metrics = evaluate_hierarchical_work_predictions(
            test_table,
            scores_by_row,
            model["levels"],
            {level: index for index, level in enumerate(GEOGRAPHIC_LEVELS)},
            records={artwork_id: artwork_statuses[artwork_id] for artwork_id in test_ids},
        )
        for row_position, (_, row) in enumerate(test_table.iterrows()):
            if row.get("label_status") == "unevaluable":
                continue
            row_scores = scores_by_row.get(row_position, {})
            for level, probability in row_scores.items():
                level_model = model["levels"][level]
                cumulative_positive_levels = {
                    candidate_level
                    for candidate_level in GEOGRAPHIC_LEVELS
                    if GEOGRAPHIC_LEVELS.index(candidate_level) <= GEOGRAPHIC_LEVELS.index(level)
                }
                actual_positive = (
                    row.get("label_status") == "match"
                    and row.get("label_level") in cumulative_positive_levels
                )
                predicted_positive = probability >= level_model["threshold"]
                level_totals[level]["evaluated"] += 1
                if actual_positive:
                    level_totals[level]["actual_positives"] += 1
                if predicted_positive:
                    level_totals[level]["predictions"] += 1
                    level_prediction_groups[level].add(str(row.get("cv_group")))
                    if actual_positive:
                        level_totals[level]["true_positives"] += 1
                    else:
                        level_totals[level]["false_positives"] += 1
                elif actual_positive:
                    level_totals[level]["false_negatives"] += 1
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
                level for level in GEOGRAPHIC_LEVELS
                if model["levels"][level]["status"] == "active"
            ],
            "metrics": metrics,
        })

    tp = totals["true_positives"]
    fp = totals["false_positives"]
    fn = totals["false_negatives"]
    tn = totals["true_negatives"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
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
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "accuracy": (tp + tn) / totals["artwork_count"] if totals["artwork_count"] else 0.0,
        "artwork_count": totals["artwork_count"],
        "depicted_artwork_count": totals["depicted_artwork_count"],
        "metrics_by_level": per_level,
        "grouping": "connected_artwork_and_target_entities",
        "thresholds_tuned_inside_outer_training_folds": True,
        "folds": folds,
    }


def predict_with_hierarchical_model(
    records,
    candidates_by_artwork,
    model,
    *,
    gazetteer_conn=None,
    nominatim_client=None,
    geocode_top_k=3,
):
    """Predict the most specific sufficiently precise geographic level."""
    if geocode_top_k < 1:
        raise ValueError("geocode_top_k must be at least 1.")

    feature_columns = model["feature_columns"]
    levels = model["levels"]
    specificity = {
        level: index for index, level in enumerate(model["specificity_order"])
    }
    location_cache = {}
    results = {}

    for record in records:
        artwork = record.get("artwork", record)
        artwork_id = artwork.get("id")
        grouped_candidates = list(
            group_candidates_by_place(
                candidates_by_artwork.get(artwork_id, [])
            ).values()
        )
        grouped_candidates.sort(
            key=lambda occurrences: max(
                occurrence.get("score_ner", 0.0) for occurrence in occurrences
            ),
            reverse=True,
        )

        scored_candidates = []
        for candidate_index, occurrences in enumerate(grouped_candidates):
            place_name = occurrences[0]["place"]
            location = None
            if candidate_index < geocode_top_k:
                cache_key = place_name.casefold().strip()
                if cache_key not in location_cache:
                    if gazetteer_conn is not None or nominatim_client is not None:
                        location_cache[cache_key] = resolve_location(
                            place_name,
                            gazetteer_conn,
                            nominatim_client=nominatim_client,
                            include_hierarchy=True,
                        )
                    else:
                        location_cache[cache_key] = None
                location = location_cache[cache_key]

            features = _add_geographic_features(
                candidate_features(occurrences),
                location,
            )
            features.update(artwork_context_features(artwork, place_name))
            feature_row = np.asarray(
                [[features[column] for column in feature_columns]],
                dtype=float,
            )
            candidate_level = _candidate_geographic_level(location)
            candidate_level_rank = specificity.get(candidate_level)
            level_scores = {}
            decisions = []

            if candidate_level_rank is not None:
                for level in model["specificity_order"]:
                    level_model = levels[level]
                    if level_model["status"] != "active":
                        continue
                    deployable_levels = model.get("deployable_levels")
                    if deployable_levels is not None and level not in deployable_levels:
                        continue
                    if candidate_level_rank > specificity[level]:
                        continue
                    probability = float(
                        level_model["model"].predict_proba(feature_row)[0][1]
                    )
                    level_scores[level] = probability
                    if probability >= level_model["threshold"]:
                        decisions.append({
                            "level": level,
                            "model_score": probability,
                            "confidence": level_model.get(
                                "outer_cv_precision",
                                level_model.get("oof_precision"),
                            ),
                            "confidence_basis": "outer_grouped_cv_empirical_precision",
                            "confidence_support": level_model.get(
                                "outer_cv_prediction_groups",
                                level_model.get("oof_prediction_groups"),
                            ),
                            "target_precision": level_model["target_precision"],
                            "threshold": level_model["threshold"],
                        })

            decisions.sort(
                key=lambda decision: (
                    specificity[decision["level"]],
                    -decision["model_score"],
                )
            )
            scored_candidates.append({
                "place": place_name,
                "score_ner_max": max(
                    occurrence.get("score_ner", 0.0) for occurrence in occurrences
                ),
                "geolocation": location,
                "candidate_level": candidate_level,
                "level_scores": level_scores,
                "decision": decisions[0] if decisions else None,
                "champs": sorted({occurrence["field"] for occurrence in occurrences}),
            })

        accepted = [
            candidate for candidate in scored_candidates if candidate["decision"]
        ]
        accepted.sort(
            key=lambda candidate: (
                specificity[candidate["decision"]["level"]],
                -candidate["decision"]["model_score"],
                -candidate["score_ner_max"],
            )
        )
        results[artwork_id] = {
            "place_status_predicted": bool(accepted),
            "prediction": accepted[0] if accepted else None,
            "candidates": scored_candidates,
        }

    return results


def predict_with_model(records, candidates_by_artwork, model):
    """
    Applique le modèle entraîné à toutes les œuvres et renvoie, pour
    chaque artwork_id, la liste des lieux candidats triés par probabilité
    décroissante — même structure de sortie que l'ancien
    merge_candidate_places (clé "score_heuristique"), pour rester
    compatible avec evaluate_binary_predictions sans modification.
    """
    clf = model["model"]
    feature_columns = model["feature_columns"]
    results = {}

    for record in records:
        artwork = record.get("artwork", record)
        artwork_id = artwork.get("id")
        groups = group_candidates_by_place(candidates_by_artwork.get(artwork_id, []))

        places = []
        for occurrences in groups.values():
            row = candidate_features(occurrences)
            X = [[row[col] for col in feature_columns]]
            probability = float(clf.predict_proba(X)[0, 1])
            places.append({
                "place": occurrences[0]["place"],
                "score_heuristique": probability,
                "champs": sorted({c["field"] for c in occurrences}),
            })

        results[artwork_id] = sorted(places, key=lambda p: p["score_heuristique"], reverse=True)

    return results


def evaluate_binary_predictions(records, heuristic_results, threshold=0.5, review_margin=0.1):
    """Evaluate whether an artwork depicts a place, regardless of place identity.

    Inchangé par rapport à la version précédente : cette fonction ne
    regarde que "au moins un candidat accepté", pas l'identité du lieu.
    Voir evaluate_place_identity ci-dessous pour cette vérification.
    """
    if review_margin < 0:
        raise ValueError("review_margin must be non-negative.")

    true_positives = false_positives = true_negatives = false_negatives = 0
    false_positive_ids = []
    false_negative_ids = []
    flagged_predictions = []
    predictions = []

    for record in records:
        artwork = record.get("artwork", record)
        artwork_id = artwork.get("id")
        actual = record.get("annotations", {}).get("place_status") == "depicted"
        artwork_places = heuristic_results.get(artwork_id, [])
        accepted_places = [
            place for place in artwork_places
            if place.get("score_heuristique", 0) >= threshold
        ]
        predicted = bool(accepted_places)
        rejected_places = [
            {
                "place": place.get("place"),
                "score_heuristique": place.get("score_heuristique", 0),
            }
            for place in artwork_places
            if threshold - review_margin
            <= place.get("score_heuristique", 0)
            < threshold
        ]
        if not predicted and rejected_places:
            flagged_predictions.append({
                "id": artwork_id,
                "rejected_places": rejected_places,
            })

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
            predictions.append({
                "id": artwork_id,
                "actual": actual,
                "predicted": predicted,
                "place_status": record.get("annotations", {}).get("place_status"),
                "predicted_places": [
                    {
                        "place": place.get("place"),
                        "score_heuristique": place.get("score_heuristique", 0),
                    }
                    for place in accepted_places
                ],
            })

    total = true_positives + false_positives + true_negatives + false_negatives
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

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
        "accuracy": (true_positives + true_negatives) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "errors": predictions,
    }


def _valid_coordinates(place):
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
    Évalue le top-1 par nom et, si possible, par proximité géographique.

    La proximité du point annoté utilise coordinate_tolerance_km. La
    métrique ville compare le candidat au centre GeoNames de la ville
    annotée dans city_tolerance_km; c'est une approximation, pas un test
    d'appartenance à une limite administrative.
    """
    total_depicted = 0
    correct_identity = 0
    wrong_identity = []  # (artwork_id, lieu_predit, lieu_reel)
    no_candidate = []
    coordinate_matches = 0
    coordinate_evaluable = 0
    coordinate_resolved_candidates = 0
    coordinate_errors = []
    city_matches = 0
    city_evaluable = 0
    city_resolved_candidates = 0
    city_errors = []

    if coordinate_tolerance_km < 0 or city_tolerance_km < 0:
        raise ValueError("Coordinate tolerances must be non-negative.")

    for record in records:
        annotations = record.get("annotations", {})
        if annotations.get("place_status") != "depicted":
            continue
        artwork = record.get("artwork", record)
        artwork_id = artwork.get("id")
        total_depicted += 1

        places = heuristic_results.get(artwork_id, [])
        if not places:
            no_candidate.append(artwork_id)
        real_place = (annotations.get("depicted_place") or {}).get("name")
        depicted_place = annotations.get("depicted_place") or {}
        top_candidate = places[0] if places else None
        top_place = top_candidate.get("place") if top_candidate else None

        identity_matches = (
            top_place is not None
            and fuzzy_match(top_place, real_place, fuzzy_threshold)
        )
        if identity_matches:
            correct_identity += 1

        actual_coordinates = _valid_coordinates(depicted_place)
        predicted_coordinates = _valid_coordinates(top_candidate)
        if predicted_coordinates is None and top_place and (
            gazetteer_conn is not None or nominatim_client is not None
        ):
            resolved_candidate = resolve_location(
                top_place,
                gazetteer_conn,
                nominatim_client=nominatim_client,
            )
            if resolved_candidate:
                predicted_coordinates = _valid_coordinates(resolved_candidate)

        distance_to_actual_km = None
        if actual_coordinates is not None:
            coordinate_evaluable += 1
            if predicted_coordinates is not None:
                coordinate_resolved_candidates += 1
                distance_to_actual_km = haversine_distance_km(
                    *actual_coordinates,
                    *predicted_coordinates,
                )
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

        city_name = depicted_place.get("city")
        if (
            city_name
            and (gazetteer_conn is not None or nominatim_client is not None)
            and actual_coordinates is not None
        ):
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
                    distance = haversine_distance_km(
                        *city_coordinates,
                        *predicted_coordinates,
                    )
                    if distance <= city_tolerance_km:
                        city_matches += 1
                if distance is None or distance > city_tolerance_km:
                    city_errors.append({
                        "id": artwork_id,
                        "predicted": top_place,
                        "city": city_name,
                        "distance_km": distance,
                    })

    return {
        "total_depicted": total_depicted,
        "correct_identity": correct_identity,
        "wrong_identity_count": len(wrong_identity),
        "no_candidate_count": len(no_candidate),
        "identity_accuracy": correct_identity / total_depicted if total_depicted else 0.0,
        "wrong_identity": wrong_identity,
        "no_candidate_ids": no_candidate,
        "coordinate_matches": coordinate_matches,
        "coordinate_evaluable": coordinate_evaluable,
        "coordinate_resolved_candidates": coordinate_resolved_candidates,
        "coordinate_candidate_coverage": (
            coordinate_resolved_candidates / coordinate_evaluable
            if coordinate_evaluable else None
        ),
        "coordinate_tolerance_km": coordinate_tolerance_km,
        "coordinate_accuracy": (
            coordinate_matches / coordinate_evaluable if coordinate_evaluable else None
        ),
        "coordinate_accuracy_when_geocoded": (
            coordinate_matches / coordinate_resolved_candidates
            if coordinate_resolved_candidates else None
        ),
        "coordinate_errors": coordinate_errors,
        "city_matches": city_matches,
        "city_evaluable": city_evaluable,
        "city_resolved_candidates": city_resolved_candidates,
        "city_candidate_coverage": (
            city_resolved_candidates / city_evaluable if city_evaluable else None
        ),
        "city_tolerance_km": city_tolerance_km,
        "city_proximity_accuracy": (
            city_matches / city_evaluable if city_evaluable else None
        ),
        "city_proximity_accuracy_when_geocoded": (
            city_matches / city_resolved_candidates
            if city_resolved_candidates else None
        ),
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
