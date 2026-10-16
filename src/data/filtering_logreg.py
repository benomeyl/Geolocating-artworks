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
from collections import defaultdict

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedGroupKFold
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
FIELD_NAMES = ["title", "themes", "subjects", "description", "short_description", "styles"]

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


def extract_candidate_artwork(artwork, ner, threshold=DEFAULT_NER_FLOOR):
    """
    Extract detected LOC candidates, including their source field and position
    within that field (required for context scoring).

    `threshold` reste volontairement bas par défaut (DEFAULT_NER_FLOOR) :
    c'est un filtre anti-bruit, pas le seuil d'acceptation final. Le score
    NER complet est conservé comme feature (score_ner_max) pour le modèle.
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
    depicted = annotations.get("depicted_place") or {}
    return int(fuzzy_match(place_name, depicted.get("name"), fuzzy_threshold))


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
