import re
import pandas as pd
from collections import defaultdict
from itertools import product
import random

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
    "title": 0.5,
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

def score_candidate(
    candidate,
    field_weights=None,
    positive_context_bonus=0.3,
    negative_context_penalty=0.4,
):
    weights = FIELDS_WEIGHTS if field_weights is None else field_weights
    score = weights.get(candidate["field"], 0.0)

    context = local_context(candidate["source_text"], candidate["start"], candidate["end"])
    if any(marker in context for marker in POSITIVES_LABELS):
        score += positive_context_bonus
    if any(marker in context for marker in NEGATIVES_LABELS):
        score -= negative_context_penalty

    return score

def merge_candidate_places(
    candidates,
    field_weights=None,
    positive_context_bonus=0.3,
    negative_context_penalty=0.4,
    multi_field_bonus=0.15,
):
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
        heuristic_score = max(
            score_candidate(
                candidate,
                field_weights=field_weights,
                positive_context_bonus=positive_context_bonus,
                negative_context_penalty=negative_context_penalty,
            )
            for candidate in occurrences
        )
        found_fields = {candidate["field"] for candidate in occurrences}

        # Add a bonus when the place appears in multiple independent fields.
        if len(found_fields) > 1:
            heuristic_score += multi_field_bonus

        merged_places.append({
            "place": occurrences[0]["place"],  # Preserve the original spelling and capitalization.
            "score_ner_max": max(candidate["score_ner"] for candidate in occurrences),
            "score_heuristique": round(heuristic_score, 3),
            "champs": sorted(found_fields),
            "occurrences": occurrences,  # Kept for debugging and traceability.
        })

    return sorted(merged_places, key=lambda place: place["score_heuristique"], reverse=True)


def evaluate_artwork(
    artwork,
    ner,
    threshold=0.85,
    *,
    field_weights=None,
    positive_context_bonus=0.3,
    negative_context_penalty=0.4,
    multi_field_bonus=0.15,
):
    candidates = extract_candidate_artwork(artwork, ner, threshold=threshold)
    merged_places = merge_candidate_places(
        candidates,
        field_weights=field_weights,
        positive_context_bonus=positive_context_bonus,
        negative_context_penalty=negative_context_penalty,
        multi_field_bonus=multi_field_bonus,
    )
    return merged_places


def evaluate_binary_predictions(
    records,
    heuristic_results,
    threshold=0.5,
    review_margin=0.1,
):
    """Evaluate whether an artwork depicts a place, regardless of place identity."""
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


def _stratified_folds(records, n_splits, random_state):
    positive_indices = []
    negative_indices = []
    for index, record in enumerate(records):
        if record.get("annotations", {}).get("place_status") == "depicted":
            positive_indices.append(index)
        else:
            negative_indices.append(index)

    if n_splits < 2 or n_splits > min(len(positive_indices), len(negative_indices)):
        raise ValueError(
            "n_splits must be at least 2 and no greater than the size of the "
            "smallest class."
        )

    rng = random.Random(random_state)
    rng.shuffle(positive_indices)
    rng.shuffle(negative_indices)
    folds = [[] for _ in range(n_splits)]
    for label_indices in (positive_indices, negative_indices):
        for offset, index in enumerate(label_indices):
            folds[offset % n_splits].append(index)

    all_indices = set(range(len(records)))
    return [
        (sorted(all_indices - set(test_indices)), sorted(test_indices))
        for test_indices in folds
    ]


def _heuristic_parameter_grid():
    field_weight_values = {
        "title": [0.2, 0.4, 0.6, 0.7, 0.8],
        "themes": [0.1, 0.2, 0.3, 0.4],
        "subjects": [0.1, 0.2, 0.3, 0.4],
        "description": [0.0, 0.1, 0.2],
        "short_description": [0.0, 0.1, 0.2],
        "styles": [-0.2, -0.1, 0.0],
    }
    field_names = list(FIELDS_WEIGHTS)
    field_weights = [
        dict(zip(field_names, values))
        for values in product(*(field_weight_values[field] for field in field_names))
    ]
    return {
        "field_weights": field_weights,
        "positive_context_bonus": [0.0, 0.1, 0.2, 0.3, 0.5],
        "negative_context_penalty": [0.0, 0.2, 0.4, 0.6],
        "multi_field_bonus": [0.0, 0.15],
        "acceptance_threshold": [0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
        "ner_threshold": [0.2, 0.4, 0.5, 0.65, 0.8, 0.9],
    }


def _predict_with_parameters(records, candidates_by_artwork, parameters):
    heuristic_results = {}
    for record in records:
        artwork = record.get("artwork", record)
        artwork_id = artwork.get("id")
        candidates = [
            candidate
            for candidate in candidates_by_artwork.get(artwork_id, [])
            if candidate["score_ner"] >= parameters["ner_threshold"]
        ]
        heuristic_results[artwork_id] = merge_candidate_places(
            candidates,
            field_weights=parameters["field_weights"],
            positive_context_bonus=parameters["positive_context_bonus"],
            negative_context_penalty=parameters["negative_context_penalty"],
            multi_field_bonus=parameters["multi_field_bonus"],
        )
    return heuristic_results


def optimize_heuristic_parameters(
    records,
    ner=None,
    *,
    candidates_by_artwork=None,
    parameter_grid=None,
    n_splits=5,
    inner_splits=3,
    random_state=42,
    minimum_recall=0.8,
    review_margin=0.1,
):
    """Tune parameters to minimize false positives while preserving recall.

    NER predictions are extracted once per artwork, then reused for every grid
    configuration. `candidates_by_artwork` can be passed to reuse an existing
    cache; its candidate lists must be extracted with NER threshold 0.0. If no
    configuration reaches `minimum_recall`, the one with the highest recall is
    selected, using false positives as the next criterion.
    """
    if not 0 <= minimum_recall <= 1:
        raise ValueError("minimum_recall must be between 0 and 1.")
    if review_margin < 0:
        raise ValueError("review_margin must be non-negative.")

    if candidates_by_artwork is None:
        if ner is None:
            raise ValueError("Pass either an NER pipeline or cached candidate lists.")
        candidates_by_artwork = {}
        for record in records:
            artwork = record.get("artwork", record)
            candidates_by_artwork[artwork.get("id")] = extract_candidate_artwork(
                artwork, ner, threshold=0.0
            )

    search_space = _heuristic_parameter_grid()
    if parameter_grid is not None:
        unknown_parameters = set(parameter_grid) - set(search_space)
        if unknown_parameters:
            raise ValueError(f"Unknown parameter(s): {sorted(unknown_parameters)}")
        search_space.update(parameter_grid)
    if any(not values for values in search_space.values()):
        raise ValueError("Every parameter grid must contain at least one value.")

    parameter_names = list(search_space)
    configuration_count = 1
    for values in search_space.values():
        configuration_count *= len(values)

    if configuration_count <= 5000:
        configurations = [
            dict(zip(parameter_names, values))
            for values in product(*(search_space[name] for name in parameter_names))
        ]
    else:
        rng = random.Random(random_state)
        sampled_indices = sorted(rng.sample(range(configuration_count), 5000))
        configurations = []
        for flat_index in sampled_indices:
            values = []
            for parameter_name in reversed(parameter_names):
                choices = search_space[parameter_name]
                flat_index, choice_index = divmod(flat_index, len(choices))
                values.append(choices[choice_index])
            configurations.append(dict(zip(parameter_names, reversed(values))))

    folds = _stratified_folds(records, n_splits, random_state)

    def best_configuration(training_records, seed):
        class_counts = [
            sum(
                record.get("annotations", {}).get("place_status") == "depicted"
                for record in training_records
            ),
            sum(
                record.get("annotations", {}).get("place_status") != "depicted"
                for record in training_records
            ),
        ]
        selection_splits = min(inner_splits, min(class_counts))
        if selection_splits < 2:
            raise ValueError(
                "Each outer training fold needs at least two examples per class "
                "for inner cross-validation."
            )
        selection_folds = _stratified_folds(
            training_records,
            selection_splits,
            seed,
        )
        best_parameters = None
        best_rank = None
        for parameters in configurations:
            validation_metrics = []
            for inner_training_indices, validation_indices in selection_folds:
                validation_records = [
                    training_records[index] for index in validation_indices
                ]
                validation_results = _predict_with_parameters(
                    validation_records,
                    candidates_by_artwork,
                    parameters,
                )
                validation_metrics.append(
                    evaluate_binary_predictions(
                        validation_records,
                        validation_results,
                        threshold=parameters["acceptance_threshold"],
                        review_margin=review_margin,
                    )
                )
            true_positives = sum(metric["true_positives"] for metric in validation_metrics)
            false_positives = sum(metric["false_positives"] for metric in validation_metrics)
            false_negatives = sum(metric["false_negatives"] for metric in validation_metrics)
            recall = (
                true_positives / (true_positives + false_negatives)
                if true_positives + false_negatives else 0.0
            )
            precision = (
                true_positives / (true_positives + false_positives)
                if true_positives + false_positives else 0.0
            )
            meets_recall_target = recall >= minimum_recall
            rank = (
                int(meets_recall_target),
                -false_positives if meets_recall_target else recall,
                precision if meets_recall_target else -false_positives,
                recall if meets_recall_target else precision,
                parameters["acceptance_threshold"],
            )
            if best_rank is None or rank > best_rank:
                best_parameters = parameters
                best_rank = rank
        return best_parameters

    fold_metrics = []
    fold_parameters = []
    for training_indices, test_indices in folds:
        training_records = [records[index] for index in training_indices]
        test_records = [records[index] for index in test_indices]
        parameters = best_configuration(training_records, random_state + len(fold_metrics))
        test_results = _predict_with_parameters(
            test_records, candidates_by_artwork, parameters
        )
        fold_metrics.append(
            evaluate_binary_predictions(
                test_records,
                test_results,
                threshold=parameters["acceptance_threshold"],
                review_margin=review_margin,
            )
        )
        fold_parameters.append(parameters)

    cv_counts = {
        metric: sum(fold[metric] for fold in fold_metrics)
        for metric in (
            "true_positives",
            "false_positives",
            "true_negatives",
            "false_negatives",
        )
    }
    cv_precision_denominator = cv_counts["true_positives"] + cv_counts["false_positives"]
    cv_recall_denominator = cv_counts["true_positives"] + cv_counts["false_negatives"]
    cv_precision = (
        cv_counts["true_positives"] / cv_precision_denominator
        if cv_precision_denominator else 0.0
    )
    cv_recall = (
        cv_counts["true_positives"] / cv_recall_denominator
        if cv_recall_denominator else 0.0
    )
    cv_f1 = (
        2 * cv_precision * cv_recall / (cv_precision + cv_recall)
        if cv_precision + cv_recall else 0.0
    )
    cv_total = sum(cv_counts.values())
    cv_metrics = {
        **cv_counts,
        "total": cv_total,
        "positive_count": cv_recall_denominator,
        "negative_count": cv_counts["true_negatives"] + cv_counts["false_positives"],
        "accuracy": (cv_counts["true_positives"] + cv_counts["true_negatives"]) / cv_total,
        "precision": cv_precision,
        "recall": cv_recall,
        "f1": cv_f1,
        "false_positive_ids": [
            artwork_id for fold in fold_metrics for artwork_id in fold["false_positive_ids"]
        ],
        "false_negative_ids": [
            artwork_id for fold in fold_metrics for artwork_id in fold["false_negative_ids"]
        ],
        "errors": [error for fold in fold_metrics for error in fold["errors"]],
        "flagged_ids": [artwork_id for fold in fold_metrics for artwork_id in fold["flagged_ids"]],
        "flagged_predictions": [
            prediction for fold in fold_metrics for prediction in fold["flagged_predictions"]
        ],
    }

    final_parameters = best_configuration(records, random_state + n_splits)
    final_results = _predict_with_parameters(records, candidates_by_artwork, final_parameters)
    final_metrics = evaluate_binary_predictions(
        records,
        final_results,
        threshold=final_parameters["acceptance_threshold"],
        review_margin=review_margin,
    )
    return {
        "best_parameters": final_parameters,
        "best_metrics": final_metrics,
        "cv_metrics": cv_metrics,
        "fold_metrics": fold_metrics,
        "fold_parameters": fold_parameters,
        "false_positive_ids": final_metrics["false_positive_ids"],
        "false_negative_ids": final_metrics["false_negative_ids"],
        "flagged_ids": final_metrics["flagged_ids"],
        "selection_objective": {
            "name": "minimize_false_positives",
            "minimum_recall": minimum_recall,
            "review_margin": review_margin,
        },
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