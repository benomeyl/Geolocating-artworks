"""
Early entry point of the project: keyword filter, NER exploration and the
legacy name-based model.

The current pipeline is driven by run_pipeline.py (hierarchical model). The
`test_*` functions are manual exploration helpers, not pytest tests.
"""

if __package__ in (None, ""):
    import api.aic as aic
    import utils
    import data.filtering_logreg as filt
    try:
        import data.ner as ner
    except ModuleNotFoundError:
        ner = None
else:
    from src.api import aic
    from src import utils
    from src.data import filtering_logreg as filt
    try:
        from src.data import ner
    except ModuleNotFoundError:
        ner = None

import argparse
import json
from pathlib import Path

import pandas as pd
import requests


def _require_ner():
    """Raise a clear error when the NER dependencies are not installed."""
    if ner is None:
        raise ModuleNotFoundError(
            "The NER module is unavailable. Install the project dependencies to use it."
        )


def _load_keywords(path="config/keywords.json"):
    """Load the keyword structure used by the keyword filter."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def test_request():
    """Fetch the first 10 artworks of the AIC API (connectivity check)."""
    BASE_URL = "https://api.artic.edu/api/v1"
    url = f"{BASE_URL}/artworks"
    params = {"limit": 10}

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def test_filtering(combined_texts):
    """Run the keyword filter on {id: text} and print what it keeps."""
    filtered_artworks = filt.filter_by_keywords(combined_texts, _load_keywords())

    for artwork in filtered_artworks["filtered_artworks"]:
        print(f"Filtered Artwork ID: {artwork['id']}, Matched Keywords: {artwork['matched_keywords']}")

    total_artworks = len(combined_texts)
    filtered_count = len(filtered_artworks["filtered_artworks"])

    print(f"Total artworks: {total_artworks}")
    print(f"Filtered artworks: {filtered_count}")
    if total_artworks:
        print(f"Filtering rate: {filtered_count / total_artworks * 100:.2f}%")

    return filtered_artworks

def test_ner():
    """Keyword-filter the raw AIC corpus, run the NER and print LOC entities (score >= 0.85)."""
    _require_ner()

    # 1) Load the corpus
    corpus = utils.read_all_json_files_and_combine_text_fields("data/raw/aic")

    if isinstance(corpus, pd.DataFrame):
        texts_by_id = corpus.set_index("id")["text"].fillna("").to_dict()
    elif isinstance(corpus, dict):
        texts_by_id = {
            artwork_id: (text if isinstance(text, str) else "" if text is None else str(text))
            for artwork_id, text in corpus.items()
        }
    else:
        raise TypeError(
            "The corpus must be a DataFrame or an id -> combined text dictionary. "
            f"Received type: {type(corpus).__name__}"
        )

    # 2) Filter the artworks with the keywords
    filtered_artworks = filt.filter_by_keywords(texts_by_id, _load_keywords())

    filtered_ids = [artwork["id"] for artwork in filtered_artworks["filtered_artworks"]]
    filtered_texts = {
        artwork_id: texts_by_id[artwork_id]
        for artwork_id in filtered_ids
        if artwork_id in texts_by_id
    }

    if not filtered_texts:
        print("No artworks were selected by the keyword filters. Stopping before NER.")
        return

    # 3) Apply NER to the filtered combined texts
    ner_model = ner.load_ner_model()
    results = ner.extract_named_entities(filtered_texts, ner_model)

    # Keep only "LOC" entities with a score >= 0.85
    for artwork_id, entities in results.items():
        results[artwork_id] = [
            entity for entity in entities
            if entity["entity_group"] == "LOC" and entity["score"] >= 0.85
        ]

    # 4) Display the detected entities
    ner.display_named_entities(results)


def test_heuristic():
    """Train and evaluate the legacy model on the gold standard."""
    _require_ner()
    train_heuristic()

def train_heuristic(ner_model=None):
    """Train the legacy name-based model on the gold standard and print its CV metrics.

    Returns (model, ner_model) so that the NER can be reused for prediction.
    """
    _require_ner()

    # 1) Load the gold-standard records once
    with Path("data/raw/gold_standard/gold_dataset.json").open(
        "r", encoding="utf-8"
    ) as f:
        artwork_records = json.load(f)["unique_artworks"]

    if ner_model is None:
        ner_model = ner.load_ner_model()

    candidates_by_artwork = filt.extract_all_candidates(artwork_records, ner_model)
    table = filt.build_training_table(artwork_records, candidates_by_artwork)
    optimization = filt.train_logistic_model(table)
    metrics = optimization["cv_metrics"]

    print("─" * 60)
    print("Évaluation croisée de la classification par candidat")
    print("─" * 60)
    print(f"Précision : {metrics['precision']:.3f}")
    print(f"Rappel : {metrics['recall']:.3f}")
    print(f"F1 : {metrics['f1']:.3f}")
    print(
        "Matrice de confusion (TP/FP/TN/FN) : "
        f"{metrics['true_positives']}/{metrics['false_positives']}/"
        f"{metrics['true_negatives']}/{metrics['false_negatives']}"
    )
    print(f"C de régularisation : {optimization['C']}")
    print(f"Seuil d’acceptation : {optimization['threshold']:.3f}")

    return optimization, ner_model


def predict_artworks_file(input_path, output_path, model, ner_model, review_margin=0.1):
    """Apply the legacy model to a JSON file of artworks and write the predictions.

    Places within `review_margin` below the threshold are reported as
    `flagged_places` for manual review.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    with input_path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = next(
            (payload[key] for key in ("unique_artworks", "artworks", "data")
             if isinstance(payload.get(key), list)),
            [payload],
        )
    else:
        raise ValueError("The input JSON must contain an artwork or a list of artworks.")

    predictions = []
    candidates_by_artwork = filt.extract_all_candidates(records, ner_model)
    places_by_artwork = filt.predict_with_model(records, candidates_by_artwork, model)
    acceptance_threshold = model["threshold"]
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each artwork entry in the input JSON must be an object.")
        artwork = record.get("artwork", record)
        places = places_by_artwork.get(artwork.get("id"), [])
        accepted_places = [
            {
                "place": place["place"],
                "score": place["score_heuristique"],
            }
            for place in places
            if place["score_heuristique"] >= acceptance_threshold
        ]
        flagged_places = [
            {
                "place": place["place"],
                "score": place["score_heuristique"],
            }
            for place in places
            if acceptance_threshold - review_margin
            <= place["score_heuristique"]
            < acceptance_threshold
        ]
        predictions.append({
            "id": artwork.get("id"),
            "has_predicted_place": bool(accepted_places),
            "predicted_places": accepted_places,
            "flagged_places": flagged_places,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump({"predictions": predictions}, output_file, ensure_ascii=False, indent=2)
    return predictions

def main(predict_input=None, prediction_output="data/processed/heuristic_predictions.json"):
    """Train the legacy model, then optionally predict `predict_input`."""
    _require_ner()

    optimization, ner_model = train_heuristic()
    if predict_input is not None:
        predictions = predict_artworks_file(
            predict_input,
            prediction_output,
            optimization,
            ner_model,
        )
        print(f"Prédictions exportées : {len(predictions)} dans {prediction_output}")
    return optimization


def get_corpus(queries=None, max_pages=5, limit=10, output_dir="data/raw/aic"):
    """Fetch and save an AIC corpus (see aic.get_corpus, which this delegates to)."""
    return aic.get_corpus(queries=queries, max_pages=max_pages, limit=limit, output_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluer et appliquer l’heuristique de géolocalisation.")
    parser.add_argument(
        "--predict-input",
        type=Path,
        help="Fichier JSON contenant les œuvres à prédire.",
    )
    parser.add_argument(
        "--prediction-output",
        type=Path,
        default=Path("data/processed/heuristic_predictions.json"),
        help="Fichier JSON de sortie des prédictions.",
    )
    arguments = parser.parse_args()
    main(arguments.predict_input, arguments.prediction_output)