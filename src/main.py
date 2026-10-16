if __package__ in (None, ""):
    import api.aic as aic
    import utils
    import data.filtering as filt
    try:
        import data.ner as ner
    except ModuleNotFoundError:
        ner = None
    try:
        import geolocation.gazetter as gaz
    except ModuleNotFoundError:
        gaz = None
else:
    from src.api import aic
    from src import utils
    from src.data import filtering as filt
    try:
        from src.data import ner
    except ModuleNotFoundError:
        ner = None
    try:
        from src.geolocation import gazetter as gaz
    except ModuleNotFoundError:
        gaz = None

import json
import argparse
from pathlib import Path
import requests
import pandas as pd

try:
    from src.api import aic as api
except ImportError:  # pragma: no cover - fallback for script execution
    import api.aic as api


def test_request():
    BASE_URL = "https://api.artic.edu/api/v1"
    url = f"{BASE_URL}/artworks"
    params = {"limit": 10}

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def test_filtering(combined_texts):
    keywords_path = Path("config/keywords.json")
    with keywords_path.open("r", encoding="utf-8") as f:
        keywords = json.load(f)

    filtered_artworks = filt.filter_by_keywords(combined_texts, keywords)

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
    
    if ner is None:
        raise ModuleNotFoundError(
            "The NER module is unavailable. Install the project dependencies to use it."
        )

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

    # 3) Load the keywords
    keywords_path = Path("config/keywords.json")
    with keywords_path.open("r", encoding="utf-8") as f:
        keywords = json.load(f)

    # 4) Filter the artworks
    filtered_artworks = filt.filter_by_keywords(texts_by_id, keywords)

    filtered_ids = [artwork["id"] for artwork in filtered_artworks["filtered_artworks"]]
    filtered_texts = {
        artwork_id: texts_by_id[artwork_id]
        for artwork_id in filtered_ids
        if artwork_id in texts_by_id
    }

    if not filtered_texts:
        print("No artworks were selected by the keyword filters. Stopping before NER.")
        return

    # 5) Apply NER to the filtered combined texts
    ner_model = ner.load_ner_model()
    results = ner.extract_named_entities(filtered_texts, ner_model)

    # Keep only "LOC" entities with a score >= 0.85
    for artwork_id, entities in results.items():
        results[artwork_id] = [
            entity for entity in entities
            if entity["entity_group"] == "LOC" and entity["score"] >= 0.85
        ]

    # 6) Display the detected entities (including locations)
    ner.display_named_entities(results)

def test_heuristic():
    if ner is None:
            raise ModuleNotFoundError(
                "The NER module is unavailable. Install the project dependencies to use it."
            )
    
    train_heuristic()

def train_heuristic(ner_model=None):
    if ner is None:
        raise ModuleNotFoundError(
            "The NER module is unavailable. Install the project dependencies to use it."
        )

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
    if ner is None:
        raise ModuleNotFoundError(
            "The NER module is unavailable. Install the project dependencies to use it."
        )

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
    """
    Get the corpus of artworks based on the provided queries.
    If queries is None, it retrieves artworks in default order.
    If queries is provided, it retrieves artworks for the specified queries.
    """
    if queries is None:
        artworks = aic.search_all_artworks(max_pages=max_pages, limit=limit)
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
        unique_artworks = aic.search_multiple_queries(
            search_queries,
            max_pages=max_pages,
            limit=limit,
            output_dir=output_dir,
        )

    if queries is None:
        aic.save_artworks_response(
            {
                "search_queries": None,
                "unique_artworks": list(unique_artworks.values()),
            },
            search_queries=None,
            output_dir=output_dir,
        )

    return unique_artworks


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