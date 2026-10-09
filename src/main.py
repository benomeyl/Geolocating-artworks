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
from pathlib import Path
import requests
import pandas as pd


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
    
    # 1) Load the corpus
    artowrks = utils.read_artworks_from_json("data/raw/gold_standard/gold_standard.json")
    ner_model = ner.load_ner_model()

    for artwork in artowrks.values():
        results = filt.evaluate_artwork(artwork, ner_model)

        # Print results with a heuristic score > 0.55
        for result in results:
            if result['score_heuristique'] >= 0.4:
                print(f"Artwork ID: {artwork.get('id')}, Place: {result['place']}, "
                        f"Score NER Max: {result['score_ner_max']}, "
                        f"Heuristic Score: {result['score_heuristique']}, "
                        f"Fields: {result['champs']}")

                
def main():
    if ner is None:
        raise ModuleNotFoundError(
            "The NER module is unavailable. Install the project dependencies to use it."
        )

    # 1) Load the gold-standard records once
    with Path("data/raw/gold_standard/oeuvres_wikidata.json").open(
        "r", encoding="utf-8"
    ) as f:
        artwork_records = json.load(f)["unique_artworks"]

    ner_model = ner.load_ner_model()

    # 2) Evaluate each artwork and retain all heuristic results
    heuristic_results = {}
    for record in artwork_records:
        artwork = record["artwork"]
        heuristic_results[artwork["id"]] = filt.evaluate_artwork(artwork, ner_model)

    # 3) Compare heuristic acceptance with annotations
    comparison = filt.compare_heuristic_places_with_annotations(
        artwork_records,
        heuristic_results,
        threshold=0.4,
    )

    # 4) Display a readable summary
    print("─" * 60)
    print("Comparaison des lieux : heuristique vs. annotations")
    print("─" * 60)
    print(f"Œuvres avec un lieu réellement représenté : {comparison['real_depicted_artworks']}")
    print(f"Œuvres acceptées par l’heuristique : {comparison['heuristic_accepted_artworks']}")
    print(f"Lieux réellement représentés : {comparison['real_depicted_places']}")
    print(f"Lieux acceptés par l’heuristique : {comparison['heuristic_accepted_places']}")
    print(f"Œuvres en divergence : {len(comparison['artworks_with_discrepancy'])}")

    for discrepancy in comparison["artworks_with_discrepancy"]:
        artwork_id = discrepancy["id"]
        print(
            f"- Artwork {artwork_id}: réel={discrepancy['real_depicted']}, "
            f"accepté={discrepancy['accepted']}"
        )


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
    main()