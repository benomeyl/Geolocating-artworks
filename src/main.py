if __package__ in (None, ""):
    import api.aic as aic
    import utils
    import data.filtering as filt
    import data.ner as ner
else:
    from src.api import aic
    from src import utils
    from src.data import filtering as filt
    from src.data import ner

import json
from pathlib import Path
import requests
import pandas as pd


def test_request():
    BASE_URL = "https://api.artic.edu/api/v1"
    url = f"{BASE_URL}/artworks"
    params = {"limit": 10}

    response = requests.get(url, params=params)
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
    # 1) Charger le corpus
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
                "Le corpus doit être un DataFrame ou un dictionnaire id -> texte combiné. "
                f"Type reçu : {type(corpus).__name__}"
            )
    
        # 3) Charger les mots-clés
        keywords_path = Path("config/keywords.json")
        with keywords_path.open("r", encoding="utf-8") as f:
            keywords = json.load(f)
    
        # 4) Filtrer les artworks
        filtered_artworks = filt.filter_by_keywords(texts_by_id, keywords)
    
        filtered_ids = [artwork["id"] for artwork in filtered_artworks["filtered_artworks"]]
        filtered_texts = {
            artwork_id: texts_by_id[artwork_id]
            for artwork_id in filtered_ids
            if artwork_id in texts_by_id
        }
    
        if not filtered_texts:
            print("Aucun artwork n’a été retenu par les filtres de mots-clés. Arrêt avant le NER.")
            return
    
        # 5) Appliquer le NER sur les textes combinés filtrés
        ner_model = ner.load_ner_model()
        results = ner.extract_named_entities(filtered_texts, ner_model)
    
        # Ne garde que les entités de type "LOC" avec un score >= 0.85
        for artwork_id, entities in results.items():
            results[artwork_id] = [
                entity for entity in entities
                if entity["entity_group"] == "LOC" and entity["score"] >= 0.85
            ]
    
        # 6) Afficher les entités détectées (dont les lieux)
        ner.display_named_entities(results)

def main():
    print("Starting the main process...")


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