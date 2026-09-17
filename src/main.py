import api.aic as aic
import utils
import data.filtering as filt
import json
from pathlib import Path
import requests

def test_request():
    BASE_URL = "https://api.artic.edu/api/v1"

    url = f"{BASE_URL}/artworks"

    params = {
        "limit": 10
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    return response.json()

def main():    
    get_corpus(queries=None, max_pages=10, limit=100, output_dir="data/raw/aic")
    
    combined_texts = utils.read_all_json_files_and_combine_text_fields("data/raw/aic")

    #read the keywords from the keywords.json file
    keywords_path = Path("config/keywords.json")
    with keywords_path.open("r", encoding="utf-8") as f:
        keywords = json.load(f)

    filtered_artworks = {}

    filtered_artworks = filt.filter_by_keywords(combined_texts, keywords)     
        

    # Print the filtered artworks
    for artwork in filtered_artworks["filtered_artworks"]:
        print(f"Filtered Artwork ID: {artwork['id']},  Matched Keywords: {artwork['matched_keywords']}")


    #print stats for the filtered artworks and compare to the total number of artworks
    total_artworks = len(combined_texts)
    filtered_count = len(filtered_artworks["filtered_artworks"])
    print(f"Total artworks: {total_artworks}")
    print(f"Filtered artworks: {filtered_count}")
    print(f"Filtering rate: {filtered_count / total_artworks * 100:.2f}%")
    

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
