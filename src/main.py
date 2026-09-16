import api.aic as aic
import utils
from pathlib import Path

def test():
    print("Geolocating Artworks")
    
    queries = ["Paris", "Chicago"]

    results = aic.search_multiple_queries(queries, limit=5, max_pages=2)

    for artwork in results.values():
        print(artwork["artwork"]["id"], artwork["artwork"]["title"])

def main():
    
    combined_texts = utils.read_all_json_files_and_combine_text_fields("data/raw/aic")

    # print the combined texts with the format id : combined_text
    for artwork_id, combined_text in combined_texts.items():
        print(f"{artwork_id} : {combined_text[:100]}... \n")  # Print the first 100 characters of the combined text        

if __name__ == "__main__":
    main()
