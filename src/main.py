import api.aic as aic

def main():
    print("Geolocating Artworks")

    queries = ["Paris", "Chicago"]

    results = aic.search_multiple_queries(queries, limit=5, max_pages=2)

    for artwork in results.values():
        print(artwork["artwork"]["id"], artwork["artwork"]["title"])


if __name__ == "__main__":
    main()
