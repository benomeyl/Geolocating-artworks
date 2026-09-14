import api.aic as aic

def main():
    print("Geolocating Artworks")

    results = aic.search_all_artworks("Paris", max_pages=10, limit=5)

    for artwork in results["data"]:
        print(artwork["id"], artwork["title"])


if __name__ == "__main__":
    main()
