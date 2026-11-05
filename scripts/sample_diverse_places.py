"""Draw AIC artworks whose title names many different cities, for annotation.

Random samples of the AIC collection are dominated by Chicago and Paris. This
script reads the local AIC dump, keeps pictorial artworks (prints,
photographs, drawings, paintings) with an image, and finds in each title a
city of GeoNames (population >= 50,000) written as a place: alone, after a
comma or after "in/at/of/near/from..." and before the end of the title or a
punctuation mark ("Le Pont Neuf, Paris", "Bridge in St. Petersburg"). The
sample is then stratified: at most --per-city artworks per city and
--per-country per country, drawn at random.

The model plays no part in the selection, so the annotated artworks can be
used for evaluation (stratum "precise_diverse"), unlike the rows of
annotation_precise_keywords_500.csv annotated in p_exact order. The measured
precision describes titles naming a city across many countries, not the
AIC distribution.

    python scripts/sample_diverse_places.py --size 800
    python scripts/predict_sample_csv.py --records data/raw/aic/diverse_places_sample.json \\
        --size 800 --model results/model_hierarchical_v6.joblib --output results/test_diverse_v6.csv
    python scripts/export_precise_places.py --input-csv results/test_diverse_v6.csv \\
        --records data/raw/aic/diverse_places_sample.json --model results/model_hierarchical_v6.joblib \\
        --output results/annotation_precise_diverse.csv --order random --sample-stratum precise_diverse
"""

import argparse
import csv
import json
import random
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

DUMP_DIR = Path("data/raw/artic-api-data/json/artworks")
PICTORIAL_TYPES = {"Print", "Photograph", "Drawing and Watercolor", "Painting"}
MINIMUM_POPULATION = 50000
# City names that are mostly something else in AIC titles: first names,
# common words, Japanese actor names ("Ichikawa Danjūrō") or mountains.
NOT_CITIES = {
    "Temple", "Reading", "Male", "Spring", "Springs", "George", "Elizabeth", "Forest", "Mary", "Roman",
    "Virginia", "David", "Orange", "Jackson", "Wedding", "Jupiter", "Paradise", "Meridian", "Martin",
    "Victoria", "Marion", "Lincoln", "Columbia", "Florida", "Aurora", "Phoenix", "Eden", "Hope", "Union",
    "Independence", "Liberty", "Concord", "Providence", "Saint John", "Salem", "Bethlehem", "Carmel",
    "Nazareth", "Ichikawa", "Narita", "Bando", "Sumida", "Koto", "Matsumoto", "Fuji", "Mobile", "Battle",
    "Garden Grove", "Irving", "Gary", "Kent", "Leon", "Laval", "Loja", "Sale", "Split", "Most", "Gran",
    "Mons", "Ede", "Este", "Troy", "Peoria", "Sunrise", "Lorca",
}
# Short forms used in titles for cities whose GeoNames name is longer.
CITY_ALIASES = {"New York": "New York City"}
PLACE_PREFIX = r"(?:^|,\s*|\b(?:in|at|of|near|from|to|de|du|a|à)\s+)"
PLACE_SUFFIX = r"(?=\s*(?:$|[,;:()\[\]–—-]|\.$))"


def _without_accents(text):
    """Text without diacritics ("Nikkō" -> "Nikko")."""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def load_cities(gazetteer_path, minimum_population=MINIMUM_POPULATION):
    """{name without accents: (geonameid, country_code, population, name)}, the most populous per name."""
    cities = {}
    with sqlite3.connect(gazetteer_path) as connection:
        rows = connection.execute(
            "SELECT geonameid, name, asciiname, country_code, population FROM geonames "
            "WHERE feature_class = 'P' AND population >= ?",
            (minimum_population,),
        )
        for geonameid, name, ascii_name, country_code, population in rows:
            for variant in {name, ascii_name}:
                if not variant or len(variant) <= 3:
                    continue
                key = _without_accents(variant)
                if key not in cities or cities[key][2] < population:
                    cities[key] = (geonameid, country_code, population, name)
    for alias, name in CITY_ALIASES.items():
        if name in cities:
            cities[alias] = cities[name]
    return {key: value for key, value in cities.items() if key not in NOT_CITIES}


def city_pattern(cities):
    """Regex matching a city name written as a place in a title (longest names first)."""
    names = "|".join(sorted((re.escape(name) for name in cities), key=len, reverse=True))
    return re.compile(rf"{PLACE_PREFIX}({names}){PLACE_SUFFIX}")


def title_city(title, cities, pattern):
    """Most populous city written as a place in `title`, or None."""
    best = None
    for match in pattern.finditer(_without_accents(title or "")):
        city = cities[match.group(1)]
        if best is None or city[2] > best[2]:
            best = city
    return best


def stratified_sample(candidates, size, per_city, per_country, seed):
    """At most `per_city` artworks per city and `per_country` per country, then `size` at random."""
    rng = random.Random(seed)
    by_city = defaultdict(list)
    for candidate in sorted(candidates, key=lambda item: item["artwork"]["id"]):
        by_city[candidate["city_id"]].append(candidate)
    kept_by_country = defaultdict(list)
    for city_id in sorted(by_city):
        city_artworks = by_city[city_id]
        kept_by_country[city_artworks[0]["country_code"]].extend(
            rng.sample(city_artworks, min(per_city, len(city_artworks)))
        )
    pool = []
    for country_code in sorted(kept_by_country):
        country_artworks = kept_by_country[country_code]
        pool.extend(rng.sample(country_artworks, min(per_country, len(country_artworks))))
    return rng.sample(pool, min(size, len(pool)))


def excluded_ids(gold_json, csv_paths):
    """Ids of the gold artworks and of the artworks already listed in annotation CSVs."""
    payload = json.loads(Path(gold_json).read_text(encoding="utf-8"))
    ids = {str(record["artwork"]["id"]) for record in payload["unique_artworks"]}
    for csv_path in csv_paths:
        with Path(csv_path).open(encoding="utf-8-sig", newline="") as csv_file:
            ids.update(row["artwork_id"] for row in csv.DictReader(csv_file))
    return ids


def main():
    """Scan the dump, sample the artworks and write them as records."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump-dir", type=Path, default=DUMP_DIR)
    parser.add_argument("--gazetteer", type=Path, default=Path("data/gazetter/geonames.db"))
    parser.add_argument("--gold-json", type=Path, default=Path("data/raw/gold_standard/gold_dataset.json"))
    parser.add_argument(
        "--exclude-csv",
        type=Path,
        nargs="*",
        default=[Path("results/annotation_precise_keywords_500.csv")],
        help="CSVs (colonne artwork_id) dont les œuvres sont déjà annotées ou en cours.",
    )
    parser.add_argument("--output", type=Path, default=Path("data/raw/aic/diverse_places_sample.json"))
    parser.add_argument("--size", type=int, default=800)
    parser.add_argument("--per-city", type=int, default=2)
    parser.add_argument("--per-country", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cities = load_cities(args.gazetteer)
    pattern = city_pattern(cities)
    excluded = excluded_ids(args.gold_json, args.exclude_csv)
    candidates = []
    for file_path in sorted(args.dump_dir.glob("*.json")):
        artwork = json.loads(file_path.read_text(encoding="utf-8"))
        if (
            artwork.get("artwork_type_title") not in PICTORIAL_TYPES
            or not artwork.get("image_id")
            or str(artwork.get("id")) in excluded
        ):
            continue
        city = title_city(artwork.get("title"), cities, pattern)
        if city is not None:
            candidates.append({
                "artwork": artwork,
                "city_id": city[0],
                "country_code": city[1],
                "city": city[3],
            })
    print(f"[info] {len(candidates)} œuvres figuratives dont le titre nomme une ville "
          f"({len({candidate['city_id'] for candidate in candidates})} villes).")

    sample = stratified_sample(candidates, args.size, args.per_city, args.per_country, args.seed)
    countries = Counter(candidate["country_code"] for candidate in sample)
    print(f"[info] {len(sample)} œuvres tirées: {len({c['city_id'] for c in sample})} villes, {len(countries)} pays.")
    print(f"[info] Pays les plus représentés: {countries.most_common(10)}")

    records = [
        {"artwork": candidate["artwork"], "sample": {"city": candidate["city"], "country_code": candidate["country_code"]}}
        for candidate in sample
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"unique_artworks": records}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"[ok] Échantillon écrit dans {args.output}")


if __name__ == "__main__":
    main()
