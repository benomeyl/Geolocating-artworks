"""Review the gold artworks annotated without a depicted place whose title names a city.

Guideline decided on 2026-11-05: an artwork titled "place, city" or "scene,
city" (a photograph "Untitled (Subway Portrait, New York)") depicts that
city, at least at the city level, unless it is an object kept there
("Jewelled Bowl in the Treasury of San Marco, Venice").

Without --apply, the gold artworks annotated "none" or "mentioned_only"
whose title names a city as a place (sample_diverse_places.title_city) are
written to a review CSV, with the city as the proposed depicted place.
Fill `review_accept` (oui/non) and optionally `review_precision` (city by
default, or exact_place / neighborhood with `review_place_name` and
coordinates), then run with --apply to update gold_dataset.json. The former
annotation is kept under annotations.previous_annotation.

    python scripts/relabel_city_in_title.py
    python scripts/relabel_city_in_title.py --apply
"""

import argparse
import csv
import json
import sqlite3
import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from api.aic import enrich_records_from_dump  # noqa: E402
from data.filtering_logreg import NON_PICTORIAL_ARTWORK_TYPES  # noqa: E402
from integrate_manual_review import TRUE_VALUES, normalize_precision, parse_coordinate  # noqa: E402
from sample_diverse_places import city_pattern, load_cities, title_city  # noqa: E402

GOLD_JSON = Path("data/raw/gold_standard/gold_dataset.json")
REVIEW_CSV = Path("results/relabel_city_in_title.csv")
STATUSES_TO_REVIEW = {"none", "mentioned_only"}
COLUMNS = [
    "artwork_id", "aic_url", "title", "artwork_type", "current_status", "current_notes",
    "proposed_city", "proposed_country_code", "proposed_latitude", "proposed_longitude",
    "review_accept", "review_precision", "review_place_name", "review_latitude", "review_longitude", "review_notes",
]


def city_coordinates(gazetteer_path, geonameid):
    """(latitude, longitude) of a GeoNames entry."""
    with sqlite3.connect(gazetteer_path) as connection:
        return connection.execute(
            "SELECT latitude, longitude FROM geonames WHERE geonameid = ?", (geonameid,)
        ).fetchone()


def review_rows(records, cities, pattern, gazetteer_path):
    """Rows of the review CSV: unplaced pictorial gold artworks whose title names a city."""
    rows = []
    for record in records:
        artwork, annotations = record["artwork"], record.get("annotations", {})
        if annotations.get("place_status") not in STATUSES_TO_REVIEW:
            continue
        if str(artwork.get("artwork_type_title") or "").casefold() in NON_PICTORIAL_ARTWORK_TYPES:
            continue
        city = title_city(artwork.get("title"), cities, pattern)
        if city is None:
            continue
        latitude, longitude = city_coordinates(gazetteer_path, city[0])
        artwork_id = artwork.get("id")
        rows.append({
            "artwork_id": artwork_id,
            "aic_url": f"https://www.artic.edu/artworks/{artwork_id}" if str(artwork_id).isdigit() else "",
            "title": artwork.get("title"),
            "artwork_type": artwork.get("artwork_type_title") or "",
            "current_status": annotations.get("place_status"),
            "current_notes": (annotations.get("ground_truth_source") or {}).get("review_notes") or "",
            "proposed_city": city[3],
            "proposed_country_code": city[1],
            "proposed_latitude": latitude,
            "proposed_longitude": longitude,
            **{column: "" for column in COLUMNS if column.startswith("review_")},
        })
    return rows


def relabelled_annotations(annotations, row, import_date):
    """Annotations of an accepted row: depicted at the reviewed precision (city by default)."""
    precision = normalize_precision(row.get("review_precision")) or "city"
    latitude = parse_coordinate(row.get("review_latitude"), -90, 90)
    longitude = parse_coordinate(row.get("review_longitude"), -180, 180)
    if latitude is None or longitude is None:
        latitude, longitude = float(row["proposed_latitude"]), float(row["proposed_longitude"])
    name = (row.get("review_place_name") or "").strip() or row["proposed_city"]
    return {
        **annotations,
        "place_status": "depicted",
        "place_type": "real",
        "depicted_place": {
            "name": name,
            "city": row["proposed_city"],
            "region": None,
            "country": row["proposed_country_code"],
            "latitude": latitude,
            "longitude": longitude,
            "geonameid": None,
            "geonameid_match": "none",
            "precision": precision,
            "coordinate_source": "manual_review" if row.get("review_latitude") else "geonames_city",
        },
        "annotation_basis": "relabel_city_in_title",
        "previous_annotation": {
            key: annotations.get(key) for key in ("place_status", "depicted_place", "annotation_basis", "date")
        },
        "relabel_notes": (row.get("review_notes") or "").strip() or None,
        "date": import_date,
    }


def apply_review(payload, csv_path, import_date):
    """Update the accepted artworks of the review CSV in place; return the number of changes."""
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as csv_file:
        accepted = {
            row["artwork_id"]: row
            for row in csv.DictReader(csv_file)
            if (row.get("review_accept") or "").strip().casefold() in TRUE_VALUES
        }
    changed = 0
    for record in payload["unique_artworks"]:
        row = accepted.get(str(record["artwork"].get("id")))
        if row is not None and record["annotations"].get("place_status") in STATUSES_TO_REVIEW:
            record["annotations"] = relabelled_annotations(record["annotations"], row, import_date)
            changed += 1
    return changed


def main():
    """Write the review CSV, or apply it to the gold standard with --apply."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold-json", type=Path, default=GOLD_JSON)
    parser.add_argument("--review-csv", type=Path, default=REVIEW_CSV)
    parser.add_argument("--gazetteer", type=Path, default=Path("data/gazetter/geonames.db"))
    parser.add_argument("--apply", action="store_true", help="Appliquer les lignes acceptées au gold standard.")
    args = parser.parse_args()

    payload = json.loads(args.gold_json.read_text(encoding="utf-8"))
    if args.apply:
        changed = apply_review(payload, args.review_csv, date.today().isoformat())
        temporary_path = args.gold_json.with_suffix(args.gold_json.suffix + ".tmp")
        temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary_path.replace(args.gold_json)
        print(f"[ok] {changed} œuvres passées en depicted dans {args.gold_json}")
        return

    records = payload["unique_artworks"]
    enrich_records_from_dump(records)
    cities = load_cities(args.gazetteer)
    rows = review_rows(records, cities, city_pattern(cities), args.gazetteer)
    with args.review_csv.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[info] Statuts actuels: {dict(Counter(row['current_status'] for row in rows))}")
    print(f"[ok] {len(rows)} œuvres à relire écrites dans {args.review_csv}")


if __name__ == "__main__":
    main()
