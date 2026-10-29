"""Turn a predict_sample_csv.py export into a CSV that is quick to annotate."""

import argparse
import csv
import json
import re
from pathlib import Path


DUMP_DIR = Path("data/raw/artic-api-data/json/artworks")
ANNOTATION_FIELDS = [
    "actual_place_status",
    "use_suggestion",
    "actual_precision",
    "actual_place_name",
    "actual_latitude",
    "actual_longitude",
    "actual_city",
    "actual_region",
    "actual_country",
    "review_notes",
]


def _one_line(value):
    """Keep every cell on one line, without ';' (it shifts columns in spreadsheets)."""
    if not isinstance(value, str):
        return value
    lines = [" ".join(line.split()) for line in value.splitlines()]
    return " | ".join(line for line in lines if line).replace(";", " |")


def _plain_text(html, limit=600):
    """Strip HTML tags and truncate to about `limit` characters."""
    text = re.sub(r"<[^>]+>", " ", html or "")
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " …"


def _dump_artwork(dump_dir, artwork_id):
    """Full AIC record of an artwork from the local dump ({} if absent)."""
    path = Path(dump_dir) / f"{artwork_id}.json"
    if not str(artwork_id).isdigit() or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def annotation_rows(prediction_rows, dump_dir=DUMP_DIR, statuses=("uncertain",)):
    """Build annotation rows for the predictions with the given statuses, nearest misses first."""
    selected = [row for row in prediction_rows if row["prediction_status"] in statuses]
    # Nearest misses first: they are the most useful to annotate.
    selected.sort(key=lambda row: -float(row["near_miss_margin"] or "-inf"))
    rows = []
    for number, row in enumerate(selected, start=1):
        artwork = _dump_artwork(dump_dir, row["artwork_id"])
        rows.append({key: _one_line(value) for key, value in {
            "review_id": f"U{number:04d}",
            "artwork_id": row["artwork_id"],
            "aic_url": row["aic_url"],
            "image_url": row["image_url"],
            "title": row["title"],
            "artwork_type": artwork.get("artwork_type_title"),
            "date_display": artwork.get("date_display"),
            "place_of_origin": row["place_of_origin"],
            "artist_display": row["artist_display"],
            "description": _plain_text(artwork.get("description") or artwork.get("short_description")),
            "candidates": row["candidates"],
            "suggested_place": row["place"],
            "suggested_precision": row["candidate_level"],
            "suggested_latitude": row["latitude"],
            "suggested_longitude": row["longitude"],
            "suggested_city": row["resolved_city"],
            "suggested_region": row["resolved_region"],
            "suggested_country": row["resolved_country"],
            **{field: "" for field in ANNOTATION_FIELDS},
        }.items()})
    return rows


def main():
    """Write the annotation CSV (refuses to overwrite an existing file)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-csv", type=Path, default=Path("results/test_sample_500_v3.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/annotation_uncertain_v3.csv"))
    parser.add_argument("--dump-dir", type=Path, default=DUMP_DIR)
    parser.add_argument("--status", nargs="+", default=["uncertain"], help="Statuts de prédiction à exporter.")
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"{args.output} existe déjà : choisir un autre --output pour ne pas écraser des annotations.")
    with args.predictions_csv.open(encoding="utf-8-sig", newline="") as input_file:
        prediction_rows = list(csv.DictReader(input_file))
    rows = annotation_rows(prediction_rows, args.dump_dir, tuple(args.status))
    if not rows:
        raise SystemExit("Aucune ligne à exporter.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] {len(rows)} œuvres à annoter dans {args.output}")


if __name__ == "__main__":
    main()
