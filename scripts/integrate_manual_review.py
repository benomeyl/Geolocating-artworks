"""Add manually reviewed artworks to gold_dataset.json.

Accepts both manual_review_sample.csv and the CSV produced by
create_annotation_csv.py. In the latter, `use_suggestion` = oui copies the
suggested_* columns into the empty actual_* columns.
"""

import argparse
import csv
import json
import math
from collections import Counter
from datetime import date
from pathlib import Path


GOLD_JSON = Path("data/raw/gold_standard/gold_dataset.json")
REVIEW_CSV = Path("results/manual_review_sample.csv")
RECORDS_JSONS = [
    Path("data/raw/aic/search_multiple_queries_all_artworks_20260917T125145Z.json"),
    Path("data/raw/aic/aic_useful_fields.json"),
]

PLACE_STATUSES = {"depicted", "none", "mentioned_only", "undeterminable"}
PLACE_STATUS_ALIASES = {
    "d": "depicted",
    "n": "none",
    "m": "mentioned_only",
    "u": "undeterminable",
    "mentionned_only": "mentioned_only",
    "mentioned": "mentioned_only",
    "mentioned only": "mentioned_only",
    "no": "none",
}
PRECISIONS = {"exact_place", "neighborhood", "city", "region", "country", "unknown"}
PRECISION_ALIASES = {
    "neighbourhood": "neighborhood",
    "landmark": "exact_place",
    "none": None,
    "": None,
}
TRUE_VALUES = {"oui", "o", "yes", "y", "1", "x", "ok", "true"}
SUGGESTION_COLUMNS = {
    "actual_place_name": "suggested_place",
    "actual_precision": "suggested_precision",
    "actual_latitude": "suggested_latitude",
    "actual_longitude": "suggested_longitude",
    "actual_city": "suggested_city",
    "actual_region": "suggested_region",
    "actual_country": "suggested_country",
}
ARTWORK_FIELDS = [
    "id",
    "title",
    "description",
    "short_description",
    "subject_titles",
    "theme_titles",
    "style_titles",
    "image_id",
    "is_public_domain",
]


def normalize_place_status(raw_value):
    value = " ".join((raw_value or "").casefold().split())
    value = PLACE_STATUS_ALIASES.get(value, value)
    if not value:
        return None
    if value not in PLACE_STATUSES:
        raise ValueError(f"Unknown place status: {raw_value!r}")
    return value


def normalize_precision(raw_value):
    value = " ".join((raw_value or "").casefold().split())
    value = PRECISION_ALIASES.get(value, value)
    if value is not None and value not in PRECISIONS:
        raise ValueError(f"Unknown precision: {raw_value!r}")
    return value


def parse_coordinate(raw_value, minimum, maximum):
    value = (raw_value or "").strip().replace(",", ".")
    if not value:
        return None
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"Coordinate out of range: {raw_value!r}")
    return number


def apply_suggestion(row):
    """Fill empty actual_* values from suggested_* when use_suggestion is set."""
    if (row.get("use_suggestion") or "").strip().casefold() not in TRUE_VALUES:
        return row
    status = normalize_place_status(row.get("actual_place_status"))
    if status not in (None, "depicted"):
        raise ValueError(f"use_suggestion with place status {status!r}")
    filled = {**row, "actual_place_status": "depicted"}
    for actual_column, suggested_column in SUGGESTION_COLUMNS.items():
        if not (filled.get(actual_column) or "").strip():
            filled[actual_column] = row.get(suggested_column) or ""
    return filled


def _artwork_id(raw_id):
    raw_id = (raw_id or "").strip()
    return int(raw_id) if raw_id.isdigit() else raw_id


def _clean(value):
    value = (value or "").strip()
    return value or None


def record_from_review_row(row, artwork, csv_name, import_date):
    """Build a gold record, or return None when the row was not annotated."""
    row = apply_suggestion(row)
    status = normalize_place_status(row.get("actual_place_status"))
    if status is None:
        return None

    depicted_place = None
    warnings = []
    if status == "depicted":
        latitude = parse_coordinate(row.get("actual_latitude"), -90, 90)
        longitude = parse_coordinate(row.get("actual_longitude"), -180, 180)
        if (latitude is None) != (longitude is None):
            raise ValueError("latitude and longitude must be filled together")
        precision = normalize_precision(row.get("actual_precision"))
        if precision is None:
            precision = "unknown"
            warnings.append("depicted without precision")
        name = _clean(row.get("actual_place_name"))
        if name is None:
            warnings.append("depicted without place name")
        if latitude is None:
            warnings.append("depicted without coordinates")
        depicted_place = {
            "name": name,
            "city": _clean(row.get("actual_city")),
            "region": _clean(row.get("actual_region")),
            "country": _clean(row.get("actual_country")),
            "latitude": latitude,
            "longitude": longitude,
            "geonameid": None,
            "geonameid_match": "none",
            "precision": precision,
            "coordinate_source": "manual_review" if latitude is not None else None,
        }

    annotations = {
        "place_status": status,
        "place_type": "real" if status == "depicted" else None,
        "depicted_place": depicted_place,
        "mentioned_places": [],
        "annotation_basis": "manual_review",
        "ground_truth_source": {
            "dataset": "manual_review",
            "csv_file": csv_name,
            "review_id": row.get("review_id"),
            "sample_stratum": row.get("sample_stratum"),
            "review_notes": _clean(row.get("review_notes")),
            "used_suggestion": (row.get("use_suggestion") or "").strip().casefold() in TRUE_VALUES,
        },
        # Le CSV ne note pas la confiance de l'annotateur : on ne l'invente pas.
        "confidence": None,
        "status": "verified",
        "annotator": "manual_review",
        "date": import_date,
    }
    return {
        "artwork": {field: artwork.get(field) for field in ARTWORK_FIELDS if field in artwork},
        "matched_queries": [],
        "annotations": annotations,
    }, warnings


def load_review_records(csv_path, records_paths, import_date=None):
    import_date = import_date or date.today().isoformat()
    if isinstance(records_paths, (str, Path)):
        records_paths = [records_paths]
    artworks_by_id = {}
    for records_path in records_paths:
        if not Path(records_path).is_file():
            continue
        payload = json.loads(Path(records_path).read_text(encoding="utf-8"))
        source_records = payload.get("unique_artworks", []) if isinstance(payload, dict) else payload
        for record in source_records:
            artwork = record.get("artwork", record)
            artworks_by_id.setdefault(artwork.get("id"), artwork)

    imported = []
    skipped_unannotated = []
    warnings = []
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as csv_file:
        for row_number, row in enumerate(csv.DictReader(csv_file), start=2):
            artwork_id = _artwork_id(row.get("artwork_id"))
            artwork = artworks_by_id.get(artwork_id) or {"id": artwork_id, "title": row.get("title")}
            try:
                result = record_from_review_row(row, artwork, Path(csv_path).name, import_date)
            except ValueError as error:
                raise ValueError(f"{csv_path} row {row_number}: {error}") from error
            if result is None:
                skipped_unannotated.append(row.get("review_id") or row_number)
                continue
            record, row_warnings = result
            imported.append(record)
            warnings.extend(f"{row.get('review_id')} ({artwork_id}): {warning}" for warning in row_warnings)
    return imported, skipped_unannotated, warnings


def merge_records(payload, imported):
    """Append reviewed records; existing gold annotations always win."""
    existing_records = payload.get("unique_artworks")
    if not isinstance(existing_records, list):
        raise ValueError("gold_dataset.json must contain a 'unique_artworks' list.")
    existing_by_id = {
        str(record.get("artwork", {}).get("id")): record for record in existing_records
    }

    additions = []
    already_present = []
    conflicts = []
    for record in imported:
        key = str(record["artwork"]["id"])
        existing = existing_by_id.get(key)
        if existing is None:
            additions.append(record)
            existing_by_id[key] = record
            continue
        already_present.append(key)
        existing_status = existing.get("annotations", {}).get("place_status")
        if existing_status != record["annotations"]["place_status"]:
            conflicts.append((key, existing_status, record["annotations"]["place_status"]))

    merged_payload = dict(payload)
    merged_payload["unique_artworks"] = existing_records + additions
    return merged_payload, additions, already_present, conflicts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-json", type=Path, default=GOLD_JSON)
    parser.add_argument("--review-csv", type=Path, default=REVIEW_CSV)
    parser.add_argument(
        "--records",
        type=Path,
        nargs="+",
        default=RECORDS_JSONS,
        help="Fichiers d'œuvres où retrouver titre/description (le premier qui contient l'œuvre gagne).",
    )
    parser.add_argument("--write", action="store_true", help="Écrire les nouveaux enregistrements dans le JSON.")
    args = parser.parse_args()

    payload = json.loads(args.gold_json.read_text(encoding="utf-8"))
    imported, skipped_unannotated, warnings = load_review_records(args.review_csv, args.records)
    merged_payload, additions, already_present, conflicts = merge_records(payload, imported)

    print(f"Annotated review rows: {len(imported)}; not annotated: {len(skipped_unannotated)}")
    print(f"Place status of additions: {dict(Counter(r['annotations']['place_status'] for r in additions))}")
    print(
        "Precision of depicted additions: "
        f"{dict(Counter(r['annotations']['depicted_place']['precision'] for r in additions if r['annotations']['depicted_place']))}"
    )
    print(f"Records to add: {len(additions)}; already in gold (kept unchanged): {len(already_present)}")
    for artwork_id, existing_status, review_status in conflicts:
        print(f"[conflict] {artwork_id}: gold={existing_status}, review={review_status} (gold kept)")
    for warning in warnings:
        print(f"[warning] {warning}")
    print(f"Total unique_artworks after merge: {len(merged_payload['unique_artworks'])}")

    if not args.write:
        print("Dry run only. Re-run with --write to update gold_dataset.json.")
        return

    temporary_path = args.gold_json.with_suffix(args.gold_json.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(merged_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(args.gold_json)
    print(f"Updated {args.gold_json}")


if __name__ == "__main__":
    main()
