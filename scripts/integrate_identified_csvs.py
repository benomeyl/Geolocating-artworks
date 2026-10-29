"""Sample described Dresden/Venice CSV rows into gold_dataset.json."""

import argparse
import ast
import csv
import json
import math
from pathlib import Path


GOLD_STANDARD_DIR = Path("data/raw/gold_standard")
CSV_DATASETS = {
    "dresden": (
        "dresden_identified_data_ground_truth.csv",
        "Dresden",
        "Saxony",
        "Germany",
    ),
    "venice": (
        "venice_identified_data_ground_truth.csv",
        "Venice",
        "Veneto",
        "Italy",
    ),
}
ALL_CSV_FILENAMES = {
    "dresden_identified_data_ground_truth.csv",
    "paris_identified_data_ground_truth.csv",
    "venice_identified_data_ground_truth.csv",
}
MAX_ROWS_PER_CITY = 20


def _find_column(row, *names):
    """Return the key of `row` matching one of `names` (case- and space-insensitive), or None."""
    normalized_names = {name.casefold() for name in names}
    for key in row:
        if key and key.strip().casefold() in normalized_names:
            return key
    return None


def parse_monuments(raw_value):
    """Parse a monument cell (a name or a Python list literal) into unique names, in order."""
    if raw_value is None or not raw_value.strip():
        return []
    value = raw_value.strip()
    if value.startswith(("[", "(")):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError) as error:
            raise ValueError(f"Cannot parse monument list: {raw_value!r}") from error
    else:
        parsed = value
    values = parsed if isinstance(parsed, (list, tuple, set)) else [parsed]

    monuments = []
    seen = set()
    for item in values:
        name = str(item).strip()
        normalized = " ".join(name.casefold().split())
        if name and normalized not in seen:
            monuments.append(name)
            seen.add(normalized)
    return monuments


def parse_coordinates(raw_value):
    """Parse a "(latitude, longitude)" cell; None when empty, ValueError when invalid."""
    if raw_value is None or not raw_value.strip():
        return None
    try:
        value = ast.literal_eval(raw_value.strip())
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError
        latitude, longitude = map(float, value)
    except (SyntaxError, TypeError, ValueError) as error:
        raise ValueError(f"Cannot parse latitude/longitude: {raw_value!r}") from error
    if (
        not math.isfinite(latitude)
        or not math.isfinite(longitude)
        or not -90 <= latitude <= 90
        or not -180 <= longitude <= 180
    ):
        raise ValueError(f"Coordinates out of range: {raw_value!r}")
    return latitude, longitude


def select_described_rows(rows, maximum=MAX_ROWS_PER_CITY):
    """Pick up to `maximum` rows that have a description, one per Wikidata id first, then one per image."""
    description_column = _find_column(rows[0], "description") if rows else None
    wikidata_column = _find_column(rows[0], "wd_id") if rows else None
    image_column = _find_column(rows[0], "Image Name") if rows else None
    if description_column is None or image_column is None:
        return []

    eligible_rows = [
        row for row in rows if (row.get(description_column) or "").strip()
    ]
    selected = []
    selected_images = set()
    selected_wikidata_ids = set()

    if wikidata_column is not None:
        for row in eligible_rows:
            wikidata_id = (row.get(wikidata_column) or "").strip()
            image_name = (row.get(image_column) or "").strip()
            if not wikidata_id or wikidata_id in selected_wikidata_ids:
                continue
            selected.append(row)
            selected_wikidata_ids.add(wikidata_id)
            selected_images.add(image_name.casefold())
            if len(selected) >= maximum:
                return selected

    for row in eligible_rows:
        image_name = (row.get(image_column) or "").strip()
        if image_name.casefold() in selected_images:
            continue
        selected.append(row)
        selected_images.add(image_name.casefold())
        if len(selected) >= maximum:
            break
    return selected


def record_from_csv_row(dataset_key, city, region, country, csv_name, row, row_number):
    """Convert one CSV row into a gold record whose depicted places are its monuments (exact_place)."""
    image_column = _find_column(row, "Image Name")
    monument_column = _find_column(row, "Monument", "Monuments")
    coordinates_column = _find_column(row, "lat_lon")
    wikidata_column = _find_column(row, "wd_id")
    title_column = _find_column(row, "title")
    description_column = _find_column(row, "description")
    date_column = _find_column(row, "date")

    image_name = (row.get(image_column) or "").strip() if image_column else ""
    if not image_name:
        raise ValueError(f"{csv_name} row {row_number}: missing Image Name")
    monuments = parse_monuments(row.get(monument_column) or "") if monument_column else []
    if not monuments:
        raise ValueError(f"{csv_name} row {row_number}: missing monument")

    coordinates = parse_coordinates(row.get(coordinates_column)) if coordinates_column else None
    latitude, longitude = coordinates if coordinates is not None else (None, None)
    wikidata_id = (row.get(wikidata_column) or "").strip() if wikidata_column else ""
    description = (row.get(description_column) or "").strip() if description_column else ""
    title = (row.get(title_column) or "").strip() if title_column else ""
    date_display = (row.get(date_column) or "").strip() if date_column else ""

    depicted_places = [
        {
            "name": monument,
            "city": city,
            "region": region,
            "country": country,
            "latitude": latitude,
            "longitude": longitude,
            "geonameid": None,
            "geonameid_match": "none",
            "precision": "exact_place",
            "wikidata_id": wikidata_id or None,
            "coordinate_source": csv_name if coordinates is not None else None,
        }
        for monument in monuments
    ]
    primary_target = depicted_places[0]
    dataset_metadata = {
        "dataset": dataset_key,
        "csv_file": csv_name,
        "csv_row_number": row_number,
        "image_name": image_name,
        "source_description": description or None,
        "source_date": date_display or None,
    }
    artwork = {
        "id": f"identified-csv:{dataset_key}:{image_name}",
        "source": "identified_data_ground_truth_csv",
        "image_id": image_name,
        "title": title or "; ".join(monuments),
        "description": description or None,
        "date_display": date_display or None,
    }
    annotations = {
        "place_status": "depicted",
        "place_type": "real",
        "depicted_place": primary_target,
        "depicted_places": depicted_places,
        "mentioned_places": [
            {"name": monument, "role": "depicted"}
            for monument in monuments
        ],
        "annotation_basis": "identified_data_ground_truth_csv",
        "ground_truth_source": dataset_metadata,
        "confidence": "high" if coordinates is not None else "medium",
        "status": "verified",
    }
    return {"artwork": artwork, "matched_queries": [], "annotations": annotations}


def load_csv_records(directory=GOLD_STANDARD_DIR):
    """Read the Dresden and Venice CSVs and return the gold records of their selected rows."""
    imported = []
    for dataset_key, (filename, city, region, country) in CSV_DATASETS.items():
        csv_path = directory / filename
        with csv_path.open(encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            rows = []
            for row_number, row in enumerate(reader, start=2):
                row["__csv_row_number"] = row_number
                rows.append(row)
            selected_rows = select_described_rows(rows)
            for row in selected_rows:
                imported.append(
                    record_from_csv_row(
                        dataset_key,
                        city,
                        region,
                        country,
                        filename,
                        row,
                        int(row["__csv_row_number"]),
                    )
                )
    return imported


def merge_records(payload, imported):
    """Replace previous CSV imports in the gold payload by `imported`.

    Returns (merged_payload, additions, already_present_count, replaced_count).
    Raises ValueError on an image id collision with another record.
    """
    existing_records = payload.get("unique_artworks")
    if not isinstance(existing_records, list):
        raise ValueError("gold_dataset.json must contain a 'unique_artworks' list.")

    retained_records = []
    replaced_count = 0
    for record in existing_records:
        source = record.get("annotations", {}).get("ground_truth_source", {})
        if source.get("csv_file") in ALL_CSV_FILENAMES:
            replaced_count += 1
        else:
            retained_records.append(record)

    existing_ids = {
        record.get("artwork", {}).get("id")
        for record in retained_records
    }
    existing_images = {
        str(record.get("artwork", {}).get("image_id", "")).casefold()
        for record in retained_records
        if record.get("artwork", {}).get("image_id")
    }
    additions = []
    skipped = 0
    seen_images = set()

    for record in imported:
        artwork = record["artwork"]
        record_id = artwork["id"]
        image_id = artwork["image_id"].casefold()
        if record_id in existing_ids:
            skipped += 1
            continue
        if image_id in existing_images or image_id in seen_images:
            raise ValueError(f"Image ID collision: {artwork['image_id']}")
        additions.append(record)
        existing_ids.add(record_id)
        seen_images.add(image_id)

    merged_payload = dict(payload)
    merged_payload["unique_artworks"] = retained_records + additions
    return merged_payload, additions, skipped, replaced_count


def main():
    """Preview the import, and write gold_dataset.json with --write."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-json", type=Path, default=GOLD_STANDARD_DIR / "gold_dataset.json")
    parser.add_argument("--csv-dir", type=Path, default=GOLD_STANDARD_DIR)
    parser.add_argument("--write", action="store_true", help="Écrire les nouveaux enregistrements dans le JSON.")
    args = parser.parse_args()

    with args.gold_json.open(encoding="utf-8") as json_file:
        payload = json.load(json_file)
    imported = load_csv_records(args.csv_dir)
    merged_payload, additions, skipped, replaced_count = merge_records(payload, imported)

    multi_target_count = sum(
        len(record["annotations"]["depicted_places"]) > 1
        for record in additions
    )
    without_coordinates = sum(
        record["annotations"]["depicted_place"]["latitude"] is None
        for record in additions
    )
    print(f"Described CSV records selected: {len(imported)}")
    selected_by_dataset = {}
    for record in additions:
        dataset = record["annotations"]["ground_truth_source"]["dataset"]
        selected_by_dataset[dataset] = selected_by_dataset.get(dataset, 0) + 1
    print(f"Selected by city: {selected_by_dataset}")
    print(f"Previous CSV records to replace: {replaced_count}")
    print(f"Selected records to add: {len(additions)}; already present: {skipped}")
    print(f"Records with multiple targets: {multi_target_count}")
    print(f"Records without coordinates: {without_coordinates}")
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