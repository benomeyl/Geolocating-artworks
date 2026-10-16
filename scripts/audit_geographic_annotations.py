"""Audit completeness of the gold-standard geographic annotations."""

import argparse
import json
from collections import Counter
from pathlib import Path


DEFAULT_RECORDS = Path("data/raw/gold_standard/gold_dataset.json")
CANONICAL_PRECISION_LEVELS = {
    "exact_place",
    "neighborhood",
    "city",
    "region",
    "country",
    "unknown",
}


def load_records(path):
    with Path(path).open(encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if isinstance(payload, dict):
        return payload.get("unique_artworks", [])
    if isinstance(payload, list):
        return payload
    raise ValueError("Expected a JSON list or an object with 'unique_artworks'.")


def audit_annotations(records):
    depicted_records = [
        record
        for record in records
        if record.get("annotations", {}).get("place_status") == "depicted"
    ]
    places = [
        (record, (record.get("annotations", {}).get("depicted_place") or {}))
        for record in depicted_records
    ]
    precision_counts = Counter(
        place.get("precision", "<missing>") for _, place in places
    )
    missing = {
        "depicted_place": [],
        "coordinates": [],
        "invalid_coordinates": [],
        "precision": [],
        "noncanonical_precision": [],
        "city": [],
        "region": [],
        "country": [],
    }
    not_applicable = {
        "city": [],
        "region": [],
    }

    for record, place in places:
        artwork_id = record.get("artwork", record).get("id")
        if not place:
            missing["depicted_place"].append(artwork_id)
            continue
        if place.get("latitude") is None or place.get("longitude") is None:
            missing["coordinates"].append(artwork_id)
        else:
            try:
                latitude = float(place["latitude"])
                longitude = float(place["longitude"])
            except (TypeError, ValueError):
                latitude, longitude = None, None
            if (
                latitude is None
                or longitude is None
                or not -90 <= latitude <= 90
                or not -180 <= longitude <= 180
            ):
                missing["invalid_coordinates"].append(artwork_id)
        if place.get("precision") in (None, "", "unknown"):
            missing["precision"].append(artwork_id)
        elif place.get("precision") not in CANONICAL_PRECISION_LEVELS:
            missing["noncanonical_precision"].append(artwork_id)
        precision = place.get("precision")
        for field in ("city", "region", "country"):
            if field == "city" and precision in {"region", "country"}:
                not_applicable[field].append(artwork_id)
            elif field == "region" and precision == "country":
                not_applicable[field].append(artwork_id)
            elif not place.get(field):
                missing[field].append(artwork_id)

    return {
        "total_records": len(records),
        "depicted_records": len(depicted_records),
        "precision_counts": dict(sorted(precision_counts.items())),
        "canonical_precision_levels": sorted(CANONICAL_PRECISION_LEVELS),
        "missing_ids": missing,
        "missing_counts": {field: len(ids) for field, ids in missing.items()},
        "not_applicable_ids": not_applicable,
        "not_applicable_counts": {
            field: len(ids) for field, ids in not_applicable.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    args = parser.parse_args()
    report = audit_annotations(load_records(args.records))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()