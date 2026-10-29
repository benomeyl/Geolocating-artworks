import csv
import json

import pytest

from scripts.integrate_manual_review import (
    load_review_records,
    merge_records,
    normalize_place_status,
    normalize_precision,
)


def _write_review(tmp_path, rows):
    fieldnames = [
        "review_id", "sample_stratum", "artwork_id", "title", "actual_place_status",
        "actual_precision", "actual_place_name", "actual_latitude", "actual_longitude",
        "actual_city", "actual_region", "actual_country", "review_outcome", "review_notes",
    ]
    path = tmp_path / "review.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    records = tmp_path / "records.json"
    records.write_text(json.dumps({"unique_artworks": [
        {"artwork": {"id": 16571, "title": "Gare Saint-Lazare", "description": "Monet", "latitude": 41.88}},
        {"artwork": {"id": 2848, "title": "Scene from Bohemian Life"}},
    ]}), encoding="utf-8")
    return path, records


def test_review_spellings_are_normalized():
    assert normalize_place_status(" Mentionned_only ") == "mentioned_only"
    assert normalize_precision("neighbourhood ") == "neighborhood"
    assert normalize_precision("none") is None
    with pytest.raises(ValueError):
        normalize_place_status("maybe")


def test_review_rows_become_gold_records(tmp_path):
    csv_path, records_path = _write_review(tmp_path, [
        {
            "review_id": "R0001", "artwork_id": "16571", "actual_place_status": "depicted",
            "actual_precision": "exact_place", "actual_place_name": "Gare Saint-Lazare",
            "actual_latitude": "48,8770414", "actual_longitude": "2.3250674",
            "actual_city": "Paris", "actual_country": "France",
        },
        {"review_id": "R0002", "artwork_id": "2848", "actual_place_status": "none"},
        {"review_id": "R0003", "artwork_id": "999"},
    ])

    imported, unannotated, warnings = load_review_records(csv_path, records_path, "2026-10-26")

    assert unannotated == ["R0003"]
    assert warnings == []
    depicted = imported[0]
    assert depicted["artwork"] == {"id": 16571, "title": "Gare Saint-Lazare", "description": "Monet"}
    assert depicted["annotations"]["depicted_place"]["latitude"] == pytest.approx(48.8770414)
    assert depicted["annotations"]["depicted_place"]["precision"] == "exact_place"
    assert imported[1]["annotations"]["place_status"] == "none"
    assert imported[1]["annotations"]["depicted_place"] is None


def test_existing_gold_annotations_are_kept(tmp_path):
    csv_path, records_path = _write_review(tmp_path, [
        {"review_id": "R0001", "artwork_id": "2848", "actual_place_status": "none"},
    ])
    imported, _, _ = load_review_records(csv_path, records_path)
    payload = {"unique_artworks": [
        {"artwork": {"id": 2848}, "annotations": {"place_status": "depicted"}},
    ]}

    merged, additions, already_present, conflicts = merge_records(payload, imported)

    assert additions == []
    assert already_present == ["2848"]
    assert conflicts == [("2848", "depicted", "none")]
    assert len(merged["unique_artworks"]) == 1


def test_use_suggestion_fills_empty_actual_columns(tmp_path):
    from scripts.integrate_manual_review import apply_suggestion

    row = {
        "actual_place_status": "",
        "use_suggestion": "oui",
        "actual_precision": "city",
        "suggested_place": "Navy Pier",
        "suggested_precision": "exact_place",
        "suggested_latitude": "41.8917",
        "suggested_longitude": "-87.6086",
        "suggested_city": "Chicago",
    }

    filled = apply_suggestion(row)

    assert filled["actual_place_status"] == "depicted"
    assert filled["actual_place_name"] == "Navy Pier"
    assert filled["actual_precision"] == "city"  # une valeur saisie l'emporte
    assert filled["actual_city"] == "Chicago"
    with pytest.raises(ValueError):
        apply_suggestion({**row, "actual_place_status": "n"})


def test_short_status_codes():
    assert normalize_place_status("d") == "depicted"
    assert normalize_place_status("N") == "none"
    assert normalize_place_status("u") == "undeterminable"


def test_annotation_csv_keeps_actual_columns_empty(tmp_path):
    from scripts.create_annotation_csv import annotation_rows

    prediction = {
        "artwork_id": "Q1", "aic_url": "", "image_url": "", "title": "Navy Pier, Chicago",
        "place_of_origin": "", "artist_display": "", "prediction_status": "uncertain",
        "candidates": "Navy Pier [exact_place]", "place": "Navy Pier", "candidate_level": "exact_place",
        "latitude": "41.89", "longitude": "-87.60", "resolved_city": "Chicago",
        "resolved_region": "Illinois", "resolved_country": "United States", "near_miss_margin": "-0.03",
    }
    rows = annotation_rows([prediction, {**prediction, "prediction_status": "no_candidate"}], tmp_path)

    assert len(rows) == 1
    assert rows[0]["suggested_place"] == "Navy Pier"
    assert rows[0]["actual_place_name"] == "" and rows[0]["use_suggestion"] == ""


def test_annotation_cells_stay_on_one_line():
    from scripts.create_annotation_csv import _one_line

    assert _one_line("Claude Monet\nFrench, 1840–1926") == "Claude Monet | French, 1840–1926"
    assert _one_line("Paris [city]; Seine [region]") == "Paris [city] | Seine [region]"
    assert _one_line(None) is None


def test_rows_without_artwork_id_are_skipped_with_a_warning(tmp_path):
    csv_path, records_path = _write_review(tmp_path, [
        {"review_id": "", "artwork_id": "", "actual_place_status": "n"},
        {"review_id": "", "artwork_id": ""},
    ])

    imported, unannotated, warnings = load_review_records(csv_path, records_path)

    assert imported == [] and unannotated == []
    assert warnings == ["row 2: annotated but without artwork_id, skipped"]
