import csv
import json

from scripts.create_manual_review_sample import build_review_sample, write_review_files


def _prediction(artwork_id, candidates, status="uncertain"):
    return {
        "id": artwork_id,
        "title": f"Artwork {artwork_id}",
        "place_status_predicted": status,
        "top_candidate": None,
        "all_candidates": candidates,
    }


def _candidate(name, *, geolocation=None, level="city", scores=None, decision=None):
    return {
        "place": name,
        "candidate_level": level,
        "score_ner_max": 0.8,
        "level_scores": scores or {},
        "decision": decision,
        "geolocation": geolocation,
        "champs": ["title"],
    }


def test_review_sample_includes_all_predictions_and_stratifies_abstentions():
    model = {
        "deployable_levels": ["city"],
        "levels": {
            "city": {"status": "active", "threshold": 0.8},
            "neighborhood": {"status": "active", "threshold": 0.8},
        },
    }
    geo = {"latitude": 46.5, "longitude": 6.6, "source": "GeoNames"}
    predictions = [
        _prediction(1, [], "depicted"),
        _prediction(2, []),
        _prediction(3, [_candidate("Near", geolocation=geo, scores={"city": 0.75})]),
        _prediction(4, [_candidate("Mid", geolocation=geo, scores={"city": 0.55})]),
        _prediction(5, [_candidate("Far", geolocation=geo, scores={"city": 0.3})]),
        _prediction(6, [_candidate("Ungeocoded", geolocation=None)]),
        _prediction(7, [_candidate("Unclassified", geolocation=geo, level=None)]),
        _prediction(8, [_candidate("Not deployable", geolocation=geo, level="exact_place", scores={"neighborhood": 0.9})]),
    ]
    sizes = {
        "no_ner_candidate": 1,
        "near_threshold": 1,
        "mid_threshold": 1,
        "far_threshold": 1,
        "candidate_ungeocoded": 1,
        "geocoded_level_unclassified": 1,
        "threshold_met_but_not_deployable": 1,
    }

    rows, metadata = build_review_sample(predictions, model, [], seed=17, sample_sizes=sizes)
    repeated_rows, _ = build_review_sample(predictions, model, [], seed=17, sample_sizes=sizes)

    assert [row["artwork_id"] for row in rows] == [row["artwork_id"] for row in repeated_rows]
    assert len(rows) == 8
    assert metadata["population_by_stratum"]["accepted_prediction"] == 1
    assert metadata["sample_by_stratum"]["accepted_prediction"] == 1
    assert metadata["population_by_stratum"]["no_ner_candidate"] == 1
    assert metadata["population_by_stratum"]["near_threshold"] == 1
    assert metadata["population_by_stratum"]["mid_threshold"] == 1
    assert metadata["population_by_stratum"]["far_threshold"] == 1
    assert metadata["population_by_stratum"]["candidate_ungeocoded"] == 1
    assert metadata["population_by_stratum"]["geocoded_level_unclassified"] == 1
    assert metadata["population_by_stratum"]["threshold_met_but_not_deployable"] == 1


def test_review_sample_export_has_blank_annotation_fields(tmp_path):
    model = {"deployable_levels": [], "levels": {}}
    predictions = [_prediction(42, [])]
    records = [{
        "artwork": {
            "id": 42,
            "description": "Description for review",
            "subject_titles": ["cityscape"],
        },
        "matched_queries": ["street"],
    }]
    rows, metadata = build_review_sample(predictions, model, records, seed=42)
    csv_path = tmp_path / "review.csv"
    json_path = tmp_path / "review_metadata.json"

    write_review_files(rows, metadata, csv_path, json_path)

    with csv_path.open(encoding="utf-8-sig", newline="") as review_file:
        exported = next(csv.DictReader(review_file))
    exported_metadata = json.loads(json_path.read_text(encoding="utf-8"))
    assert exported["artwork_id"] == "42"
    assert exported["description"] == "Description for review"
    assert exported["actual_place_status"] == ""
    assert exported["actual_latitude"] == ""
    assert exported_metadata["seed"] == 42
    assert exported_metadata["population_by_stratum"]["no_ner_candidate"] == 1