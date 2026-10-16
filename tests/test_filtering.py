import json
from src.data.filtering import (
    build_training_table,
    compare_heuristic_places_with_annotations,
    evaluate_binary_predictions,
    evaluate_place_identity,
    filter_by_keywords,
    predict_with_model,
    train_logistic_model,
)
from types import SimpleNamespace


def test_compare_heuristic_places_with_annotations_counts_actual_and_accepted_places():
    records = [
        {
            "artwork": {"id": 1},
            "annotations": {"place_status": "depicted", "depicted_place": {"name": "Paris"}},
        },
        {
            "artwork": {"id": 2},
            "annotations": {"place_status": "depicted", "depicted_place": {"name": "London"}},
        },
        {
            "artwork": {"id": 3},
            "annotations": {"place_status": "mentioned_only", "depicted_place": None},
        },
        {
            "artwork": {"id": 4},
            "annotations": {"place_status": "depicted", "depicted_place": {"name": "Paris"}},
        },
    ]
    heuristic_results = {
        1: [
            {"place": "Paris", "score_heuristique": 0.5},
            {"place": "France", "score_heuristique": 0.35},
        ],
        2: [{"place": "Rome", "score_heuristique": 0.2}],
        4: [{"place": "Paris", "score_heuristique": 0.6}],
    }

    comparison = compare_heuristic_places_with_annotations(
        records,
        heuristic_results,
        threshold=0.4,
    )

    assert comparison["real_depicted_artworks"] == 3
    assert comparison["heuristic_accepted_artworks"] == 2
    assert comparison["real_depicted_places"] == 3
    assert comparison["heuristic_accepted_places"] == 2
    assert comparison["artworks_with_discrepancy"] == [
        {"id": 2, "real_depicted": True, "accepted": False},
    ]


def test_filter_by_keywords_flattens_nested_keywords_and_keeps_all_matches():
    combined_texts = {42: "Street of Paris", 99: "A road outside"}
    keywords = {
        "urban": {"strong": ["street"], "weak": ["city"]},
        "location": {"strong": ["Paris"]},
    }

    result = filter_by_keywords(combined_texts, keywords)

    assert result == {
        "filtered_artworks": [
            {"id": 42, "matched_keywords": ["street", "Paris"]},
        ]
    }


def test_filter_by_keywords_matches_complete_words_not_substrings():
    result = filter_by_keywords({1: "A streetlamp"}, ["street"])

    assert result == {"filtered_artworks": []}


def test_binary_evaluation_treats_all_non_depicted_statuses_as_negative():
    records = [
        {"artwork": {"id": 1}, "annotations": {"place_status": "depicted"}},
        {"artwork": {"id": 2}, "annotations": {"place_status": "mentioned_only"}},
        {"artwork": {"id": 3}, "annotations": {"place_status": "none"}},
        {"artwork": {"id": 4}, "annotations": {"place_status": "undeterminable"}},
    ]
    heuristic_results = {
        1: [{"score_heuristique": 0.6}],
            2: [{"place": "Rome", "score_heuristique": 0.8}],
        3: [],
        4: [],
    }

    metrics = evaluate_binary_predictions(records, heuristic_results, threshold=0.5)

    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["true_negatives"] == 2
    assert metrics["false_negatives"] == 0
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1.0
    assert metrics["false_positive_ids"] == [2]
    assert metrics["false_negative_ids"] == []
    assert metrics["errors"][0]["predicted_places"] == [
        {"place": "Rome", "score_heuristique": 0.8}
    ]


def test_binary_evaluation_flags_rejected_places_near_threshold():
    records = [
        {"artwork": {"id": 1}, "annotations": {"place_status": "none"}},
    ]
    heuristic_results = {
        1: [{"place": "Lyon", "score_heuristique": 0.45}],
    }

    metrics = evaluate_binary_predictions(records, heuristic_results, threshold=0.5)

    assert metrics["true_negatives"] == 1
    assert metrics["flagged_ids"] == [1]
    assert metrics["flagged_predictions"] == [{
        "id": 1,
        "rejected_places": [{"place": "Lyon", "score_heuristique": 0.45}],
    }]


def test_logistic_model_trains_with_grouped_candidates_and_predicts_identity():
    records = []
    candidates_by_artwork = {}
    for index in range(1, 9):
        artwork_id = index if index % 2 else f"art-{index}"
        is_depicted = index <= 4
        place = "Paris" if is_depicted else "Rome"
        source_text = f"view of {place}" if is_depicted else f"born in {place}"
        records.append({
            "artwork": {"id": artwork_id},
            "annotations": {
                "place_status": "depicted" if is_depicted else "none",
                "depicted_place": {"name": "Paris"} if is_depicted else None,
            },
        })
        candidates_by_artwork[artwork_id] = [{
            "place": place,
            "score_ner": 0.9,
            "field": "title",
            "start": 8,
            "end": len(source_text),
            "source_text": source_text,
        }]

    table = build_training_table(records, candidates_by_artwork)
    model = train_logistic_model(table, n_splits=2)
    results = predict_with_model(records, candidates_by_artwork, model)
    identity = evaluate_place_identity(records, results)

    assert model["cv_metrics"]["true_positives"] == 4
    assert model["cv_metrics"]["true_negatives"] == 4
    assert model["cv_metrics"]["f1"] == 1.0
    assert identity["identity_accuracy"] == 1.0


def test_predict_artworks_file_writes_model_predictions(tmp_path, monkeypatch):
    import src.main as main_module

    input_path = tmp_path / "artworks.json"
    output_path = tmp_path / "nested" / "predictions.json"
    input_path.write_text(json.dumps({
        "unique_artworks": [
            {"artwork": {"id": 101, "title": "Paris"}},
            {"artwork": {"id": 102, "title": "Unknown"}},
        ]
    }), encoding="utf-8")
    monkeypatch.setattr(
        main_module.filt,
        "extract_all_candidates",
        lambda records, ner_model: {101: [], 102: []},
    )
    monkeypatch.setattr(
        main_module.filt,
        "predict_with_model",
        lambda records, candidates, model: {
            101: [{"place": "Paris", "score_heuristique": 0.9}],
            102: [{"place": "Lyon", "score_heuristique": 0.55}],
        },
    )
    model = {"threshold": 0.6}

    predictions = main_module.predict_artworks_file(
        input_path, output_path, model, ner_model=object()
    )

    assert predictions == [
        {
            "id": 101,
            "has_predicted_place": True,
            "predicted_places": [{"place": "Paris", "score": 0.9}],
            "flagged_places": [],
        },
        {
            "id": 102,
            "has_predicted_place": False,
            "predicted_places": [],
            "flagged_places": [{"place": "Lyon", "score": 0.55}],
        },
    ]
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "predictions": predictions,
    }


def test_main_requires_ner_module(monkeypatch):
    import src.main as main_module

    monkeypatch.setattr(main_module, "ner", None)

    import pytest

    with pytest.raises(ModuleNotFoundError, match="NER module is unavailable"):
        main_module.main()


def test_main_prints_logistic_model_metrics(monkeypatch, capsys):
    import src.main as main_module

    records = [
        {
            "artwork": {"id": 1},
            "annotations": {"place_status": "depicted", "depicted_place": {"name": "Paris"}},
        },
        {
            "artwork": {"id": 2},
            "annotations": {"place_status": "mentioned_only", "depicted_place": None},
        },
    ]
    optimization = {
        "C": 1.0,
        "threshold": 0.5,
        "cv_metrics": {
            "precision": 0.75,
            "recall": 0.6,
            "f1": 0.667,
            "true_positives": 3,
            "false_positives": 1,
            "true_negatives": 4,
            "false_negatives": 2,
        },
    }

    monkeypatch.setattr(main_module, "ner", SimpleNamespace(load_ner_model=lambda: object()))
    monkeypatch.setattr(
        main_module.Path,
        "open",
        lambda self, *args, **kwargs: __import__("io").StringIO(
            json.dumps({"unique_artworks": records})
        ),
    )
    monkeypatch.setattr(main_module.filt, "extract_all_candidates", lambda records, model: {})
    monkeypatch.setattr(main_module.filt, "build_training_table", lambda records, candidates: object())
    monkeypatch.setattr(main_module.filt, "train_logistic_model", lambda table: optimization)
    main_module.main()

    output = capsys.readouterr().out
    assert "classification par candidat" in output
    assert "Précision : 0.750" in output
    assert "Rappel : 0.600" in output
    assert "F1 : 0.667" in output
    assert "C de régularisation : 1.0" in output
    assert "Seuil d’acceptation : 0.500" in output