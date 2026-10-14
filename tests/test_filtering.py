from src.data.filtering import (
    compare_heuristic_places_with_annotations,
    evaluate_binary_predictions,
    filter_by_keywords,
    optimize_heuristic_parameters,
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


def test_optimizer_uses_stratified_out_of_fold_predictions():
    records = []
    candidates_by_artwork = {}
    for artwork_id in range(1, 9):
        is_depicted = artwork_id <= 4
        place = "Paris" if is_depicted else "Rome"
        source_text = f"view of {place}" if is_depicted else f"born in {place}"
        records.append({
            "artwork": {"id": artwork_id},
            "annotations": {
                "place_status": "depicted" if is_depicted else "none",
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

    result = optimize_heuristic_parameters(
        records,
        candidates_by_artwork=candidates_by_artwork,
        n_splits=2,
    )

    assert result["cv_metrics"]["true_positives"] == 4
    assert result["cv_metrics"]["true_negatives"] == 4
    assert result["cv_metrics"]["f1"] == 1.0
    assert len(result["fold_metrics"]) == 2
    assert result["false_positive_ids"] == []
    assert result["false_negative_ids"] == []


def test_parameter_grid_varies_each_field_weight():
    from src.data.filtering import _heuristic_parameter_grid

    field_weights = _heuristic_parameter_grid()["field_weights"]

    assert len(field_weights) > 3
    for field in ("title", "description", "short_description", "themes", "subjects", "styles"):
        assert len({weights[field] for weights in field_weights}) > 1


def test_optimizer_minimizes_false_positives_with_recall_constraint():
    records = []
    candidates_by_artwork = {}
    for artwork_id in range(1, 9):
        is_depicted = artwork_id <= 4
        records.append({
            "artwork": {"id": artwork_id},
            "annotations": {"place_status": "depicted" if is_depicted else "none"},
        })
        if is_depicted:
            candidates_by_artwork[artwork_id] = [{
                "place": "Paris",
                "score_ner": 0.9,
                "field": "title",
                "start": 0,
                "end": 5,
                "source_text": "Paris",
            }]
        elif artwork_id == 5:
            candidates_by_artwork[artwork_id] = [{
                "place": "Rome",
                "score_ner": 0.9,
                "field": "themes",
                "start": 0,
                "end": 4,
                "source_text": "Rome",
            }]

    result = optimize_heuristic_parameters(
        records,
        candidates_by_artwork=candidates_by_artwork,
        parameter_grid={
            "field_weights": [{"title": 0.8, "themes": 0.6}],
            "positive_context_bonus": [0.0],
            "negative_context_penalty": [0.0],
            "multi_field_bonus": [0.0],
            "acceptance_threshold": [0.5, 0.7],
            "ner_threshold": [0.5],
        },
        n_splits=2,
        inner_splits=2,
    )

    assert result["best_parameters"]["acceptance_threshold"] == 0.7
    assert result["cv_metrics"]["false_positives"] == 0


def test_main_requires_ner_module(monkeypatch):
    import src.main as main_module

    monkeypatch.setattr(main_module, "ner", None)

    import pytest

    with pytest.raises(ModuleNotFoundError, match="NER module is unavailable"):
        main_module.main()


def test_main_comparison_prints_summary(monkeypatch, capsys):
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
        "cv_metrics": {
            "precision": 0.75,
            "recall": 0.6,
            "f1": 0.667,
            "true_positives": 3,
            "false_positives": 1,
            "true_negatives": 4,
            "false_negatives": 2,
        },
        "best_parameters": {"acceptance_threshold": 0.5},
        "selection_objective": {
            "name": "minimize_false_positives",
            "minimum_recall": 0.8,
            "review_margin": 0.1,
        },
        "fold_metrics": [{
            "false_positive_ids": [2],
            "false_negative_ids": [1],
            "flagged_predictions": [{
                "id": 3,
                "rejected_places": [
                    {"place": "Lyon", "score_heuristique": 0.59}
                ],
            }],
            "errors": [{
                "id": 2,
                "actual": False,
                "predicted": True,
                "predicted_places": [
                    {"place": "Rome", "score_heuristique": 0.7}
                ],
            }],
        }],
    }

    monkeypatch.setattr(
        main_module,
        "ner",
        SimpleNamespace(load_ner_model=lambda: object()),
    )
    monkeypatch.setattr(
        main_module.filt,
        "optimize_heuristic_parameters",
        lambda records, model: optimization,
    )
    monkeypatch.setattr(
        main_module.Path,
        "open",
        lambda self, *args, **kwargs: __import__("io").StringIO(
            __import__("json").dumps({"unique_artworks": records})
        ),
    )
    main_module.main()

    output = capsys.readouterr().out
    assert "Précision : 0.750" in output
    assert "Rappel : 0.600" in output
    assert "F1 : 0.667" in output
    assert "Objectif : minimiser les faux positifs" in output
    assert "Rappel minimum visé : 80%" in output
    assert "2 : Rome (0.700)" in output
    assert "Faux négatifs : [1]" in output
    assert "3 : Lyon (0.590)" in output
    assert "acceptance_threshold: 0.5" in output