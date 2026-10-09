from src.data.filtering import compare_heuristic_places_with_annotations, filter_by_keywords


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


def test_main_accepts_dict_corpus(monkeypatch):
    import src.main as main_module

    monkeypatch.setattr(
        main_module.utils,
        "read_all_json_files_and_combine_text_fields",
        lambda path: {42: "Street in Paris"},
    )
    monkeypatch.setattr(
        main_module.filt,
        "filter_by_keywords",
        lambda texts_by_id, keywords: {"filtered_artworks": [{"id": 42}]},
    )
    monkeypatch.setattr(main_module.ner, "load_ner_model", lambda: lambda text: [])
    monkeypatch.setattr(
        main_module.ner,
        "extract_named_entities",
        lambda texts, model: {"status": "ok", "texts": list(texts)},
    )
    monkeypatch.setattr(main_module.ner, "display_named_entities", lambda _: None)

    main_module.main()

    assert True


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
    heuristic_results = {
        1: [{"place": "Paris", "score_heuristique": 0.6}],
        2: [{"place": "Paris", "score_heuristique": 0.2}],
    }

    monkeypatch.setattr(
        main_module.Path,
        "open",
        lambda self, *args, **kwargs: __import__("io").StringIO(
            __import__("json").dumps({"unique_artworks": records})
        ),
    )
    monkeypatch.setattr(main_module.ner, "load_ner_model", lambda: lambda text: [])
    monkeypatch.setattr(
        main_module.filt,
        "evaluate_artwork",
        lambda artwork, model: heuristic_results[artwork["id"]],
    )

    main_module.main()

    output = capsys.readouterr().out
    assert "Œuvres avec un lieu réellement représenté : 1" in output
    assert "Œuvres acceptées par l’heuristique : 1" in output
    assert "Lieux réellement représentés : 1" in output
    assert "Lieux acceptés par l’heuristique : 1" in output