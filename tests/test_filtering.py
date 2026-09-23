from src.data.filtering import filter_by_keywords


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
    monkeypatch.setattr(main_module.ner, "load_ner_model", lambda: object())
    monkeypatch.setattr(
        main_module.ner,
        "extract_named_entities",
        lambda texts, model: {"status": "ok", "texts": list(texts)},
    )
    monkeypatch.setattr(main_module.ner, "display_named_entities", lambda _: None)

    main_module.main()

    assert True