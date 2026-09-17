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