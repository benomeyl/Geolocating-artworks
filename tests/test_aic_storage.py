import json

from src.api.aic import save_artworks_response, search_multiple_queries


def test_search_multiple_queries_writes_one_json_file_with_queries_and_unique_artworks(tmp_path, monkeypatch):
    def fake_search_all_artworks(query, max_pages=None, limit=10):
        if query == "Paris":
            return [
                {"id": 1, "title": "Paris art"},
                {"id": 1, "title": "Paris art"},
            ]
        if query == "Chicago":
            return [
                {"id": 1, "title": "Paris art"},
                {"id": 2, "title": "Chicago art"},
            ]
        return []

    monkeypatch.setattr("src.api.aic.search_all_artworks", fake_search_all_artworks)

    unique_artworks = search_multiple_queries(
        ["Paris", "Chicago"],
        limit=5,
        output_dir=tmp_path,
    )

    assert unique_artworks[1]["artwork"] == {"id": 1, "title": "Paris art"}
    assert unique_artworks[1]["matched_queries"] == ["Paris", "Chicago"]
    assert unique_artworks[2]["artwork"] == {"id": 2, "title": "Chicago art"}
    assert unique_artworks[2]["matched_queries"] == ["Chicago"]

    files = list(tmp_path.glob("search_multiple_queries_*.json"))
    assert len(files) == 1

    payload = json.loads(files[0].read_text())
    assert payload["search_queries"] == ["Paris", "Chicago"]
    assert payload["unique_artworks"][0]["artwork"] == {"id": 1, "title": "Paris art"}
    assert payload["unique_artworks"][0]["matched_queries"] == ["Paris", "Chicago"]
    assert payload["unique_artworks"][1]["artwork"] == {"id": 2, "title": "Chicago art"}
    assert payload["unique_artworks"][1]["matched_queries"] == ["Chicago"]


def test_save_artworks_response_supports_custom_payload_output(tmp_path):
    payload = {
        "search_queries": ["Paris"],
        "unique_artworks": [{
            "artwork": {"id": 1, "title": "Painting"},
            "matched_queries": ["Paris"],
        }],
    }

    output_path = save_artworks_response(
        payload,
        search_queries=["Paris"],
        output_dir=tmp_path,
    )

    assert output_path.exists()
    assert output_path.parent == tmp_path
    stored = json.loads(output_path.read_text())
    assert stored["search_queries"] == ["Paris"]
    assert stored["unique_artworks"][0]["artwork"] == {"id": 1, "title": "Painting"}
    assert stored["unique_artworks"][0]["matched_queries"] == ["Paris"]
