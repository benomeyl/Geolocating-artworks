import json

from src.utils import read_artworks_from_json


def test_read_artworks_from_json_unwraps_records_and_indexes_by_id(tmp_path):
    payload = {
        "unique_artworks": [
            {"artwork": {"id": 1, "title": "Painting"}, "matched_queries": ["Paris"]},
            {"artwork": {"id": 2, "title": "Sculpture"}, "matched_queries": ["Rome"]},
        ]
    }
    json_file = tmp_path / "artworks.json"
    json_file.write_text(json.dumps(payload), encoding="utf-8")

    assert read_artworks_from_json(json_file) == {
        1: {"id": 1, "title": "Painting"},
        2: {"id": 2, "title": "Sculpture"},
    }