import json
from pathlib import Path

from src.api.aic import save_artworks_response


def test_save_artworks_response_writes_json_file(tmp_path):
    payload = {
        "data": [
            {"id": 1, "title": "Painting"}
        ],
        "pagination": {"total": 1}
    }

    output_path = save_artworks_response(
        payload,
        query="Paris",
        limit=5,
        output_dir=tmp_path,
    )

    assert output_path.exists()
    assert output_path.parent == tmp_path
    assert json.loads(output_path.read_text())["data"][0]["title"] == "Painting"
