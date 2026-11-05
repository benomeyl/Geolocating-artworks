import csv

from scripts.relabel_city_in_title import COLUMNS, apply_review


def test_accepted_rows_become_city_level_depictions_and_keep_the_old_annotation(tmp_path):
    payload = {"unique_artworks": [
        {"artwork": {"id": 1, "title": "Perv's House, Chicago"}, "annotations": {"place_status": "none", "date": "2026-10-27"}},
        {"artwork": {"id": 2, "title": "Jewelled Bowl, Venice"}, "annotations": {"place_status": "none"}},
    ]}
    rows = [
        {"artwork_id": "1", "proposed_city": "Chicago", "proposed_country_code": "US",
         "proposed_latitude": "41.85", "proposed_longitude": "-87.65", "review_accept": "oui"},
        {"artwork_id": "2", "proposed_city": "Venice", "proposed_country_code": "IT",
         "proposed_latitude": "45.44", "proposed_longitude": "12.33", "review_accept": "non"},
    ]
    csv_path = tmp_path / "review.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in COLUMNS} for row in rows)

    assert apply_review(payload, csv_path, "2026-11-05") == 1

    accepted, refused = (record["annotations"] for record in payload["unique_artworks"])
    assert accepted["place_status"] == "depicted"
    assert accepted["depicted_place"]["precision"] == "city"
    assert accepted["depicted_place"]["latitude"] == 41.85
    assert accepted["previous_annotation"]["place_status"] == "none"
    assert refused == {"place_status": "none"}
