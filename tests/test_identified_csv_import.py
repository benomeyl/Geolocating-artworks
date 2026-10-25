from scripts.integrate_identified_csvs import (
    merge_records,
    parse_coordinates,
    parse_monuments,
    record_from_csv_row,
    select_described_rows,
)
from src.data.filtering_logreg import geographic_candidate_label, label_candidate


def test_csv_parsers_keep_multiple_names_and_latitude_longitude_order():
    assert parse_monuments("['Arc de Triomphe', 'Place de l’Étoile']") == [
        "Arc de Triomphe",
        "Place de l’Étoile",
    ]
    assert parse_coordinates("(45.434, 12.339)") == (45.434, 12.339)
    assert parse_coordinates("") is None


def test_csv_record_preserves_multiple_places_and_missing_coordinates():
    record = record_from_csv_row(
        "paris",
        "Paris",
        "Ile-de-France",
        "France",
        "paris_identified_data_ground_truth.csv",
        {
            "Image Name": "multi.jpg",
            "Monuments": "['Arc de Triomphe', 'Place de l’Étoile']",
            "title": "Paris view",
            "date": "1930",
        },
        2,
    )

    annotations = record["annotations"]
    assert record["artwork"]["id"] == "identified-csv:paris:multi.jpg"
    assert annotations["depicted_place"]["name"] == "Arc de Triomphe"
    assert [place["name"] for place in annotations["depicted_places"]] == [
        "Arc de Triomphe",
        "Place de l’Étoile",
    ]
    assert all(place["latitude"] is None for place in annotations["depicted_places"])


def test_merge_records_is_idempotent_and_keeps_existing_annotations():
    existing_record = {
        "artwork": {"id": "existing", "image_id": "existing.jpg"},
        "annotations": {"place_status": "none"},
    }
    imported = record_from_csv_row(
        "venice",
        "Venice",
        "Veneto",
        "Italy",
        "venice_identified_data_ground_truth.csv",
        {
            "Image Name": "venice.jpg",
            "Monument": "Rialto Bridge",
            "wd_id": "Q52505",
            "lat_lon": "(45.438008, 12.335639)",
        },
        2,
    )
    payload = {"unique_artworks": [existing_record]}

    merged, additions, skipped, replaced = merge_records(payload, [imported])
    rerun, rerun_additions, rerun_skipped, rerun_replaced = merge_records(merged, [imported])

    assert len(additions) == 1
    assert skipped == 0
    assert replaced == 0
    assert len(rerun_additions) == 1
    assert rerun_skipped == 0
    assert rerun_replaced == 1
    assert rerun["unique_artworks"][0] == existing_record
    assert len(rerun["unique_artworks"]) == 2


def test_selection_requires_description_and_prefers_distinct_wikidata_places():
    rows = [
        {"Image Name": "one.jpg", "wd_id": "Q1", "description": "caption one"},
        {"Image Name": "two.jpg", "wd_id": "Q1", "description": "caption two"},
        {"Image Name": "three.jpg", "wd_id": "Q2", "description": "caption three"},
        {"Image Name": "four.jpg", "wd_id": "Q3", "description": ""},
    ]

    selected = select_described_rows(rows, maximum=2)

    assert [row["Image Name"] for row in selected] == ["one.jpg", "three.jpg"]


def test_geographic_label_matches_any_depicted_place():
    record = {
        "annotations": {
            "place_status": "depicted",
            "depicted_place": {"name": "Paris", "precision": "exact_place"},
            "depicted_places": [
                {
                    "name": "Paris",
                    "precision": "exact_place",
                    "latitude": 48.85,
                    "longitude": 2.35,
                    "city": "Paris",
                    "region": "Ile-de-France",
                    "country": "France",
                },
                {
                    "name": "Delft Gate",
                    "precision": "exact_place",
                    "latitude": 52.004166666,
                    "longitude": 4.361111111,
                    "city": "Delft",
                    "region": "South Holland",
                    "country": "Netherlands",
                },
            ],
        }
    }
    candidate = {
        "name": "Schiedamse poort",
        "feature_code": "RSTN",
        "source": "GeoNames",
        "latitude": 52.0042,
        "longitude": 4.3611,
        "hierarchy": {
            "city": "Delft",
            "region": "South Holland",
            "country": "Netherlands",
        },
    }

    assert label_candidate(record, "Delft Gate") == 1
    assert geographic_candidate_label(record, candidate)["level"] == "exact_place"