from scripts.audit_geographic_annotations import audit_annotations


def test_audit_treats_city_as_not_applicable_for_region_precision():
    records = [
        {
            "artwork": {"id": "region-target"},
            "annotations": {
                "place_status": "depicted",
                "depicted_place": {
                    "name": "Iwo Jima",
                    "precision": "region",
                    "latitude": 24.75,
                    "longitude": 141.28,
                    "city": None,
                    "region": "Ogasawara",
                    "country": "Japan",
                },
            },
        },
        {
            "artwork": {"id": "place-with-missing-city"},
            "annotations": {
                "place_status": "depicted",
                "depicted_place": {
                    "name": "A station",
                    "precision": "exact_place",
                    "latitude": 46.52,
                    "longitude": 6.63,
                    "city": None,
                    "region": "Vaud",
                    "country": "Switzerland",
                },
            },
        },
    ]

    report = audit_annotations(records)

    assert report["missing_ids"]["city"] == ["place-with-missing-city"]
    assert report["not_applicable_ids"]["city"] == ["region-target"]