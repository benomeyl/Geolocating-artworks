import sqlite3

from src.data.filtering_logreg import (
    GEOGRAPHIC_FEATURE_COLUMNS,
    _artwork_target_cv_groups,
    build_geographic_label_table,
    evaluate_place_identity,
    evaluate_hierarchical_work_predictions,
    evaluate_hierarchical_grouped_cv,
    geographic_candidate_label,
    predict_with_hierarchical_model,
    train_hierarchical_model,
)
from src.geolocation.gazetter import NominatimClient, geocode_candidates, resolve_location


class FakeNominatim:
    def __init__(self):
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        return [{
            "name": "Example Place",
            "country_code": "FR",
            "latitude": 48.0,
            "longitude": 2.0,
            "feature_code": "city",
            "population": None,
            "source": "Nominatim",
            "attribution": "© OpenStreetMap contributors",
        }]


def test_geographic_candidate_label_falls_back_from_exact_place_to_city():
    record = {
        "annotations": {
            "place_status": "depicted",
            "depicted_place": {
                "name": "Schiedamse poort",
                "precision": "exact_place",
                "latitude": 52.004166666,
                "longitude": 4.361111111,
                "city": "Delft",
                "region": "South Holland",
                "country": "Netherlands",
            },
        }
    }
    city_candidate = {
        "name": "Delft",
        "feature_code": "PPL",
        "source": "GeoNames",
        "latitude": 52.010,
        "longitude": 4.360,
        "hierarchy": {
            "city": "Delft",
            "region": "South Holland",
            "country": "Netherlands",
        },
    }
    exact_candidate = {
        **city_candidate,
        "name": "Schiedamse poort",
        "feature_code": "RSTN",
        "latitude": 52.0042,
        "longitude": 4.3611,
    }

    assert geographic_candidate_label(record, city_candidate)["level"] == "city"
    assert geographic_candidate_label(record, exact_candidate)["level"] == "exact_place"


def test_geographic_candidate_label_does_not_force_unresolved_to_negative():
    record = {
        "annotations": {
            "place_status": "depicted",
            "depicted_place": {
                "name": "Delft",
                "precision": "city",
                "latitude": 52.0,
                "longitude": 4.0,
                "city": "Delft",
                "region": "South Holland",
                "country": "Netherlands",
            },
        }
    }

    label = geographic_candidate_label(record, None)

    assert label["level"] == "unevaluable"
    assert label["status"] == "unevaluable"


def test_geographic_candidate_label_marks_no_place_records_negative():
    record = {"annotations": {"place_status": "mentioned_only"}}

    assert geographic_candidate_label(record, None)["level"] == "none"


def test_geographic_label_table_resolves_candidate_and_emits_city_label():
    class DelftGeocoder:
        def search(self, query):
            return [{
                "name": "Delft",
                "country_code": "NL",
                "latitude": 52.010,
                "longitude": 4.360,
                "feature_code": "city",
                "source": "Nominatim",
                "hierarchy": {
                    "neighborhood": None,
                    "city": "Delft",
                    "region": "South Holland",
                    "country": "Netherlands",
                },
            }]

    records = [{
        "artwork": {"id": "delft-artwork"},
        "annotations": {
            "place_status": "depicted",
            "depicted_place": {
                "name": "Schiedamse poort",
                "precision": "exact_place",
                "latitude": 52.004166666,
                "longitude": 4.361111111,
                "city": "Delft",
                "region": "South Holland",
                "country": "Netherlands",
            },
        },
    }]
    candidates = {
        "delft-artwork": [{
            "place": "Delft",
            "score_ner": 0.9,
            "field": "title",
            "start": 0,
            "end": 5,
            "source_text": "Delft",
        }],
    }

    table = build_geographic_label_table(
        records,
        candidates,
        nominatim_client=DelftGeocoder(),
    )

    assert table.iloc[0]["label_level"] == "city"
    assert table.iloc[0]["label_status"] == "match"
    assert table.iloc[0]["candidate_geolocation"]["name"] == "Delft"


def test_hierarchical_prediction_falls_back_to_city_when_exact_is_below_target():
    class FixedProbabilityModel:
        def __init__(self, probability):
            self.probability = probability

        def predict_proba(self, features):
            return [[1 - self.probability, self.probability]]

    class ExactPlaceGeocoder:
        def search(self, query):
            return [{
                "name": "Schiedamse poort",
                "country_code": "NL",
                "latitude": 52.0042,
                "longitude": 4.3611,
                "feature_code": "yes",
                "place_class": "historic",
                "place_rank": 30,
                "importance": 0.6,
                "source": "Nominatim",
                "hierarchy": {
                    "neighborhood": None,
                    "city": "Delft",
                    "region": "South Holland",
                    "country": "Netherlands",
                },
            }]

    levels = {}
    for level in ("exact_place", "neighborhood", "city", "region", "country"):
        active = level in {"exact_place", "city"}
        levels[level] = {
            "status": "active" if active else "insufficient_positive_groups",
            "model": FixedProbabilityModel(0.99 if level == "exact_place" else 0.91),
            "threshold": 0.90,
            "oof_precision": 0.95 if level == "exact_place" else 0.90,
            "oof_support": 12,
            "target_precision": 0.95 if level == "exact_place" else 0.90,
        }
    model = {
        "feature_columns": GEOGRAPHIC_FEATURE_COLUMNS,
        "levels": levels,
        "specificity_order": ["exact_place", "neighborhood", "city", "region", "country"],
        "deployable_levels": ["city"],
    }
    records = [{"artwork": {"id": 1}}]
    candidates = {1: [{
        "place": "Schiedamse poort",
        "score_ner": 0.95,
        "field": "title",
        "start": 0,
        "end": 17,
        "source_text": "Schiedamse poort",
    }]}

    predictions = predict_with_hierarchical_model(
        records,
        candidates,
        model,
        nominatim_client=ExactPlaceGeocoder(),
    )

    decision = predictions[1]["prediction"]["decision"]
    assert predictions[1]["place_status_predicted"] is True
    assert decision["level"] == "city"
    assert decision["model_score"] == 0.91
    assert decision["confidence"] == 0.90


def test_hierarchical_training_disables_level_below_independent_group_floor():
    rows = []
    for index in range(12):
        positive = index < 4
        row = {column: 0.0 for column in GEOGRAPHIC_FEATURE_COLUMNS}
        row.update({
            "cv_group": f"group-{index}",
            "artwork_id": index,
            "label_status": "match" if positive else "negative",
            "label_level": "exact_place" if positive else "no_match",
        })
        rows.append(row)

    import pandas as pd

    model = train_hierarchical_model(
        pd.DataFrame(rows),
        minimum_positive_groups=5,
    )

    assert model["levels"]["exact_place"]["status"] == "insufficient_positive_groups"
    assert model["levels"]["exact_place"]["model"] is None


def test_hierarchical_training_selects_oof_thresholds_by_target_precision():
    import pandas as pd

    rows = []
    for index in range(24):
        positive = index < 12
        row = {column: 0.0 for column in GEOGRAPHIC_FEATURE_COLUMNS}
        row["field_title"] = int(positive)
        row.update({
            "cv_group": f"group-{index}",
            "artwork_id": index,
            "place_status": "depicted" if positive else "none",
            "label_status": "match" if positive else "negative",
            "label_level": "exact_place" if positive else "no_match",
            "candidate_level": "exact_place",
        })
        rows.append(row)

    model = train_hierarchical_model(
        pd.DataFrame(rows),
        regularization_values=(1.0,),
        minimum_positive_groups=5,
        minimum_prediction_groups=5,
        minimum_predictions=5,
    )

    exact = model["levels"]["exact_place"]
    assert exact["status"] == "active"
    assert exact["oof_precision"] >= exact["target_precision"]
    assert model["threshold_tuning_work_metrics"]["recall"] == 1.0
    assert model["minimum_work_recall_met"] is True


def test_hierarchical_work_metrics_count_artworks_without_candidates_as_fn():
    import pandas as pd

    table = pd.DataFrame([{
        "artwork_id": "has-candidate",
        "candidate_level": "exact_place",
        "label_status": "match",
    }])
    levels = {
        level: {"status": "insufficient_positive_groups", "threshold": None}
        for level in ("exact_place", "neighborhood", "city", "region", "country")
    }
    specificity = {
        level: index for index, level in enumerate(levels)
    }

    metrics = evaluate_hierarchical_work_predictions(
        table,
        {},
        levels,
        specificity,
        records={
            "has-candidate": "depicted",
            "no-candidate": "depicted",
        },
    )

    assert metrics["false_negatives"] == 2
    assert metrics["depicted_artwork_count"] == 2
    assert metrics["recall"] == 0


def test_city_ground_truth_does_not_validate_an_exact_place_prediction():
    import pandas as pd

    table = pd.DataFrame([{
        "artwork_id": "city-work",
        "candidate_level": "exact_place",
        "label_status": "match",
        "label_level": "city",
    }])
    levels = {
        "exact_place": {"status": "active", "threshold": 0.5},
        "neighborhood": {"status": "insufficient_positive_groups", "threshold": None},
        "city": {"status": "insufficient_positive_groups", "threshold": None},
        "region": {"status": "insufficient_positive_groups", "threshold": None},
        "country": {"status": "insufficient_positive_groups", "threshold": None},
    }
    specificity = {level: index for index, level in enumerate(levels)}

    metrics = evaluate_hierarchical_work_predictions(
        table,
        {0: {"exact_place": 0.9}},
        levels,
        specificity,
        records={"city-work": "depicted"},
    )

    assert metrics["true_positives"] == 0
    assert metrics["mislocated_count"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1


def test_nested_grouped_evaluation_measures_outer_work_predictions():
    import pandas as pd

    records = []
    rows = []
    for index in range(30):
        positive = index < 15
        artwork_id = f"art-{index}"
        records.append({
            "artwork": {"id": artwork_id},
            "annotations": {"place_status": "depicted" if positive else "none"},
        })
        row = {column: 0.0 for column in GEOGRAPHIC_FEATURE_COLUMNS}
        row["field_title"] = int(positive)
        row.update({
            "artwork_id": artwork_id,
            "cv_group": f"location-group-{index}",
            "place_status": "depicted" if positive else "none",
            "label_status": "match" if positive else "negative",
            "label_level": "exact_place" if positive else "none",
            "candidate_level": "exact_place",
        })
        rows.append(row)
    table = pd.DataFrame(rows)
    table.attrs["artwork_statuses"] = {
        record["artwork"]["id"]: record["annotations"]["place_status"]
        for record in records
    }
    table.attrs["artwork_cv_groups"] = {
        row["artwork_id"]: row["cv_group"] for row in rows
    }

    metrics = evaluate_hierarchical_grouped_cv(
        records,
        table,
        n_splits=3,
        training_options={
            "n_splits": 3,
            "regularization_values": (1.0,),
            "minimum_positive_groups": 3,
            "minimum_prediction_groups": 3,
            "minimum_predictions": 3,
        },
    )

    assert metrics["fold_count"] == 3
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["thresholds_tuned_inside_outer_training_folds"] is True


def test_cv_groups_join_artworks_that_share_a_target_entity():
    shared_place = {
        "name": "Dresden Cathedral",
        "precision": "exact_place",
        "wikidata_id": "Q12345",
        "latitude": 51.05349,
        "longitude": 13.73735,
    }
    records = [
        {
            "artwork": {"id": "photo-a"},
            "annotations": {
                "place_status": "depicted",
                "depicted_place": shared_place,
            },
        },
        {
            "artwork": {"id": "photo-b"},
            "annotations": {
                "place_status": "depicted",
                "depicted_place": {**shared_place, "name": "Kathedrale Dresden"},
            },
        },
        {
            "artwork": {"id": "other-place"},
            "annotations": {
                "place_status": "depicted",
                "depicted_place": {
                    **shared_place,
                    "name": "Frauenkirche",
                    "wikidata_id": "Q99999",
                    "latitude": 51.051,
                    "longitude": 13.741,
                },
            },
        },
    ]

    groups = _artwork_target_cv_groups(records)

    assert groups["photo-a"] == groups["photo-b"]
    assert groups["photo-a"] != groups["other-place"]


def test_resolve_location_uses_nominatim_only_when_geonames_has_no_match():
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE geonames (name TEXT, asciiname TEXT, latitude REAL, longitude REAL, "
        "country_code TEXT, feature_code TEXT, population INTEGER)"
    )
    connection.execute(
        "INSERT INTO geonames VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("Paris", "Paris", 48.85339, 2.34864, "FR", "PPLC", 2138551),
    )
    nominatim = FakeNominatim()

    geonames_match = resolve_location("Paris", connection, nominatim_client=nominatim)
    fallback_match = resolve_location("Example Place", connection, nominatim_client=nominatim)

    assert geonames_match["source"] == "GeoNames"
    assert fallback_match["source"] == "Nominatim"
    assert nominatim.queries == ["Example Place"]
    connection.close()


def test_place_evaluation_uses_nominatim_without_geonames_database():
    nominatim = FakeNominatim()
    records = [{
        "artwork": {"id": 2},
        "annotations": {
            "place_status": "depicted",
            "depicted_place": {
                "name": "Example Place",
                "city": "Example City",
                "latitude": 48.0,
                "longitude": 2.0,
            },
        },
    }]
    results = {2: [{"place": "Example Place", "score_heuristique": 0.9}]}

    metrics = evaluate_place_identity(
        records,
        results,
        nominatim_client=nominatim,
    )

    assert metrics["coordinate_candidate_coverage"] == 1
    assert metrics["city_candidate_coverage"] == 1
    assert nominatim.queries == ["Example Place", "Example City"]


def test_nominatim_client_caches_successful_responses(tmp_path):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return [{
                "name": "Example Place",
                "display_name": "Example Place, France",
                "lat": "48.0",
                "lon": "2.0",
                "type": "city",
                "class": "place",
                "place_rank": 16,
                "importance": 0.8,
                "osm_type": "relation",
                "osm_id": 123,
                "address": {
                    "country_code": "fr",
                    "country": "France",
                    "city": "Paris",
                    "suburb": "Latin Quarter",
                    "state": "Ile-de-France",
                },
            }]

    class FakeSession:
        calls = 0

        def get(self, *args, **kwargs):
            self.calls += 1
            return FakeResponse()

    session = FakeSession()
    client = NominatimClient(
        user_agent="test-suite/1.0",
        cache_path=tmp_path / "nominatim.sqlite",
        minimum_interval=0,
        session=session,
    )

    first = client.search("Example Place")
    second = client.search("Example Place")

    assert first == second
    assert first[0]["source"] == "Nominatim"
    assert first[0]["country_code"] == "FR"
    assert first[0]["hierarchy"] == {
        "neighborhood": "Latin Quarter",
        "city": "Paris",
        "region": "Ile-de-France",
        "country": "France",
        "country_code": "FR",
    }
    assert session.calls == 1


def test_geonames_candidate_gets_hierarchy_from_cached_nominatim_reverse(tmp_path):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "name": "Paris",
                "lat": "48.85339",
                "lon": "2.34864",
                "type": "city",
                "address": {
                    "country_code": "fr",
                    "country": "France",
                    "city": "Paris",
                    "state": "Ile-de-France",
                },
            }

    class FakeSession:
        calls = 0

        def get(self, *args, **kwargs):
            self.calls += 1
            return FakeResponse()

    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE geonames (name TEXT, asciiname TEXT, latitude REAL, longitude REAL, "
        "country_code TEXT, feature_code TEXT, population INTEGER)"
    )
    connection.execute(
        "INSERT INTO geonames VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("Paris", "Paris", 48.85339, 2.34864, "FR", "PPLC", 2138551),
    )
    session = FakeSession()
    client = NominatimClient(
        cache_path=tmp_path / "reverse-cache.sqlite",
        minimum_interval=0,
        session=session,
    )
    first = {1: [{"place": "Paris", "score_heuristique": 0.9}]}
    second = {2: [{"place": "Paris", "score_heuristique": 0.8}]}

    geocode_candidates(first, conn=connection, nominatim_client=client)
    geocode_candidates(second, conn=connection, nominatim_client=client)

    assert first[1][0]["geolocation"]["source"] == "GeoNames"
    assert first[1][0]["geolocation"]["hierarchy"]["city"] == "Paris"
    assert first[1][0]["geolocation"]["hierarchy_source"] == "Nominatim"
    assert second[2][0]["geolocation"]["hierarchy"]["region"] == "Ile-de-France"
    assert session.calls == 1
    connection.close()


def test_geocode_candidates_attaches_fallback_result_to_top_k():
    nominatim = FakeNominatim()
    results = {
        1: [{"place": "Example Place", "score_heuristique": 0.9}],
        3: [
            {"place": "Example Place", "score_heuristique": 0.8},
            {"place": "Other Place", "score_heuristique": 0.7},
        ],
        2: [],
    }

    geocode_candidates(results, nominatim_client=nominatim, limit_per_artwork=1)

    assert results[1][0]["geolocation"]["source"] == "Nominatim"
    assert results[1][0]["geolocation"]["latitude"] == 48.0
    assert results[3][0]["geolocation"]["source"] == "Nominatim"
    assert "geolocation" not in results[3][1]
    assert results[2] == []
    assert nominatim.queries == ["Example Place"]


def test_place_evaluation_recognizes_city_when_exact_place_is_not_found():
    connection = sqlite3.connect(":memory:")
    connection.execute(
        """
        CREATE TABLE geonames (
            name TEXT,
            asciiname TEXT,
            latitude REAL,
            longitude REAL,
            country_code TEXT,
            feature_code TEXT,
            population INTEGER
        )
        """
    )
    connection.execute(
        "INSERT INTO geonames VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("Paris", "Paris", 48.85339, 2.34864, "FR", "PPLC", 2138551),
    )
    records = [{
        "artwork": {"id": 1},
        "annotations": {
            "place_status": "depicted",
            "depicted_place": {
                "name": "Notre-Dame de Paris",
                "city": "Paris",
                "latitude": 48.853,
                "longitude": 2.349,
            },
        },
    }]
    results = {
        1: [{
            "place": "Paris suburb",
            "score_heuristique": 0.9,
            "latitude": 48.98,
            "longitude": 2.349,
        }],
    }

    metrics = evaluate_place_identity(
        records,
        results,
        gazetteer_conn=connection,
        coordinate_tolerance_km=5,
        city_tolerance_km=30,
    )

    assert metrics["identity_accuracy"] == 0
    assert metrics["wrong_identity"][0]["distance_km"] is not None
    assert metrics["wrong_identity"][0]["distance_km"] > 0
    assert metrics["coordinate_accuracy"] == 0
    assert metrics["coordinate_candidate_coverage"] == 1
    assert metrics["city_proximity_accuracy"] == 1
    assert metrics["city_candidate_coverage"] == 1
    assert metrics["city_evaluable"] == 1

    connection.close()