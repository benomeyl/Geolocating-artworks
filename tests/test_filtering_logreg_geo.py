import sqlite3

from src.data.filtering_logreg import (
    build_geographic_label_table,
    evaluate_place_identity,
    geographic_candidate_label,
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