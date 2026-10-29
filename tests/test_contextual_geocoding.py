import sqlite3

from src.data.filtering_logreg import _name_similarity
from src.geolocation.gazetter import location_alternatives, resolve_artwork_places, resolve_location


def _geonames(rows, with_alternate_names=False):
    connection = sqlite3.connect(":memory:")
    columns = "name TEXT, asciiname TEXT, latitude REAL, longitude REAL, country_code TEXT, feature_code TEXT, population INTEGER"
    if with_alternate_names:
        columns += ", alternatenames TEXT"
    connection.execute(f"CREATE TABLE geonames ({columns})")
    placeholders = ",".join("?" for _ in rows[0]) if rows else ""
    for row in rows:
        connection.execute(f"INSERT INTO geonames VALUES ({placeholders})", row)
    return connection


def _nominatim_location(name, latitude, longitude, country_code, region=None):
    return {
        "name": name,
        "latitude": latitude,
        "longitude": longitude,
        "country_code": country_code,
        "feature_code": "administrative",
        "place_rank": 12,
        "source": "Nominatim",
        "hierarchy": {"region": region, "country_code": country_code},
    }


class FakeNominatim:
    def __init__(self, results):
        self.results = results
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        return self.results.get(query, [])


def _mention(text, name, field="title"):
    start = text.index(name)
    return name, [{"field": field, "start": start, "end": start + len(name), "source_text": text}]


def test_untrusted_geonames_feature_falls_back_to_nominatim():
    connection = _geonames([("Louvre", "Louvre", 39.65, 19.84, "GR", "HTL", 0)])
    nominatim = FakeNominatim({"Louvre": [_nominatim_location("Musée du Louvre", 48.861, 2.336, "FR")]})

    location = resolve_location("Louvre", connection, nominatim_client=nominatim)

    assert location["source"] == "Nominatim"
    assert location["country_code"] == "FR"


def test_untrusted_geonames_feature_is_kept_when_nominatim_finds_nothing():
    connection = _geonames([("Schiedamse Poort", "Schiedamse Poort", 52.004, 4.361, "NL", "RSTN", 0)])

    location = resolve_location("Schiedamse Poort", connection, nominatim_client=FakeNominatim({}))

    assert location["source"] == "GeoNames"


def test_country_found_by_alternate_name_before_homonyms():
    connection = _geonames(
        [
            ("Spain", "Spain", 30.86, -83.54, "US", "PPL", 0, ""),
            ("Kingdom of Spain", "Kingdom of Spain", 40.0, -4.0, "ES", "PCLI", 46723749, "Espagne,Spain,ES"),
        ],
        with_alternate_names=True,
    )

    alternatives = location_alternatives("Spain", connection)

    assert alternatives[0]["country_code"] == "ES"
    assert alternatives[0]["feature_code"] == "PCLI"


def test_places_mentioned_together_choose_agreeing_alternatives():
    text = "Earl and Carol Davis, Cornland, Logan County"
    nominatim = FakeNominatim({
        "Logan County": [
            _nominatim_location("Logan County", 35.92, -97.44, "US", "Oklahoma"),
            _nominatim_location("Logan County", 40.12, -89.37, "US", "Illinois"),
        ],
        "Cornland": [_nominatim_location("Cornland", 39.94, -89.40, "US", "Illinois")],
    })

    resolved = resolve_artwork_places(
        [_mention(text, "Cornland"), _mention(text, "Logan County")], None, nominatim
    )

    assert resolved["Logan County"]["hierarchy"]["region"] == "Illinois"
    assert resolved["Logan County"]["context_conflict"] is False


def test_contextual_search_then_conflict_flag():
    text = "Crater Range San Salvador Peru"
    peru = {**_nominatim_location("Perú", -9.0, -75.0, "PE"), "feature_code": "PCLI"}
    nominatim = FakeNominatim({
        "Peru": [peru],
        "Crater Range": [_nominatim_location("Crater Range", 32.6, -112.9, "US", "Arizona")],
        "San Salvador": [_nominatim_location("San Salvador", 13.7, -89.2, "SV")],
        "San Salvador, Peru": [_nominatim_location("San Salvador", -12.9, -72.1, "PE")],
    })

    resolved = resolve_artwork_places(
        [_mention(text, "Crater Range"), _mention(text, "San Salvador"), _mention(text, "Peru")],
        None,
        nominatim,
    )

    assert resolved["San Salvador"]["country_code"] == "PE"
    assert resolved["San Salvador"]["context_conflict"] is False
    assert resolved["Crater Range"]["country_code"] == "US"
    assert resolved["Crater Range"]["context_conflict"] is True


def test_distant_mentions_do_not_constrain_each_other():
    text = "Born in Spain, " + "x" * 120 + " he painted London"
    nominatim = FakeNominatim({
        "Spain": [{**_nominatim_location("España", 40.0, -4.0, "ES"), "feature_code": "PCLI"}],
        "London": [_nominatim_location("London", 51.5, -0.12, "GB")],
    })

    resolved = resolve_artwork_places([_mention(text, "Spain"), _mention(text, "London")], None, nominatim)

    assert resolved["London"]["country_code"] == "GB"
    assert resolved["London"]["context_conflict"] is False
    assert "London, Spain" not in nominatim.queries


def test_name_similarity_flags_unrelated_geocodes():
    assert _name_similarity("Rouen Cathedral", {"name": "Cathédrale de Rouen"}) == 1.0
    assert _name_similarity("Louvre", {"name": "Γουβιά"}) == 0.0
    assert 0.5 < _name_similarity("Venice", {"name": "Venezia"}) < 1.0
    assert _name_similarity("Paris", None) == 0.0


def test_misread_common_words_do_not_give_context():
    text = "Surf, Point Lobos"
    surf = {**_nominatim_location("Surfside Beach", 28.9, -95.3, "US", "Texas"), "place_rank": 30}
    nominatim = FakeNominatim({
        "Surf": [surf],
        "Point Lobos": [
            _nominatim_location("Point Lobos", 36.52, -121.95, "US", "California"),
            _nominatim_location("Point Lobos", 28.95, -95.30, "US", "Texas"),
        ],
    })

    resolved = resolve_artwork_places([_mention(text, "Surf"), _mention(text, "Point Lobos")], None, nominatim)

    assert resolved["Point Lobos"]["hierarchy"]["region"] == "California"


def test_nominatim_results_confirmed_by_geonames_come_first():
    connection = _geonames([("Point Lobos", "Point Lobos", 36.52, -121.95, "US", "CAPE", 0)])
    nominatim = FakeNominatim({"Point Lobos": [
        _nominatim_location("Point Lobos", 28.95, -95.30, "US", "Texas"),
        _nominatim_location("Point Lobos", 36.52, -121.94, "US", "California"),
    ]})

    location = resolve_location("Point Lobos", connection, nominatim_client=nominatim)

    assert location["hierarchy"]["region"] == "California"


def test_small_hamlet_does_not_hide_a_famous_place():
    connection = _geonames([("Puerta del Sol", "Puerta del Sol", 29.5, -110.3, "MX", "PPL", 184)])
    nominatim = FakeNominatim({"Puerta del Sol": [_nominatim_location("Puerta del Sol", 40.417, -3.703, "ES")]})

    assert resolve_location("Puerta del Sol", connection, nominatim_client=nominatim)["country_code"] == "ES"


def test_country_is_not_revised_by_smaller_neighbours():
    text = "San Salvador Peru"
    peru = {**_nominatim_location("Perú", -9.0, -75.0, "PE"), "feature_code": "PCLI"}
    nominatim = FakeNominatim({
        "Peru": [peru],
        "San Salvador": [_nominatim_location("San Salvador", 13.7, -89.2, "SV")],
    })

    resolved = resolve_artwork_places([_mention(text, "San Salvador"), _mention(text, "Peru")], None, nominatim)

    assert resolved["Peru"]["country_code"] == "PE"
    assert resolved["Peru"]["context_conflict"] is False
