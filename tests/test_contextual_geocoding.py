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
    def __init__(self, results, within=None):
        self.results = results
        self.within = within or {}
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        return self.results.get(query, [])

    def search_within(self, query, country_code, viewbox=None):
        self.queries.append((query, country_code, viewbox is not None))
        return self.within.get((query, country_code), [])


def _city(name, latitude, longitude, country_code, importance):
    return {**_nominatim_location(name, latitude, longitude, country_code), "place_rank": 16, "importance": importance}


def _monument(name, latitude, longitude, country_code, importance=0.3):
    return {**_nominatim_location(name, latitude, longitude, country_code), "place_rank": 30, "importance": importance}


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


def test_distant_homonyms_are_counted_with_their_importance_gap():
    text = "London"
    nominatim = FakeNominatim({"London": [
        {**_nominatim_location("London", 51.5, -0.12, "GB"), "importance": 0.9},
        {**_nominatim_location("London", 42.98, -81.25, "CA"), "importance": 0.6},
        {**_nominatim_location("City of London", 51.51, -0.09, "GB"), "importance": 0.5},
    ]})

    resolved = resolve_artwork_places([_mention(text, "London")], None, nominatim)

    assert resolved["London"]["homonym_count"] == 1
    assert abs(resolved["London"]["importance_gap"] - 0.3) < 1e-9


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


def _important(location, importance, **extra):
    return {**location, "importance": importance, **extra}


def test_famous_nominatim_place_beats_obscure_homonyms_of_both_sources():
    # Corcovado: a 1,527-inhabitant town in Puerto Rico (trusted GeoNames) and the mountain in Rio.
    connection = _geonames([("Corcovado", "Corcovado", 18.06, -66.4, "PR", "PPL", 1527)])
    nominatim = FakeNominatim({
        "Corcovado": [_important(_nominatim_location("Corcovado", -22.95, -43.21, "BR"), 0.45)],
        "Patmos": [
            _important(_nominatim_location("Patmos", 33.3, -93.5, "US"), 0.36),
            _important(_nominatim_location("Πάτμος", 37.32, 26.55, "GR"), 0.54),
        ],
    })

    assert resolve_location("Corcovado", connection, nominatim_client=nominatim)["country_code"] == "BR"
    assert resolve_location("Patmos", connection, nominatim_client=nominatim)["country_code"] == "GR"


def test_geonames_entry_takes_the_importance_of_the_same_nominatim_place():
    connection = _geonames([
        ("Paris", "Paris", 48.86, 2.35, "FR", "ADM3", 4380654),
        ("Paris", "Paris", 48.85, 2.35, "FR", "PPLC", 2138551),
        ("Paris", "Paris", 33.66, -95.56, "US", "PPL", 24171),
    ])
    nominatim = FakeNominatim({"Paris": [
        _important(_nominatim_location("Paris", 48.85, 2.34, "FR"), 0.9),
        _important(_nominatim_location("Paris", 33.66, -95.55, "US"), 0.53),
    ]})

    alternatives = location_alternatives("Paris", connection, nominatim)

    assert [(location["source"], location["feature_code"]) for location in alternatives] == [
        ("GeoNames", "PPLC"), ("GeoNames", "ADM3"), ("GeoNames", "PPL"),
    ]
    assert alternatives[0]["importance"] == 0.9


def test_shops_and_stops_are_not_places():
    nominatim = FakeNominatim({
        "Microcosm": [_nominatim_location("Microcosm", 53.9, 27.56, "BY") | {"place_class": "shop", "feature_code": "copyshop"}],
        "Archives": [
            _nominatim_location("Archives", 38.89, -77.02, "US") | {"feature_code": "subway_entrance"},
            _important(_nominatim_location("National Archives", 38.89, -77.02, "US"), 0.72, feature_code="museum"),
        ],
    })

    assert location_alternatives("Microcosm", None, nominatim) == []
    assert [location["name"] for location in location_alternatives("Archives", None, nominatim)] == ["National Archives"]


def test_prominent_place_is_kept_against_a_foreign_context():
    text = "Rome, France"
    france = {**_nominatim_location("France", 46.6, 1.9, "FR"), "feature_code": "PCLI"}
    nominatim = FakeNominatim({
        "France": [france],
        "Rome": [_important(_nominatim_location("Roma", 41.89, 12.48, "IT"), 0.86)],
        "Rome, France": [_nominatim_location("Rome", 48.88, 2.32, "FR") | {"feature_code": "station"}],
    })

    resolved = resolve_artwork_places([_mention(text, "Rome"), _mention(text, "France")], None, nominatim)

    assert resolved["Rome"]["country_code"] == "IT"
    assert resolved["Rome"]["context_conflict"] is True
    assert "Rome, France" not in nominatim.queries


def test_precise_place_is_searched_inside_its_city():
    text = "Palazzo Vecchio, Florence"
    nominatim = FakeNominatim(
        {
            "Florence": [_city("Firenze", 43.77, 11.25, "IT", 0.8), _city("Florence", 34.8, -87.68, "US", 0.5)],
            "Palazzo Vecchio": [_monument("Palazzo Vecchio", 40.0, -75.0, "US")],
        },
        within={("Palazzo Vecchio", "IT"): [_monument("Palazzo Vecchio", 43.769, 11.256, "IT")]},
    )

    resolved = resolve_artwork_places([_mention(text, "Palazzo Vecchio"), _mention(text, "Florence")], None, nominatim)

    assert resolved["Florence"]["country_code"] == "IT"
    assert resolved["Palazzo Vecchio"]["country_code"] == "IT"
    assert resolved["Palazzo Vecchio"]["resolution"] == "searched_in_anchor"


def test_a_city_named_far_away_in_the_description_still_constrains_the_title():
    description = "x" * 200 + " photographed in Chicago"
    title_mention = _mention("Monroe Street", "Monroe Street")
    city_mention = _mention(description, "Chicago", "description")
    nominatim = FakeNominatim({
        "Chicago": [_city("Chicago", 41.88, -87.63, "US", 0.8)],
        "Monroe Street": [
            _monument("Monroe Street", 42.33, -83.04, "US"),
            _monument("Monroe Street", 41.88, -87.63, "US"),
        ],
    })

    resolved = resolve_artwork_places([title_mention, city_mention], None, nominatim)

    assert resolved["Monroe Street"]["latitude"] == 41.88
    assert resolved["Monroe Street"]["resolution"] == "in_anchor"


def test_a_precise_place_not_found_in_its_area_gets_no_location():
    text = "Le Nys, Amsterdam"
    nominatim = FakeNominatim({
        "Amsterdam": [_city("Amsterdam", 52.37, 4.9, "NL", 0.8)],
        "Le Nys": [_monument("Carrière de Nys", 49.5, 3.6, "FR")],
    })

    resolved = resolve_artwork_places([_mention(text, "Le Nys"), _mention(text, "Amsterdam")], None, nominatim)

    assert resolved["Le Nys"] is None
    assert resolved["Amsterdam"]["country_code"] == "NL"
    assert ("Le Nys", "NL", True) in nominatim.queries


def test_a_famous_place_outside_the_anchors_is_kept_with_a_conflict():
    text = "The Louvre, New York"
    nominatim = FakeNominatim({
        "New York": [_city("New York", 40.71, -74.0, "US", 0.9)],
        "Louvre": [_monument("Musée du Louvre", 48.86, 2.34, "FR", importance=0.8)],
    })

    resolved = resolve_artwork_places([_mention(text, "Louvre"), _mention(text, "New York")], None, nominatim)

    assert resolved["Louvre"]["country_code"] == "FR"
    assert resolved["Louvre"]["context_conflict"] is True


def test_generic_names_are_only_searched_inside_an_anchor():
    nominatim = FakeNominatim(
        {"Park": [_monument("Park", 10.0, 10.0, "NG")], "Versailles": [_city("Versailles", 48.8, 2.13, "FR", 0.7)]},
        within={("Park", "FR"): [_monument("Park of Versailles", 48.81, 2.11, "FR")]},
    )

    alone = resolve_artwork_places([_mention("Park", "Park")], None, nominatim)
    text = "Park at Versailles"
    with_anchor = resolve_artwork_places([_mention(text, "Park"), _mention(text, "Versailles")], None, nominatim)

    assert alone["Park"] is None
    assert with_anchor["Park"]["name"] == "Park of Versailles"


def test_a_place_of_origin_is_not_an_anchor():
    text = "View of the Pont Neuf. Printed in London"
    nominatim = FakeNominatim({
        "London": [_city("London", 51.5, -0.12, "GB", 0.9)],
        "Pont Neuf": [_monument("Pont Neuf", 48.857, 2.341, "FR", importance=0.5)],
    })

    resolved = resolve_artwork_places([_mention(text, "Pont Neuf"), _mention(text, "London")], None, nominatim)

    assert resolved["Pont Neuf"]["country_code"] == "FR"
    assert resolved["Pont Neuf"]["resolution"] == "global"


def test_an_anchor_without_country_code_is_not_searched():
    text = "Le Nys, Amsterdam"
    nominatim = FakeNominatim({
        "Amsterdam": [{**_city("Amsterdam", 52.37, 4.9, "NL", 0.8), "country_code": None}],
        "Le Nys": [_monument("Carrière de Nys", 49.5, 3.6, "FR")],
    })

    resolved = resolve_artwork_places([_mention(text, "Le Nys"), _mention(text, "Amsterdam")], None, nominatim)

    assert resolved["Le Nys"] is None


def test_results_found_in_an_area_must_carry_the_mention_words():
    text = "Cardiff Castle, Cardiff"
    nominatim = FakeNominatim(
        {"Cardiff": [_city("Cardiff", 51.48, -3.18, "GB", 0.7)], "Cardiff Castle": []},
        within={("Cardiff Castle", "GB"): [_monument("Cardiff Avenue", 51.49, -3.17, "GB")]},
    )

    resolved = resolve_artwork_places([_mention(text, "Cardiff Castle"), _mention(text, "Cardiff")], None, nominatim)

    assert resolved["Cardiff Castle"] is None


def test_a_prominent_city_is_not_dragged_by_a_city_level_homonym():
    text = "The Red House, Berlin"
    nominatim = FakeNominatim({
        "Red House": [_city("Town of Red House", 37.1, -78.8, "US", 0.3)],
        "Berlin": [_city("Berlin", 52.52, 13.4, "DE", 0.85), _city("Berlin", 37.0, -78.0, "US", 0.4)],
    })

    resolved = resolve_artwork_places([_mention(text, "Red House"), _mention(text, "Berlin")], None, nominatim)

    assert resolved["Berlin"]["country_code"] == "DE"


def test_a_distant_anchor_keeps_the_best_guess_with_a_conflict():
    description = "x" * 200 + " The artist lived in Berlin."
    nominatim = FakeNominatim({
        "Berlin": [_city("Berlin", 52.52, 13.4, "DE", 0.85)],
        "Nidden": [_city("Nida", 55.3, 21.0, "LT", 0.4) | {"place_rank": 19}],
    })
    # "lived in" is an origin cue: use a neutral sentence for the anchor.
    description = description.replace("The artist lived in Berlin.", "Berlin and its painters.")

    resolved = resolve_artwork_places(
        [_mention("Nidden", "Nidden", "description"), _mention(description, "Berlin", "description")],
        None,
        nominatim,
    )

    assert resolved["Nidden"]["country_code"] == "LT"
    assert resolved["Nidden"]["context_conflict"] is True


def test_generic_names_are_never_anchors():
    text = "The Entrance to the Forest by the Bridge, Kew"
    nominatim = FakeNominatim({
        "Forest": [_city("Forest County", 45.6, -88.8, "US", 0.4)],
        "Kew": [_city("Kew", 51.48, -0.29, "GB", 0.5) | {"place_rank": 19}],
    })

    resolved = resolve_artwork_places([_mention(text, "Forest"), _mention(text, "Kew")], None, nominatim)

    assert resolved["Forest"] is None
    assert resolved["Kew"]["country_code"] == "GB"


def test_a_town_named_like_a_monument_is_searched_inside_the_city_that_follows():
    text = "Notre Dame, Paris"
    cathedral = _monument("Cathédrale Notre-Dame de Paris", 48.853, 2.350, "FR", importance=0.61)
    nominatim = FakeNominatim({
        "Notre Dame": [_city("Notre Dame", 41.70, -86.24, "US", 0.63), cathedral],
        "Paris": [_city("Paris", 48.853, 2.349, "FR", 0.9)],
    })

    resolved = resolve_artwork_places([_mention(text, "Notre Dame"), _mention(text, "Paris")], None, nominatim)

    assert resolved["Notre Dame"]["name"] == "Cathédrale Notre-Dame de Paris"
    assert resolved["Notre Dame"]["resolution"] == "in_anchor"
    assert resolved["Paris"]["country_code"] == "FR"


def test_a_town_inside_the_place_that_follows_stays_an_anchor():
    text = "Moonrise, Hernandez, New Mexico"
    new_mexico = {**_nominatim_location("New Mexico", 34.4, -106.1, "US"), "place_rank": 8, "importance": 0.7}
    nominatim = FakeNominatim({
        "Hernandez": [_city("Hernandez", 36.06, -106.12, "US", 0.3)],
        "New Mexico": [new_mexico],
    })

    resolved = resolve_artwork_places(
        [_mention(text, "Hernandez"), _mention(text, "New Mexico")], None, nominatim
    )

    assert resolved["Hernandez"]["resolution"] == "anchor"


def test_a_famous_homonym_under_another_name_is_not_kept_next_to_its_city():
    text = "The Clock Tower, Venice"
    nominatim = FakeNominatim({
        "Clock Tower": [_monument("Ghanta Ghar", 30.32, 78.04, "IN", importance=0.63)],
        "Venice": [_city("Venice", 45.44, 12.33, "IT", 0.77)],
    })

    resolved = resolve_artwork_places([_mention(text, "Clock Tower"), _mention(text, "Venice")], None, nominatim)

    assert resolved["Clock Tower"] is None


def test_a_demoted_town_not_found_inside_the_container_keeps_its_anchor_resolution():
    text = "Merchant's House in Takehara, Hiroshima"
    nominatim = FakeNominatim({
        "Takehara": [_city("Takehara", 34.34, 132.91, "JP", 0.48)],
        "Hiroshima": [_city("Hiroshima", 34.39, 132.46, "JP", 0.68)],
    })

    resolved = resolve_artwork_places([_mention(text, "Takehara"), _mention(text, "Hiroshima")], None, nominatim)

    assert resolved["Takehara"]["name"] == "Takehara"
    assert resolved["Takehara"]["resolution"] == "anchor"


def test_a_famous_place_under_a_local_name_is_kept_next_to_its_city():
    text = "From a Day of Grief, Kerch, Crimea"
    crimea = _monument("Крымский полуостров", 45.3, 34.4, "UA", importance=0.72)
    nominatim = FakeNominatim({
        "Kerch": [_city("Kerch", 45.36, 36.47, "UA", 0.59)],
        "Crimea": [crimea],
    })

    resolved = resolve_artwork_places([_mention(text, "Kerch"), _mention(text, "Crimea")], None, nominatim)

    assert resolved["Crimea"]["name"] == "Крымский полуостров"


def test_a_famous_place_sharing_a_distinctive_word_is_kept_abroad():
    text = "Vienna, Eiffel Tower"
    nominatim = FakeNominatim({
        "Vienna": [_city("Vienna", 48.21, 16.37, "AT", 0.83)],
        "Eiffel Tower": [_monument("Tour Eiffel", 48.858, 2.294, "FR", importance=0.62)],
    })

    resolved = resolve_artwork_places([_mention(text, "Vienna"), _mention(text, "Eiffel Tower")], None, nominatim)

    assert resolved["Eiffel Tower"]["name"] == "Tour Eiffel"


def _wikidata_place(name, names, latitude, longitude, country_code="FR"):
    return {
        "name": name, "names": names, "country_code": country_code, "latitude": latitude, "longitude": longitude,
        "feature_code": "wikidata_item", "place_class": "wikidata", "place_rank": 30, "importance": None,
        "source": "Wikidata", "hierarchy": None,
    }


def test_wikidata_name_score_handles_abbreviations_and_translated_generic_words():
    from src.geolocation.gazetter import _wikidata_name_score

    saint_ouen = _wikidata_place("Abbey-Church of Saint-Ouen, Rouen", ["Abbey-Church of Saint-Ouen, Rouen"], 49.44, 1.10)
    etienne = _wikidata_place("Saint-Étienne-du-Mont", ["Saint-Étienne-du-Mont", "église Saint-Étienne-du-Mont"], 48.85, 2.35)
    station = _wikidata_place("Mainz Central Station", ["Mainz Central Station", "Mainz Hauptbahnhof"], 50.0, 8.26, "DE")
    assert _wikidata_name_score("St. Ouen", saint_ouen) > 0
    assert _wikidata_name_score("Church of St. Etienne du Mont", etienne) > 0
    assert _wikidata_name_score("Cathedral of Mainz", station) == 0  # the generic word must match too
    assert _wikidata_name_score("Clock Tower", station) == 0  # generic names are never searched


def test_wikidata_is_the_last_resort_inside_the_anchor_and_prefers_the_closest_name():
    from src.geolocation.gazetter import resolve_artwork_places

    rouen = {
        "name": "Rouen", "country_code": "FR", "latitude": 49.44, "longitude": 1.10, "feature_code": "city",
        "addresstype": "city", "place_rank": 16, "importance": 0.7, "source": "Nominatim",
    }

    class Nominatim:
        def search(self, query):
            return [rouen] if query == "Rouen" else []

        def search_within(self, query, country_code, viewbox=None):
            return []

        def reverse(self, latitude, longitude):
            return None

    class Wikidata:
        def search_places(self, name):
            return [
                _wikidata_place("Saint-Ouen", ["Saint-Ouen", "Saint-Ouen-sur-Seine"], 48.91, 2.33),  # outside Rouen
                _wikidata_place("Saint-Ouen Street", ["rue Saint-Ouen"], 49.441, 1.101),
                _wikidata_place("Abbey-Church of Saint-Ouen", ["Abbey-Church of Saint-Ouen", "Saint-Ouen"], 49.443, 1.099),
            ]

    client = Nominatim()
    client.wikidata_client = Wikidata()
    text = "St. Ouen, Rouen"
    mentions = [
        ("St. Ouen", [{"field": "title", "start": 0, "end": 8, "source_text": text}]),
        ("Rouen", [{"field": "title", "start": 10, "end": 15, "source_text": text}]),
    ]
    resolved = resolve_artwork_places(mentions, None, client)
    assert resolved["St. Ouen"]["resolution"] == "wikidata"
    assert resolved["St. Ouen"]["name"] == "Abbey-Church of Saint-Ouen"
