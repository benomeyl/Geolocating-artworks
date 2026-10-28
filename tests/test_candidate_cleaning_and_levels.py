from src.data.filtering_logreg import (
    _candidate_geographic_level,
    _clean_entity_spans,
    build_geographic_label_table,
    extract_candidate_artwork,
    geographic_candidate_label,
)
from src.geolocation.gazetter import _prefer_populated_place


def _entity(text, fragment, score=0.9, occurrence=0):
    start = -1
    for _ in range(occurrence + 1):
        start = text.index(fragment, start + 1)
    return {"entity_group": "LOC", "start": start, "end": start + len(fragment), "score": score}


def _surfaces(text, entities):
    return [span["surface"] for span in _clean_entity_spans(text, entities)]


def test_spans_are_read_from_source_text_and_extended_to_whole_words():
    text = "Arrival of the Normandy Train, Gare Saint-Lazare"
    assert _surfaces(text, [_entity(text, "Gare Saint-Lazare")]) == ["Gare Saint-Lazare"]

    text = "schloss sonnenstein"
    assert _surfaces(text, [_entity(text, "sonnen")]) == ["sonnenstein"]


def test_adjacent_fragments_are_merged_and_keep_best_score():
    text = "The Shelton with Sunspots, N.Y."
    spans = _clean_entity_spans(
        text,
        [_entity(text, "N", score=0.4), _entity(text, "Y", score=0.8)],
    )
    assert [span["surface"] for span in spans] == ["N.Y"]
    assert spans[0]["score"] == 0.8


def test_leading_connectors_and_elisions_are_repaired():
    text = "Kashmir, a region fissured by the Line of Control between India and Pakistan"
    assert _surfaces(text, [_entity(text, "of Control")]) == ["Line of Control"]

    text = "restauration de la tour de la cathédrale d'Aix-en-Provence"
    assert _surfaces(text, [_entity(text, "d'Aix-en-Provence")]) == ["Aix-en-Provence"]

    text = "View of the Hague"
    assert _surfaces(text, [_entity(text, "the Hague")]) == ["Hague"]


def test_function_words_and_single_letters_are_dropped():
    text = "Portrait of a Lady with A Fan"
    assert _surfaces(text, [_entity(text, "of"), _entity(text, "A")]) == []


def test_extract_candidate_artwork_uses_cleaned_surface():
    def fake_ner(text):
        if text != "Gare Saint-Lazare, Paris":
            return []
        return [
            {**_entity(text, "Gare Saint"), "word": "Gare Saint"},
            {**_entity(text, "Lazare", score=0.95), "word": "##Lazare"},
            {**_entity(text, "Paris"), "word": "Paris"},
        ]

    candidates = extract_candidate_artwork({"title": "Gare Saint-Lazare, Paris"}, fake_ner)

    assert [candidate["place"] for candidate in candidates] == ["Gare Saint-Lazare", "Paris"]
    assert candidates[0]["score_ner"] == 0.95


def test_geonames_levels_distinguish_communes_regions_and_countries():
    def level(code):
        return _candidate_geographic_level({"source": "GeoNames", "feature_code": code})

    assert level("PPLC") == "city"
    assert level("ADM3") == "city"
    assert level("ADM1") == "region"
    assert level("RGN") == "region"
    assert level("PCLD") == "country"
    assert level("PPLX") == "neighborhood"
    assert level("CH") == "exact_place"


def test_nominatim_administrative_boundaries_use_place_rank():
    def level(**location):
        return _candidate_geographic_level({"source": "Nominatim", **location})

    # Paris, Berlin... : type "administrative", classés "region" auparavant.
    assert level(feature_code="administrative", place_rank=16) == "city"
    assert level(feature_code="administrative", place_rank=8) == "region"
    assert level(feature_code="administrative", place_rank=4) == "country"
    assert level(feature_code="administrative", place_rank=12, addresstype="city") == "city"
    assert level(feature_code="place_of_worship", place_class="amenity", place_rank=30) == "exact_place"


def test_geonames_prefers_city_over_same_name_administrative_unit():
    paris_district = {"name": "Paris", "feature_code": "ADM3", "latitude": 48.853, "longitude": 2.349}
    paris_city = {"name": "Paris", "feature_code": "PPLC", "latitude": 48.853, "longitude": 2.349}
    paris_texas = {"name": "Paris", "feature_code": "PPLA2", "latitude": 33.66, "longitude": -95.56}
    texas = {"name": "Texas", "feature_code": "ADM1", "latitude": 31.25, "longitude": -99.25}

    assert _prefer_populated_place([paris_district, paris_texas, paris_city])[0] is paris_city
    assert _prefer_populated_place([texas, paris_texas])[0] is texas


def _saxony_record():
    return {
        "artwork": {"id": "dresden"},
        "annotations": {
            "place_status": "depicted",
            "depicted_place": {
                "name": "Saxony",
                "precision": "region",
                "latitude": 51.05,
                "longitude": 13.74,
                "region": "Saxony",
                "country": "Germany",
            },
        },
    }


def test_country_matches_by_code_when_names_are_in_different_languages():
    record = _saxony_record()
    record["annotations"]["depicted_place"]["precision"] = "country"
    record["annotations"]["depicted_place"]["country_code"] = "DE"
    candidate = {
        "name": "Deutschland",
        "source": "Nominatim",
        "feature_code": "administrative",
        "place_rank": 4,
        "latitude": 51.1,
        "longitude": 10.4,
        "hierarchy": {"country": "Deutschland", "country_code": "DE"},
    }

    assert geographic_candidate_label(record, candidate)["level"] == "country"


def test_reverse_geocoded_target_hierarchy_matches_local_language_region():
    class Geocoder:
        def search(self, query):
            return [{
                "name": "Sachsen",
                "source": "Nominatim",
                "feature_code": "administrative",
                "place_rank": 8,
                "latitude": 51.0,
                "longitude": 13.3,
                "hierarchy": {"region": "Sachsen", "country": "Deutschland", "country_code": "DE"},
            }]

        def reverse(self, latitude, longitude):
            return {"hierarchy": {"city": "Dresden", "region": "Sachsen", "country": "Deutschland", "country_code": "DE"}}

    candidates = {"dresden": [{
        "place": "Sachsen",
        "score_ner": 0.9,
        "field": "title",
        "start": 0,
        "end": 7,
        "source_text": "Sachsen",
    }]}

    without_reverse = geographic_candidate_label(_saxony_record(), Geocoder().search("Sachsen")[0])
    table = build_geographic_label_table([_saxony_record()], candidates, nominatim_client=Geocoder())

    assert without_reverse["status"] == "negative"
    assert table.iloc[0]["label_level"] == "region"
    assert table.iloc[0]["label_status"] == "match"


def test_context_features_never_use_museum_location_fields():
    from src.data.filtering_logreg import artwork_context_features

    artwork = {
        "place_of_origin": "France",
        "artist_display": "Claude Monet (French, 1840–1926), born Paris",
        "exhibition_history": "Art Institute of Chicago, 1933, cat. 12.",
        "gallery_title": "Gallery 243",
        "latitude": 41.8796,
        "longitude": -87.6237,
    }

    assert artwork_context_features(artwork, "Paris") == {
        "has_context_metadata": 1,
        "object_non_pictorial": 0,
        "in_place_of_origin": 0,
        "in_artist_display": 1,
        "in_provenance_or_exhibition": 0,
    }
    assert artwork_context_features(artwork, "Chicago")["in_provenance_or_exhibition"] == 1
    assert set(artwork_context_features({}, "Paris").values()) == {0}


def test_context_flag_is_not_a_model_feature():
    from src.data.filtering_logreg import GEOGRAPHIC_FEATURE_COLUMNS

    assert "has_context_metadata" not in GEOGRAPHIC_FEATURE_COLUMNS


def test_dump_enrichment_fills_only_aic_records(tmp_path):
    import json

    from src.api.aic import enrich_records_from_dump

    (tmp_path / "16571.json").write_text(json.dumps({
        "id": 16571,
        "place_of_origin": "France",
        "alt_titles": None,
        "latitude": 41.88,
    }), encoding="utf-8")
    records = [
        {"artwork": {"id": 16571, "title": "Gare Saint-Lazare"}},
        {"artwork": {"id": "Q523974", "title": "View of Delft"}},
    ]

    assert enrich_records_from_dump(records, tmp_path) == 1
    assert records[0]["artwork"]["place_of_origin"] == "France"
    assert "latitude" not in records[0]["artwork"]
    assert records[1]["artwork"] == {"id": "Q523974", "title": "View of Delft"}


def test_entities_separated_by_space_or_sentence_break_stay_distinct():
    text = "Dresden. Albert Square"
    assert _surfaces(text, [_entity(text, "Dresden"), _entity(text, "Albert Square")]) == [
        "Dresden",
        "Albert Square",
    ]
    text = "kasernen albertstadt"
    assert _surfaces(text, [_entity(text, "kasernen"), _entity(text, "albertstadt")]) == [
        "kasernen",
        "albertstadt",
    ]
    text = "Pirna-Posta"
    assert _surfaces(text, [_entity(text, "Pirna"), _entity(text, "Posta")]) == ["Pirna-Posta"]


def test_possessive_fragment_is_reattached_to_its_word():
    text = "The Famous Architect’s Building"
    assert _surfaces(text, [_entity(text, "’s Building")]) == ["Architect’s Building"]


def test_non_pictorial_objects_are_flagged_from_artwork_type():
    from src.data.filtering_logreg import artwork_context_features

    assert artwork_context_features({"artwork_type_title": "Vessel"}, "Attica")["object_non_pictorial"] == 1
    assert artwork_context_features({"artwork_type_title": "Painting"}, "Paris")["object_non_pictorial"] == 0
    assert artwork_context_features({}, "Paris")["object_non_pictorial"] == 0
