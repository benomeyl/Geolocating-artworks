import json

from src.data import llm_places
from src.data.llm_places import LocalLlm, artwork_prompt, verify_place

ROUEN = {
    "name": "Rouen", "country_code": "FR", "latitude": 49.44, "longitude": 1.10, "feature_code": "city",
    "addresstype": "city", "place_rank": 16, "importance": 0.7, "source": "Nominatim",
}


class FakeNominatim:
    def search(self, query):
        return [ROUEN] if query.startswith("Rouen") else []

    def search_within(self, query, country_code, viewbox=None):
        return []


class FakeWikidata:
    def search_places(self, name):
        return [
            {"name": "Saint-Ouen", "names": ["Saint-Ouen"], "country_code": "FR", "latitude": 48.91,
             "longitude": 2.33, "source": "Wikidata"},  # the commune near Paris, outside Rouen
            {"name": "Abbey-Church of Saint-Ouen", "names": ["Abbey-Church of Saint-Ouen", "Abbatiale Saint-Ouen"],
             "country_code": "FR", "latitude": 49.443, "longitude": 1.099, "source": "Wikidata"},
        ]


def test_prompt_keeps_filled_fields_strips_html_and_cuts_long_descriptions():
    artwork = {
        "title": "St. Ouen, Rouen", "artist_display": "Donald Shaw MacLaughlan\nAmerican",
        "description": "<p>" + "word " * 1000 + "</p>", "subject_titles": [], "place_of_origin": None,
    }
    prompt = artwork_prompt(artwork)
    assert prompt.startswith("Title: St. Ouen, Rouen\nArtist: Donald Shaw MacLaughlan American")
    assert "<p>" not in prompt and "Subjects" not in prompt and "Place of origin" not in prompt
    assert prompt.endswith("[...]")
    assert len(prompt) < llm_places.MAX_DESCRIPTION_CHARS + 200


def test_a_monument_read_by_the_model_is_verified_inside_its_city():
    place = {"mention": "St. Ouen", "name": "Abbatiale Saint-Ouen", "type": "religious building",
             "city": "Rouen", "region": "Normandy", "country": "France", "role": "depicted"}
    location = verify_place(place, FakeNominatim(), FakeWikidata())
    assert location["name"] == "Abbey-Church of Saint-Ouen"
    assert location["verified_by"] == "wikidata"


def test_a_place_without_a_known_container_is_not_verified():
    place = {"mention": "Clock Tower", "name": "Clock Tower", "type": "monument",
             "city": "", "region": "", "country": "", "role": "depicted"}
    assert verify_place(place, FakeNominatim(), FakeWikidata()) is None


def test_answers_are_cached_and_invalid_json_is_reported(tmp_path, monkeypatch):
    calls = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "not json"}}

    def fake_post(url, json, timeout):
        calls.append(json)
        return Response()

    monkeypatch.setattr(llm_places.requests, "post", fake_post)
    llm = LocalLlm("gemma3:4b", cache_path=tmp_path / "cache.sqlite")
    first, _ = llm.read_places({"title": "Untitled"})
    second, _ = llm.read_places({"title": "Untitled"})
    assert first["places"] == [] and "invalid JSON" in first["error"]
    assert second == first and len(calls) == 1
    assert "think" not in calls[0]  # Gemma 3 has no thinking mode
    assert LocalLlm("qwen3:4b", cache_path=tmp_path / "c2.sqlite", think=False).think is False
    assert json.dumps(calls[0]["format"])  # the JSON schema is sent


def test_a_city_with_a_region_rank_keeps_a_city_radius():
    dresden = {"name": "Dresden", "country_code": "DE", "latitude": 51.05, "longitude": 13.74,
               "addresstype": "city", "place_rank": 12, "source": "Nominatim"}

    class Nominatim:
        def search(self, query):
            return [dresden] if query.startswith("Dresden") else []

        def search_within(self, query, country_code, viewbox=None):
            return []

    class Wikidata:
        def search_places(self, name):
            return [{"name": "Rathaus", "names": ["Rathaus"], "country_code": "DE", "latitude": 48.14,
                     "longitude": 11.58, "source": "Wikidata"}]  # Munich, 360 km away

    place = {"mention": "rathaus", "name": "Rathaus", "type": "building", "city": "Dresden",
             "region": "Saxony", "country": "Germany", "role": "depicted"}
    assert verify_place(place, Nominatim(), Wikidata()) is None
