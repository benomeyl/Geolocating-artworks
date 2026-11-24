"""
Places of an artwork read by a local LLM (Ollama), then checked in gazetteers.

Why: Nominatim only finds names written in OpenStreetMap, and a Wikidata
search returns many homonyms. A language model knows that "St. Ouen, Rouen"
is the Abbatiale Saint-Ouen and that "Conway Castle" is Conwy Castle in
Wales; it can also tell a place that the artwork *shows* from a place that is
only *mentioned* (where it was made, the city of a portrait's sitter), the
main source of the model's confident false positives.

How, in three steps:

1. `LocalLlm.read_places(artwork)` sends the artwork's metadata (text only,
   never the image) to a model served by Ollama on this machine and gets
   back JSON (`PLACES_SCHEMA`): whether the artwork depicts a place, and for
   each place its name as written, its usual name, its type, city, region,
   country and role (depicted / origin / mentioned).
2. `verify_place(place, ...)` never trusts the model's knowledge alone: the
   place's city/country is geocoded with Nominatim, then the place is
   searched in Wikidata and Nominatim *inside that area*; it is kept only
   if a gazetteer entry there carries its name.
3. The caller decides what to do with verified places (scripts/
   compare_local_llms.py only measures them for now).

Answers are cached in SQLite (data/processed/llm_cache.sqlite) per model and
prompt, so a run can be resumed and repeated identically. Requests use
temperature 0 and a fixed seed.
"""

import hashlib
import html
import json
import re
import sqlite3
import time
from pathlib import Path

import requests

try:
    from src.geolocation.gazetter import (
        _ARTICLES,
        _CITY_BOX_DEGREES,
        _name_tokens,
        _REGION_BOX_DEGREES,
        CITY_RADIUS_KM,
        REGION_RADIUS_KM,
        _is_relevant_result,
        _names_the_mention,
        _wikidata_name_score,
        haversine_distance_km,
    )
except ImportError:  # pragma: no cover - when run with src/ on sys.path
    from geolocation.gazetter import (
        _ARTICLES,
        _CITY_BOX_DEGREES,
        _name_tokens,
        _REGION_BOX_DEGREES,
        CITY_RADIUS_KM,
        REGION_RADIUS_KM,
        _is_relevant_result,
        _names_the_mention,
        _wikidata_name_score,
        haversine_distance_km,
    )

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_CACHE_PATH = Path("data/processed/llm_cache.sqlite")
# Long AIC descriptions are cut to keep CPU inference time reasonable
# (a 4B model on 8 cores reads roughly 50-100 tokens per second).
MAX_DESCRIPTION_CHARS = 1500

PLACE_TYPES = [
    "building", "religious building", "castle or palace", "bridge", "street", "square", "park or garden",
    "monument", "natural site", "neighborhood", "city", "region", "country", "other",
]
PLACE_ROLES = ["depicted", "origin", "mentioned"]
# Types whose "place" is itself an area: verified by geocoding the name, not
# by searching inside a container.
AREA_TYPES = {"city", "region", "country", "neighborhood"}

PLACES_SCHEMA = {
    "type": "object",
    "properties": {
        "depicts_place": {"type": "boolean"},
        "places": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "mention": {"type": "string"},
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": PLACE_TYPES},
                    "city": {"type": "string"},
                    "region": {"type": "string"},
                    "country": {"type": "string"},
                    "role": {"type": "string", "enum": PLACE_ROLES},
                },
                "required": ["mention", "name", "type", "city", "region", "country", "role"],
            },
        },
    },
    "required": ["depicts_place", "places"],
}

SYSTEM_PROMPT = """You read the catalogue record of an artwork and list the real places it names.
For each place, give:
- mention: the words of the record that name it, copied exactly;
- name: its usual full name in English, or in its own language if that is how it is known
  (e.g. "Abbey of Saint-Germain-des-Prés", "Ponte Vecchio");
- type: one of the allowed types;
- city, region, country: where it is (empty string if unknown or not applicable);
- role: "depicted" if the artwork shows this place (a view, a landscape, a building, a street,
  the interior of a named building); "origin" if it is where the artwork, the object or the
  artist comes from, or where it was made, printed or published; "mentioned" if it is only
  named (the city of a portrait's sitter, an owner, an event) without being shown.
depicts_place is true only if at least one place is depicted.
Rules: use only places named in the record or certainly identified by it; never invent one.
A portrait, a still life or an object is not a view of the place in its title. If unsure of
the role, prefer "mentioned". If no place is named, return an empty list.

Examples:
Title "The Ponte Vecchio at Sunset, Florence" ->
  {"depicts_place": true, "places": [
   {"mention": "Ponte Vecchio", "name": "Ponte Vecchio", "type": "bridge", "city": "Florence",
    "region": "Tuscany", "country": "Italy", "role": "depicted"},
   {"mention": "Florence", "name": "Florence", "type": "city", "city": "Florence",
    "region": "Tuscany", "country": "Italy", "role": "depicted"}]}
Title "Portrait of the Artist's Wife, Berlin" ->
  {"depicts_place": false, "places": [
   {"mention": "Berlin", "name": "Berlin", "type": "city", "city": "Berlin", "region": "",
    "country": "Germany", "role": "mentioned"}]}
Title "Covered Jar", place of origin "Jingdezhen, China" ->
  {"depicts_place": false, "places": [
   {"mention": "Jingdezhen", "name": "Jingdezhen", "type": "city", "city": "Jingdezhen",
    "region": "Jiangxi", "country": "China", "role": "origin"}]}
Answer with JSON only."""

# Record fields given to the model, in this order, with their label.
_FIELDS = [
    ("title", "Title"),
    ("alt_titles", "Other titles"),
    ("artwork_type_title", "Type"),
    ("date_display", "Date"),
    ("artist_display", "Artist"),
    ("place_of_origin", "Place of origin"),
    ("subject_titles", "Subjects"),
    ("short_description", "Short description"),
    ("description", "Description"),
]


def _plain_text(value):
    """Text of a field value: lists joined, HTML tags removed, whitespace collapsed."""
    if isinstance(value, (list, tuple)):
        value = "; ".join(str(item) for item in value if item)
    text = html.unescape(re.sub(r"<[^>]+>", " ", str(value or "")))
    return " ".join(text.split())


def artwork_prompt(artwork):
    """The user message describing one artwork (fields that are empty are left out)."""
    lines = []
    for key, label in _FIELDS:
        text = _plain_text(artwork.get(key))
        if not text:
            continue
        if key == "description" and len(text) > MAX_DESCRIPTION_CHARS:
            text = text[:MAX_DESCRIPTION_CHARS].rsplit(" ", 1)[0] + " [...]"
        lines.append(f"{label}: {text}")
    return "\n".join(lines)


class LocalLlm:
    """A model served by a local Ollama server, with a persistent cache of its answers.

    `think` is passed to Ollama only when not None: Qwen3 is a reasoning
    model whose thinking is switched off here (think=False) to keep answers
    short; Gemma 3 has no thinking mode and rejects the parameter.
    """

    def __init__(self, model, *, url=DEFAULT_OLLAMA_URL, cache_path=DEFAULT_CACHE_PATH, think=None,
                 timeout=900, context_tokens=8192):
        self.model = model
        # Name used in reports: "qwen3:4b+think" when the reasoning mode is on.
        self.label = f"{model}+think" if think else model
        self.url = url.rstrip("/")
        self.think = think
        self.timeout = timeout
        self.context_tokens = context_tokens
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.cache_path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS llm_cache (key TEXT PRIMARY KEY, answer TEXT NOT NULL, seconds REAL)"
            )

    def _key(self, prompt):
        """Cache key: model, system prompt, schema and artwork prompt (+ "think" when reasoning is on).

        The reasoning flag is only added when it is on, so that answers
        cached before it existed (think off or not applicable) stay valid.
        """
        parts = [self.model, SYSTEM_PROMPT, PLACES_SCHEMA, prompt] + (["think"] if self.think else [])
        payload = json.dumps(parts, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def read_places(self, artwork):
        """(answer dict, seconds) for one artwork; seconds is the original inference time if cached.

        The answer follows PLACES_SCHEMA; a reply that is not valid JSON is
        returned as {"depicts_place": False, "places": [], "error": ...}.
        """
        return self.read_prompt(artwork_prompt(artwork))

    def cached(self, prompt):
        """(answer, seconds) cached for a prompt, or None."""
        with sqlite3.connect(self.cache_path) as connection:
            row = connection.execute("SELECT answer, seconds FROM llm_cache WHERE key = ?", (self._key(prompt),)).fetchone()
        return (json.loads(row[0]), row[1]) if row else None

    def store(self, prompt, answer, seconds):
        """Put an answer computed elsewhere (scripts/run_llm_places.py import) in the cache."""
        with sqlite3.connect(self.cache_path) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO llm_cache VALUES (?, ?, ?)", (self._key(prompt), json.dumps(answer), seconds)
            )

    def read_prompt(self, prompt):
        """(answer dict, seconds) for an artwork prompt already built by `artwork_prompt`."""
        key = self._key(prompt)
        with sqlite3.connect(self.cache_path) as connection:
            row = connection.execute("SELECT answer, seconds FROM llm_cache WHERE key = ?", (key,)).fetchone()
        if row:
            return json.loads(row[0]), row[1]

        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            "format": PLACES_SCHEMA,
            "stream": False,
            "options": {"temperature": 0, "seed": 0, "num_ctx": self.context_tokens},
        }
        if self.think is not None:
            body["think"] = self.think
        started = time.monotonic()
        response = requests.post(f"{self.url}/api/chat", json=body, timeout=self.timeout)
        response.raise_for_status()
        seconds = time.monotonic() - started
        content = response.json().get("message", {}).get("content", "")
        try:
            answer = json.loads(content)
        except json.JSONDecodeError as error:
            answer = {"depicts_place": False, "places": [], "error": f"invalid JSON: {error}", "raw": content[:500]}
        with sqlite3.connect(self.cache_path) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO llm_cache VALUES (?, ?, ?)", (key, json.dumps(answer), seconds)
            )
        return answer, seconds


def _first_relevant(locations):
    """First Nominatim result that is a representable place, or None."""
    return next((location for location in locations if _is_relevant_result(location)), None)


def locate_container(place, nominatim_client):
    """(location, scale) of the area given by the model for a place, or (None, None).

    The scale comes from the field the model filled ("city" or "region"),
    not from Nominatim's place_rank: German district-free cities such as
    Dresden or Nuremberg have the rank of a region (12), which would widen
    the search to 500 km. A country alone is not precise enough to confirm
    a monument and is not returned.
    """
    country = (place.get("country") or "").strip()
    for field in ("city", "region"):
        container = (place.get(field) or "").strip()
        if container:
            query = f"{container}, {country}" if country else container
            location = _first_relevant(nominatim_client.search(query))
            if location is not None and location.get("country_code"):
                return location, field
    return None, None


def _in_container(location, container, scale):
    """True if a location is in the container's country and within its radius (city 30 km, region 500 km)."""
    if not location.get("country_code") or location["country_code"] != container["country_code"]:
        return False
    radius = CITY_RADIUS_KM if scale == "city" else REGION_RADIUS_KM
    return distance_km(location, container["latitude"], container["longitude"]) <= radius


def _container_area(container, scale):
    """(country_code, viewbox) for a bounded Nominatim search inside the container."""
    half_size = _CITY_BOX_DEGREES if scale == "city" else _REGION_BOX_DEGREES
    return container["country_code"], (
        container["longitude"] - half_size, container["latitude"] - half_size,
        container["longitude"] + half_size, container["latitude"] + half_size,
    )


def verify_place(place, nominatim_client, wikidata_client=None):
    """Gazetteer location confirming a place read by the model, or None.

    - Areas (city, region, country, neighbourhood): the place's own name is
      geocoded with its country ("Rouen, France").
    - Other places: searched inside their city or region (`locate_container`), in
      Wikidata (labels and aliases, `_wikidata_name_score` > 0 for the
      model's name or the record's words), then with a bounded Nominatim
      search; the result must lie in the container's area.
    The location gets `verified_by` ("nominatim_area", "wikidata" or
    "nominatim_bounded").
    """
    name = (place.get("name") or "").strip()
    mention = (place.get("mention") or "").strip()
    if not name and not mention:
        return None
    country = (place.get("country") or "").strip()
    if place.get("type") in AREA_TYPES:
        query = f"{name or mention}, {country}" if country else (name or mention)
        location = _first_relevant(nominatim_client.search(query))
        return {**location, "verified_by": "nominatim_area"} if location else None

    container, scale = locate_container(place, nominatim_client)
    if container is None:
        return None
    if wikidata_client is not None:
        scored = []
        for query in dict.fromkeys(filter(None, (name, mention))):
            for index, location in enumerate(wikidata_client.search_places(query)):
                if not _in_container(location, container, scale):
                    continue
                score = max(_wikidata_name_score(name, location), _wikidata_name_score(mention, location))
                if score > 0:
                    scored.append((score, -index, location))
        if scored:
            return {**max(scored, key=lambda item: item[:2])[2], "verified_by": "wikidata"}
    for query in dict.fromkeys(filter(None, (name, mention))):
        for location in nominatim_client.search_within(query, *_container_area(container, scale)):
            if _is_relevant_result(location) and _in_container(location, container, scale) and _names_the_mention(query, location):
                return {**location, "verified_by": "nominatim_bounded"}
    return None


def distance_km(location, latitude, longitude):
    """Distance from a location to a point, in km."""
    return haversine_distance_km(location["latitude"], location["longitude"], latitude, longitude)


# ---------------------------------------------------------------------------
# Features of the hierarchical model taken from the model's answer
# ---------------------------------------------------------------------------

LLM_FEATURE_COLUMNS = [
    "llm_depicts_place",      # the LLM says the artwork shows a place
    "llm_role_depicted",      # the candidate is a place the LLM calls depicted
    "llm_role_origin",        # ... a place of origin (made in, artist from)
    "llm_role_mentioned",     # ... a place only mentioned
    "llm_in_depicted_place",  # the candidate is the city/region/country of a depicted place
]


def _tokens(text):
    """Name tokens without articles."""
    return _name_tokens(text) - _ARTICLES


def _same_name(candidate_tokens, text):
    """True if a candidate and a name of the answer share their words (one inside the other)."""
    tokens = _tokens(text)
    return bool(candidate_tokens) and bool(tokens) and (candidate_tokens <= tokens or tokens <= candidate_tokens)


def candidate_llm_features(candidate_name, answer):
    """LLM_FEATURE_COLUMNS of one NER candidate, from the LLM answer of its artwork (all 0 without answer).

    A candidate matches a place of the answer when its words and those of
    the place's mention or usual name are one inside the other ("Cathedral
    of Mainz" ~ "Mainz Cathedral" does not match, "St. Ouen" ~ "St. Ouen"
    does); city-level candidates also get `llm_in_depicted_place` when they
    name the city, region or country of a depicted place.
    """
    features = dict.fromkeys(LLM_FEATURE_COLUMNS, 0)
    if not answer:
        return features
    features["llm_depicts_place"] = int(bool(answer.get("depicts_place")))
    candidate_tokens = _tokens(candidate_name)
    for place in answer.get("places", []):
        if _same_name(candidate_tokens, place.get("mention")) or _same_name(candidate_tokens, place.get("name")):
            role = place.get("role")
            if role in PLACE_ROLES:
                features[f"llm_role_{role}"] = 1
        if place.get("role") == "depicted" and any(
            _same_name(candidate_tokens, place.get(field)) for field in ("city", "region", "country")
        ):
            features["llm_in_depicted_place"] = 1
    return features
