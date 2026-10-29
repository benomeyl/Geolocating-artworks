"""
Geocoding of place names: local GeoNames database first, then Nominatim.

`resolve_location` is the entry point. GeoNames results come from a local
SQLite copy (see scripts/geonames.py); Nominatim results come from the public
OpenStreetMap API through `NominatimClient`, which throttles requests and
caches every response in SQLite so that a name is only queried once.
"""

import json
import logging
import math
import os
import sqlite3
import threading
import time
import unicodedata
from pathlib import Path

import requests


_LOGGER = logging.getLogger(__name__)
_NOMINATIM_ATTRIBUTION = "© OpenStreetMap contributors"
# Suffix of the cache keys; bumping it invalidates older cached responses
# (the first version did not store the address hierarchy).
_CACHE_VERSION = "address-hierarchy-v2"


class NominatimClient:
    """Small, cached and throttled client for the Nominatim search and reverse endpoints."""

    def __init__(
        self,
        *,
        endpoint=None,
        reverse_endpoint=None,
        user_agent=None,
        cache_path=None,
        minimum_interval=1.1,
        timeout=10,
        session=None,
    ):
        """Configure endpoints, User-Agent and cache (each falls back to an environment variable).

        `minimum_interval` is the minimum delay in seconds between two HTTP
        requests (the public service allows one request per second).
        """
        self.endpoint = endpoint or os.getenv(
            "NOMINATIM_URL",
            "https://nominatim.openstreetmap.org/search",
        )
        self.cache_endpoint = f"{self.endpoint}|{_CACHE_VERSION}"
        self.reverse_endpoint = reverse_endpoint or os.getenv(
            "NOMINATIM_REVERSE_URL",
            self.endpoint.rsplit("/", 1)[0] + "/reverse",
        )
        self.reverse_cache_endpoint = f"{self.reverse_endpoint}|{_CACHE_VERSION}"
        self.user_agent = user_agent or os.getenv(
            "NOMINATIM_USER_AGENT",
            "GeolocatingArtworks/0.1 (EPFL research project)",
        )
        self.cache_path = Path(
            cache_path
            or os.getenv("NOMINATIM_CACHE_PATH", "data/processed/nominatim_cache.sqlite")
        )
        self.minimum_interval = minimum_interval
        self.timeout = timeout
        self.session = session or requests.Session()
        self._last_request_at = 0.0
        self._request_lock = threading.Lock()

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.cache_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS nominatim_cache (
                    endpoint TEXT NOT NULL,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    PRIMARY KEY (endpoint, query)
                )
                """
            )

    def _cache_get(self, endpoint, key):
        """Return the cached response for (endpoint, key), or None if absent."""
        with sqlite3.connect(self.cache_path) as connection:
            row = connection.execute(
                "SELECT response FROM nominatim_cache WHERE endpoint = ? AND query = ?",
                (endpoint, key),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def _cache_put(self, endpoint, key, response):
        """Store a response in the cache."""
        with sqlite3.connect(self.cache_path) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO nominatim_cache VALUES (?, ?, ?)",
                (endpoint, key, json.dumps(response)),
            )

    def _cached_response(self, query):
        """Cached search response for `query` (case-insensitive), or None."""
        return self._cache_get(self.cache_endpoint, query.casefold().strip())

    def _store_response(self, query, response):
        """Cache a search response for `query`."""
        self._cache_put(self.cache_endpoint, query.casefold().strip(), response)

    def _request_json(self, url, params):
        """GET `url` respecting `minimum_interval`; return the JSON body or raise.

        Must be called while holding `_request_lock`.
        """
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.minimum_interval:
            time.sleep(self.minimum_interval - elapsed)
        response = self.session.get(
            url,
            params={**params, "format": "jsonv2", "addressdetails": 1},
            headers={"User-Agent": self.user_agent},
            timeout=self.timeout,
        )
        self._last_request_at = time.monotonic()
        response.raise_for_status()
        return response.json()

    def _normalize_result(self, result, fallback_name):
        """Convert a raw Nominatim result into the project's location dict (None if no coordinates)."""
        try:
            latitude = float(result["lat"])
            longitude = float(result["lon"])
        except (KeyError, TypeError, ValueError):
            return None
        address = result.get("address") or {}
        country_code = (address.get("country_code") or "").upper() or None
        return {
            "name": result.get("name") or result.get("display_name") or fallback_name,
            "country_code": country_code,
            "latitude": latitude,
            "longitude": longitude,
            "feature_code": result.get("type"),
            # jsonv2 names the OSM class "category" ("class" in the json format).
            "place_class": result.get("category") or result.get("class"),
            "addresstype": result.get("addresstype"),
            "place_rank": result.get("place_rank"),
            "importance": result.get("importance"),
            "osm_type": result.get("osm_type"),
            "osm_id": result.get("osm_id"),
            "hierarchy": {
                "neighborhood": (
                    address.get("neighbourhood")
                    or address.get("suburb")
                    or address.get("city_district")
                ),
                "city": (
                    address.get("city")
                    or address.get("town")
                    or address.get("village")
                    or address.get("municipality")
                ),
                "region": (
                    address.get("state")
                    or address.get("region")
                    or address.get("province")
                    or address.get("county")
                ),
                "country": address.get("country"),
                "country_code": country_code,
            },
            "population": None,
            "source": "Nominatim",
            "attribution": _NOMINATIM_ATTRIBUTION,
        }

    def search(self, query):
        """Return up to 5 normalized locations for a place name ([] on failure, not cached)."""
        cached = self._cached_response(query)
        if cached is not None:
            return cached

        with self._request_lock:
            # Another thread may have fetched it while we were waiting.
            cached = self._cached_response(query)
            if cached is not None:
                return cached
            try:
                results = self._request_json(self.endpoint, {"q": query, "limit": 5})
            except (requests.RequestException, ValueError) as error:
                _LOGGER.warning("Nominatim lookup failed for %r: %s", query, error)
                return []

            normalized = [
                location
                for result in results
                if (location := self._normalize_result(result, query)) is not None
            ]
            self._store_response(query, normalized)
            return normalized

    def reverse(self, latitude, longitude):
        """Return the normalized location at these coordinates, or None (failures are not cached)."""
        key = f"reverse:{latitude:.5f},{longitude:.5f}"
        cached = self._cache_get(self.reverse_cache_endpoint, key)
        if cached:
            return cached

        with self._request_lock:
            cached = self._cache_get(self.reverse_cache_endpoint, key)
            if cached:
                return cached
            try:
                result = self._request_json(self.reverse_endpoint, {"lat": latitude, "lon": longitude})
            except (requests.RequestException, ValueError) as error:
                _LOGGER.warning("Nominatim reverse lookup failed for %s: %s", key, error)
                return None

            location = self._normalize_result(result, key)
            if location is not None:
                self._cache_put(self.reverse_cache_endpoint, key, location)
            return location


def get_location_info(name, conn):
    """
    Look up a place name in the local GeoNames database.

    Args:
        name (str): Exact name (or ASCII name) to search for.
        conn (sqlite3.Connection): Connection to the GeoNames SQLite database.
    Returns:
        list: Up to 10 tuples (name, country_code, latitude, longitude,
        feature_code, population), most populated first.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT name, country_code, latitude, longitude, feature_code, population
        FROM geonames
        WHERE name = ? OR asciiname = ?
        ORDER BY population DESC
        LIMIT 10
    """, (name, name))
    return cur.fetchall()


def _prefer_populated_place(locations, max_distance_km=30.0):
    """Put a city before the administrative unit that shares its name and area.

    GeoNames often ranks the commune or district first ("Paris" ADM3, 4.4M
    inhabitants, ahead of "Paris" PPLC), which would label the candidate as
    a region. A populated place (PPL*) close to that unit is preferred.
    """
    top = locations[0]
    if not str(top.get("feature_code") or "").startswith("ADM"):
        return locations
    for index, location in enumerate(locations[1:], start=1):
        if str(location.get("feature_code") or "").startswith("PPL") and haversine_distance_km(
            top["latitude"], top["longitude"], location["latitude"], location["longitude"],
        ) <= max_distance_km:
            return [location] + locations[:index] + locations[index + 1:]
    return locations


def haversine_distance_km(latitude_a, longitude_a, latitude_b, longitude_b):
    """Return the great-circle distance between two coordinates in km."""
    earth_radius_km = 6371.0088
    latitude_a, longitude_a, latitude_b, longitude_b = map(
        math.radians,
        (latitude_a, longitude_a, latitude_b, longitude_b),
    )
    latitude_delta = latitude_b - latitude_a
    longitude_delta = longitude_b - longitude_a
    haversine = (
        math.sin(latitude_delta / 2) ** 2
        + math.cos(latitude_a) * math.cos(latitude_b) * math.sin(longitude_delta / 2) ** 2
    )
    return 2 * earth_radius_km * math.asin(math.sqrt(haversine))


def _geonames_location(row):
    """Convert a get_location_info row into the project's location dict."""
    name, country_code, latitude, longitude, feature_code, population = row
    return {
        "name": name,
        "country_code": country_code,
        "latitude": float(latitude),
        "longitude": float(longitude),
        "feature_code": feature_code,
        "population": population,
        "source": "GeoNames",
        "hierarchy": {
            "neighborhood": None,
            "city": None,
            "region": None,
            "country": None,
            "country_code": country_code,
        },
    }


# GeoNames feature codes of sovereign and dependent countries.
_COUNTRY_FEATURE_CODES = ("PCLI", "PCLD", "PCLF", "PCLS", "PCLIX", "PCL", "TERR")
# Country indexes already built, by database file.
_COUNTRY_INDEXES = {}
# Inhabited places smaller than this are not trusted on an exact-name match
# alone (a 184-inhabitant "Puerta del Sol" in Mexico hides the Madrid square).
_TRUSTED_MIN_POPULATION = 1000
# A Nominatim result this close to a GeoNames entry of the same name is
# confirmed by two independent sources.
_CONFIRMATION_KM = 5.0


def _database_path(conn):
    """File of the main database of a connection ("" for an in-memory database)."""
    try:
        return next((row[2] for row in conn.execute("PRAGMA database_list") if row[1] == "main"), "")
    except sqlite3.Error:
        return ""


def _country_index(conn):
    """Map every name of a country (official, ASCII, alternate) to its GeoNames location.

    GeoNames names countries officially ("Kingdom of Spain", "Republic of
    Peru"), so an exact-name lookup of "Spain" only finds homonyms such as a
    hamlet in Georgia (USA). Built once per database file (a full scan of a
    few seconds); empty if the table has no alternatenames column.
    """
    path = _database_path(conn)
    if path and path in _COUNTRY_INDEXES:
        return _COUNTRY_INDEXES[path]
    placeholders = ",".join("?" for _ in _COUNTRY_FEATURE_CODES)
    try:
        rows = conn.execute(
            "SELECT name, asciiname, alternatenames, country_code, latitude, longitude, feature_code, population "
            f"FROM geonames WHERE feature_code IN ({placeholders}) AND population > 0",
            _COUNTRY_FEATURE_CODES,
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    index = {}
    for name, ascii_name, alternate_names, *location_row in sorted(rows, key=lambda row: -(row[7] or 0)):
        location = _geonames_location((name, *location_row))
        for alias in [name, ascii_name, *(alternate_names or "").split(",")]:
            key = (alias or "").casefold().strip()
            if len(key) > 2:  # Skip ISO codes such as "es".
                index.setdefault(key, location)
    if path:
        _COUNTRY_INDEXES[path] = index
    return index


def _is_trusted_geonames(location):
    """True for GeoNames countries, administrative units and inhabited places of some size.

    Other GeoNames features (hotels, farms, hills...) are only used to
    confirm Nominatim's answer: an exact-name hit like the "Louvre" hotel in
    Greece is worse than Nominatim's answer.
    """
    feature_code = str(location.get("feature_code") or "")
    if feature_code.startswith(("PCL", "ADM")) or feature_code == "TERR":
        return True
    return feature_code.startswith("PPL") and (location.get("population") or 0) >= _TRUSTED_MIN_POPULATION


def location_alternatives(name, conn, nominatim_client=None):
    """Ranked candidate locations for a place name, without hierarchy lookup.

    Trusted GeoNames matches (countries included, see `_country_index`) come
    first, most populated first. If there is none, Nominatim's results are
    used: those confirmed by an untrusted GeoNames entry of the same name
    first ("Point Lobos" in California rather than a Texas street), then the
    others. Untrusted GeoNames matches are used only when Nominatim finds
    nothing.
    """
    geonames = []
    if conn is not None:
        geonames = [_geonames_location(row) for row in get_location_info(name, conn)]
        country = _country_index(conn).get(name.casefold().strip())
        already_listed = country is not None and any(
            location["country_code"] == country["country_code"]
            and str(location["feature_code"]).startswith("PCL")
            for location in geonames
        )
        if country is not None and not already_listed:
            geonames.append(country)
        geonames.sort(key=lambda location: -(location.get("population") or 0))
        geonames = _prefer_populated_place(geonames) if geonames else []
    trusted = [location for location in geonames if _is_trusted_geonames(location)]
    if trusted:
        return trusted
    nominatim = nominatim_client.search(name) if nominatim_client is not None else []
    if not nominatim:
        return geonames
    confirmed = [
        location for location in nominatim
        if any(
            haversine_distance_km(
                location["latitude"], location["longitude"], other["latitude"], other["longitude"],
            ) <= _CONFIRMATION_KM
            for other in geonames
        )
    ]
    return confirmed + [location for location in nominatim if location not in confirmed]


def _with_hierarchy(location, nominatim_client):
    """Give a GeoNames location its city/region/country from a Nominatim reverse lookup."""
    if location.get("source") != "GeoNames" or nominatim_client is None:
        return location
    reverse_result = nominatim_client.reverse(location["latitude"], location["longitude"])
    if not reverse_result:
        return location
    return {**location, "hierarchy": reverse_result["hierarchy"], "hierarchy_source": "Nominatim"}


def resolve_location(
    name,
    conn,
    reference_coordinates=None,
    nominatim_client=None,
    include_hierarchy=False,
):
    """Resolve a place name with `location_alternatives` and return one location.

    With `reference_coordinates`, the alternative closest to them is returned,
    otherwise the first one. With `include_hierarchy`, a GeoNames result gets
    its city/region/country from a Nominatim reverse lookup. Returns None if
    nothing is found.
    """
    locations = location_alternatives(name, conn, nominatim_client)
    if not locations:
        return None

    if reference_coordinates is not None:
        reference_latitude, reference_longitude = reference_coordinates
        result = min(
            locations,
            key=lambda location: haversine_distance_km(
                reference_latitude, reference_longitude, location["latitude"], location["longitude"],
            ),
        )
    else:
        result = locations[0]
    return _with_hierarchy(result, nominatim_client) if include_hierarchy else result


# ---------------------------------------------------------------------------
# Joint resolution of the places mentioned together in an artwork
# ---------------------------------------------------------------------------

# Two mentions closer than this (in characters, same field) give each other context.
CONTEXT_WINDOW_CHARS = 80
# Two places closer than this are considered to belong to the same area.
NEARBY_KM = 150.0
# Finest Nominatim place_rank of an anchor (16 city, 19 village, 20 hamlet).
_ANCHOR_MAX_PLACE_RANK = 20


def _name_tokens(name):
    """Lowercase words of a name, without accents or punctuation."""
    text = unicodedata.normalize("NFKD", (name or "").casefold())
    text = "".join(character if character.isalnum() else " " for character in text if not unicodedata.combining(character))
    return set(text.split())


def _is_anchor(name, location):
    """True if a mention is reliable enough to give context to its neighbours.

    Anchors are countries, regions and inhabited places whose name matches
    the mention ("Peru" -> "Republic of Peru", "Illinois"). Other mentions,
    often misread common words ("Surf", "Memorial Column") geocoded to a
    random homonym, would mislead the disambiguation.
    """
    if location.get("source") == "GeoNames":
        reliable_level = _is_trusted_geonames(location)
    else:
        try:
            place_rank = int(location.get("place_rank") or 0)
        except (TypeError, ValueError):
            place_rank = 0
        reliable_level = 0 < place_rank <= _ANCHOR_MAX_PLACE_RANK
    mention_tokens, location_tokens = _name_tokens(name), _name_tokens(location.get("name"))
    names_match = bool(mention_tokens) and (mention_tokens <= location_tokens or location_tokens <= mention_tokens)
    return reliable_level and names_match


def _mentioned_together(occurrences_a, occurrences_b, window=CONTEXT_WINDOW_CHARS):
    """True if two places are mentioned in the same field within `window` characters."""
    return any(
        a["field"] == b["field"]
        and a.get("source_text") == b.get("source_text")
        and max(a["start"], b["start"]) - min(a["end"], b["end"]) <= window
        for a in occurrences_a
        for b in occurrences_b
    )


def _is_country(location):
    """True for a country (GeoNames PCL* or Nominatim rank 4 / type country)."""
    if str(location.get("feature_code") or "").startswith("PCL"):
        return True
    return location.get("source") == "Nominatim" and (
        location.get("feature_code") == "country" or location.get("place_rank") == 4
    )


def _agreement(location, context_location):
    """How well two locations fit together: same country (+1) and nearby (+1)."""
    country_code = location.get("country_code")
    same_country = bool(country_code) and country_code == context_location.get("country_code")
    nearby = not _is_country(context_location) and haversine_distance_km(
        location["latitude"], location["longitude"],
        context_location["latitude"], context_location["longitude"],
    ) <= NEARBY_KM
    return int(same_country) + int(nearby)


def _contextual_search(name, context_names, context_locations, nominatim_client):
    """Search "name, context" with Nominatim and keep a result in the context's country."""
    if nominatim_client is None:
        return None
    for context_name, context_location in zip(context_names, context_locations):
        for location in nominatim_client.search(f"{name}, {context_name}"):
            if _agreement(location, context_location) > 0:
                return location
    return None


def resolve_artwork_places(mentions, conn, nominatim_client=None, alternatives_cache=None):
    """Resolve the places of one artwork together, using each other as context.

    `mentions` is a list of (name, occurrences) where occurrences carry
    `field`, `start`, `end` and `source_text`. A place mentioned near anchor
    places (`_is_anchor`: "Logan County" next to "Cornland", "San Salvador"
    next to "Peru") takes the alternative that best agrees with their best
    guesses: same country, nearby. If none agrees, "name, context" is
    searched with Nominatim; if that fails too, the first alternative is kept
    and flagged with `context_conflict`.

    Returns {name: location or None}; locations include `context_conflict`.
    """
    alternatives_cache = {} if alternatives_cache is None else alternatives_cache

    def alternatives(name):
        """Cached `location_alternatives` of a name."""
        key = name.casefold().strip()
        if key not in alternatives_cache:
            alternatives_cache[key] = location_alternatives(name, conn, nominatim_client)
        return alternatives_cache[key]

    resolved = {}
    for name, occurrences in mentions:
        options = alternatives(name)
        if not options:
            resolved[name] = None
            continue
        context = [
            (other_name, alternatives(other_name)[0])
            for other_name, other_occurrences in mentions
            if other_name != name
            and _mentioned_together(occurrences, other_occurrences)
            and alternatives(other_name)
            and _is_anchor(other_name, alternatives(other_name)[0])
        ]
        choice, conflict = options[0], False
        # A country name is not ambiguous enough to be revised by smaller places.
        if context and not _is_country(options[0]):
            scores = [sum(_agreement(option, location) for _, location in context) for option in options]
            best_index = max(range(len(options)), key=lambda index: (scores[index], -index))
            if scores[best_index] > 0:
                choice = options[best_index]
            else:
                contextual = _contextual_search(
                    name, [n for n, _ in context], [loc for _, loc in context], nominatim_client
                )
                choice, conflict = (contextual, False) if contextual else (options[0], True)
        resolved[name] = {**_with_hierarchy(choice, nominatim_client), "context_conflict": conflict}
    return resolved


def geocode_candidates(results, conn=None, nominatim_client=None, limit_per_artwork=3):
    """Attach a `geolocation` to the first `limit_per_artwork` candidates of each artwork (in place)."""
    location_cache = {}
    for candidates in results.values():
        selected_candidates = candidates if limit_per_artwork is None else candidates[:limit_per_artwork]
        for candidate in selected_candidates:
            name = candidate.get("place")
            if not name:
                continue
            cache_key = name.casefold().strip()
            if cache_key not in location_cache:
                location_cache[cache_key] = resolve_location(
                    name,
                    conn,
                    nominatim_client=nominatim_client,
                    include_hierarchy=True,
                )
            if location_cache[cache_key] is not None:
                candidate["geolocation"] = location_cache[cache_key]
    return results


def geocode_top_candidates(results, conn=None, nominatim_client=None):
    """Backwards-compatible helper that geocodes only each artwork's top candidate."""
    return geocode_candidates(results, conn=conn, nominatim_client=nominatim_client, limit_per_artwork=1)


def validate_location_candidate(loc_candidate, conn):
    """
    Check whether a place name exists in GeoNames.

    Args:
        loc_candidate (str): Place name to look up.
        conn (sqlite3.Connection): Connection to the GeoNames SQLite database.
    Returns:
        dict: {"validate": False, "message": ...} when not found, otherwise
        {"validate": True} with the name, country code, coordinates and
        feature code of the most populated match.
    """
    results = get_location_info(loc_candidate, conn)
    if not results:
        return {"validate": False, "message": f"No location found for '{loc_candidate}'."}

    best = results[0]
    return {
        "validate": True,
        "name": best[0],
        "country_code": best[1],
        "latitude": best[2],
        "longitude": best[3],
        "feature_code": best[4],
    }
