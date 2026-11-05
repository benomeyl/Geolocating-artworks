"""
Geocoding of place names: local GeoNames database first, then Nominatim.

`resolve_location` is the entry point. GeoNames results come from a local
SQLite copy (see scripts/geonames.py); Nominatim results come from the public
OpenStreetMap API through `NominatimClient`, which throttles requests and
caches every response in SQLite so that a name is only queried once.
Homonyms from both sources are ranked by Nominatim importance
(`location_alternatives`).
"""

import json
import logging
import math
import os
import re
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

    def search_within(self, query, country_code, viewbox=None):
        """Return up to 5 locations named `query` inside a country, and inside `viewbox` if given.

        `viewbox` is (west, south, east, north) in degrees; results outside it
        are excluded (Nominatim `bounded=1`). Cached like `search`, under a
        key that includes the area ([] on failure, not cached).
        """
        area = f"cc={country_code.lower()}"
        if viewbox is not None:
            area += "|vb=" + ",".join(f"{value:.2f}" for value in viewbox)
        key = f"{query}||{area}"
        cached = self._cached_response(key)
        if cached is not None:
            return cached

        params = {"q": query, "limit": 5, "countrycodes": country_code.lower()}
        if viewbox is not None:
            params.update({"viewbox": ",".join(f"{value:.4f}" for value in viewbox), "bounded": 1})
        with self._request_lock:
            cached = self._cached_response(key)
            if cached is not None:
                return cached
            try:
                results = self._request_json(self.endpoint, params)
            except (requests.RequestException, ValueError) as error:
                _LOGGER.warning("Nominatim bounded lookup failed for %r: %s", key, error)
                return []
            normalized = [
                location
                for result in results
                if (location := self._normalize_result(result, query)) is not None
            ]
            self._store_response(key, normalized)
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
# Importance bonus of a Nominatim result confirmed that way.
_CONFIRMATION_BONUS = 0.05
# A trusted GeoNames entry and a Nominatim result describe the same place
# when they share the country and are this close (administrative units are
# compared through their centroids, hence the larger radius).
_SAME_PLACE_KM = 25.0
_SAME_ADMINISTRATIVE_UNIT_KM = 100.0
# Importance assumed for a trusted GeoNames entry absent from Nominatim's
# 5 results: it beats obscure homonyms, not famous ones (Corcovado, the
# mountain in Rio, 0.45, beats a 1,500-inhabitant Corcovado in Puerto Rico).
_UNMATCHED_GEONAMES_IMPORTANCE = 0.25
# Nominatim results that cannot be a depicted place: shops, restaurants,
# transport stops, plain buildings, minor paths, street artworks
# ("Microcosm" -> a copy shop in Belarus, "Sukiya Bridge" -> a fast food).
_IRRELEVANT_CLASSES = {"shop", "craft", "office", "emergency", "healthcare"}
_IRRELEVANT_TYPES = {
    "fast_food", "restaurant", "cafe", "pub", "bar", "biergarten", "copyshop", "hotel", "motel",
    "hostel", "guest_house", "apartment", "subway_entrance", "bus_stop", "platform", "tram_stop",
    "stop_position", "post_depot", "post_office", "parking", "fuel", "bank", "atm", "pharmacy",
    "path", "footway", "cycleway", "service", "track", "steps", "industrial", "orchard",
    "farmland", "farmyard", "allotments", "garages", "construction", "artwork", "company",
    "apartments", "detached", "commercial", "retail",
}


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


def _is_relevant_result(location):
    """False for Nominatim results that cannot be depicted (shops, stops, plain buildings...)."""
    if location.get("source") != "Nominatim":
        return True
    place_class, place_type = location.get("place_class"), location.get("feature_code")
    if place_class in _IRRELEVANT_CLASSES or (place_class == "building" and place_type == "yes"):
        return False
    return place_type not in _IRRELEVANT_TYPES


def _same_place(geonames_location, nominatim_location):
    """True if a trusted GeoNames entry and a Nominatim result describe the same place."""
    country_code = geonames_location.get("country_code")
    if not country_code or country_code != nominatim_location.get("country_code"):
        return False
    feature_code = str(geonames_location.get("feature_code") or "")
    if feature_code.startswith("PCL") or feature_code == "TERR":
        return _is_country(nominatim_location)
    radius = _SAME_ADMINISTRATIVE_UNIT_KM if feature_code.startswith("ADM") else _SAME_PLACE_KM
    return haversine_distance_km(
        geonames_location["latitude"], geonames_location["longitude"],
        nominatim_location["latitude"], nominatim_location["longitude"],
    ) <= radius


def _rank_by_importance(trusted, nominatim, geonames):
    """Merge trusted GeoNames entries and Nominatim results, most important first.

    A trusted GeoNames entry takes the importance of the Nominatim result
    describing the same place (which is then dropped as a duplicate), or
    `_UNMATCHED_GEONAMES_IMPORTANCE`. A Nominatim result confirmed by any
    GeoNames entry of the same name gets `_CONFIRMATION_BONUS`. The sort is
    stable: GeoNames' order is kept between equal importances (Paris PPLC
    before the Paris ADM3 commune).
    """
    scored = []
    for location in trusted:
        importances = [other.get("importance") or 0.0 for other in nominatim if _same_place(location, other)]
        if importances:
            scored.append((max(importances), {**location, "importance": max(importances)}))
        else:
            scored.append((_UNMATCHED_GEONAMES_IMPORTANCE, location))
    for location in nominatim:
        if any(_same_place(other, location) for other in trusted):
            continue
        confirmed = any(
            haversine_distance_km(
                location["latitude"], location["longitude"], other["latitude"], other["longitude"],
            ) <= _CONFIRMATION_KM
            for other in geonames
        )
        scored.append(((location.get("importance") or 0.0) + _CONFIRMATION_BONUS * confirmed, location))
    return [location for _, location in sorted(scored, key=lambda item: -item[0])]


def location_alternatives(name, conn, nominatim_client=None):
    """Ranked candidate locations for a place name, without hierarchy lookup.

    Trusted GeoNames matches (countries included, see `_country_index`) and
    Nominatim's results (shops, stops and other `_is_relevant_result`
    failures removed) are ranked together by Nominatim importance
    (`_rank_by_importance`), so that a famous place beats its homonyms
    whatever the source: Patmos the Greek island, not Patmos (Arkansas);
    Normandy the French region, not Normandy (Missouri). Without Nominatim,
    trusted GeoNames matches come first, most populated first. Untrusted
    GeoNames matches are used only when nothing else is found.
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
    nominatim = nominatim_client.search(name) if nominatim_client is not None else []
    nominatim = [location for location in nominatim if _is_relevant_result(location)]
    if not nominatim:
        return trusted or geonames
    return _rank_by_importance(trusted, nominatim, geonames)


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
# Since the top-down resolution, only used to order anchors (closest first).
CONTEXT_WINDOW_CHARS = 80
# A precise place "fits" a city anchor within this distance, a region anchor
# within REGION_RADIUS_KM (and the same country).
CITY_RADIUS_KM = 30.0
# Large: states are long (Recife is 330 km from the centre of Pernambuco).
REGION_RADIUS_KM = 500.0
# Half-size in degrees of the box searched around a city (about 30 km) or a
# region anchor; a country anchor restricts the search to the country.
_CITY_BOX_DEGREES = 0.3
_REGION_BOX_DEGREES = 2.0
# Words before a mention saying it is where the artwork or its author comes
# from ("printed in London", "born in Leiden"): such places are not anchors.
ORIGIN_CUES = [
    "born in", "died in", "lived in", "living in", "studied in", "worked in", "working in", "trained in",
    "made in", "printed in", "published in", "published by", "manufactured in", "produced in",
    "exhibited in", "exhibition in", "acquired in", "purchased in", "settled in", "moved to",
    "returned to", "emigrated to", "né à", "née à", "mort à", "morte à", "imprimé à", "publié à",
    "fabriqué à", "geboren in", "gestorben in", "gedruckt in",
]
_ORIGIN_CUE_PATTERN = re.compile(
    r"(?:^|[^\w])(?:" + "|".join(sorted(map(re.escape, ORIGIN_CUES), key=len, reverse=True)) + r")\s*$"
)
# Words that do not name a place on their own: a mention made only of them
# ("Park", "Town Hall", "Grove of Trees", "South Gate of") is searched only
# inside the area of an anchor, never worldwide.
GENERIC_PLACE_WORDS = {
    "the", "of", "a", "an", "and", "de", "du", "des", "la", "le", "les", "l", "d", "di", "del", "der",
    "die", "das", "van", "von", "el", "at", "on", "in", "by", "near",
    "park", "forest", "wood", "woods", "garden", "gardens", "grove", "tree", "trees", "field", "fields",
    "meadow", "river", "lake", "pond", "sea", "bay", "coast", "beach", "shore", "island", "mountain",
    "mountains", "hill", "hills", "valley", "cliff", "cliffs", "rock", "falls", "harbor", "harbour",
    "port", "pier", "quay", "canal", "bridge", "gate", "tower", "wall", "walls", "column", "memorial",
    "monument", "fountain", "statue", "church", "chapel", "cathedral", "abbey", "temple", "castle",
    "palace", "fort", "hall", "house", "home", "room", "living", "window", "street", "road", "avenue",
    "lane", "path", "square", "place", "market", "station", "town", "city", "village", "district",
    "quarter", "business", "center", "centre", "museum", "gallery", "school", "college", "university",
    "royal", "old", "new", "great", "little", "small", "big", "upper", "lower", "central", "north",
    "south", "east", "west", "wooden", "entrance", "main", "high",
}
# Two places closer than this are considered to belong to the same area.
NEARBY_KM = 150.0
# Finest Nominatim place_rank of an anchor (16 city, 19 village, 20 hamlet).
_ANCHOR_MAX_PLACE_RANK = 20
# A place at least this important (Rome 0.86, Paris 0.90) is not replaced by
# a "name, context" search: "Rome, France" finds a metro station in Paris.
_PROMINENT_IMPORTANCE = 0.6


def _name_tokens(name):
    """Lowercase words of a name, without accents or punctuation."""
    text = unicodedata.normalize("NFKD", (name or "").casefold())
    text = "".join(character if character.isalnum() else " " for character in text if not unicodedata.combining(character))
    return set(text.split())


def _is_anchor(name, location):
    """True if a mention is reliable enough to give context to its neighbours.

    Anchors are countries, regions and inhabited places whose name matches
    the mention ("Peru" -> "Republic of Peru", "Illinois"), or that are
    prominent under a local name ("Florence" -> "Firenze"). Other mentions,
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
    # Nominatim names places in the local language: "Florence" -> "Firenze".
    # A prominent place is trusted under another name.
    exonym = (location.get("importance") or 0.0) >= _PROMINENT_IMPORTANCE
    return reliable_level and (names_match or exonym)


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


_ARTICLES = {"the", "of", "a", "an", "de", "du", "des", "la", "le", "les", "l", "d", "di", "del", "der", "die", "das"}


def _names_the_mention(name, location):
    """True if every word of the mention (articles aside) is in the location's name.

    Guards searches restricted to an area, which return anything found
    there: "Cardiff Castle" must not become "Cardiff Avenue", nor "Theatre"
    the "Nürnberger Symphoniker".
    """
    mention_tokens = _name_tokens(name) - _ARTICLES
    return bool(mention_tokens) and mention_tokens <= _name_tokens(location.get("name"))


def _is_generic_name(name):
    """True if every word of `name` is a GENERIC_PLACE_WORDS ("Town Hall", "Grove of Trees")."""
    tokens = _name_tokens(name)
    return bool(tokens) and tokens <= GENERIC_PLACE_WORDS


def _origin_mentions_only(occurrences):
    """True if every occurrence follows an origin cue ("printed in London")."""
    return all(
        _ORIGIN_CUE_PATTERN.search(
            " ".join(occurrence["source_text"][max(0, occurrence["start"] - 40):occurrence["start"]].lower().split())
        )
        for occurrence in occurrences
    )


def _anchor_scale(location):
    """"country", "region" or "city": the area an anchor stands for."""
    if _is_country(location):
        return "country"
    feature_code = str(location.get("feature_code") or "")
    if location.get("source") == "GeoNames":
        return "region" if feature_code.startswith("ADM") else "city"
    try:
        place_rank = int(location.get("place_rank") or 0)
    except (TypeError, ValueError):
        place_rank = 0
    return "region" if 0 < place_rank <= 12 else "city"


def _fits_anchor(location, anchor):
    """True if a location lies in the area of an anchor (same country, and near a city or region anchor)."""
    country_code = location.get("country_code")
    if not country_code or country_code != anchor.get("country_code"):
        return False
    scale = _anchor_scale(anchor)
    if scale == "country":
        return True
    radius = CITY_RADIUS_KM if scale == "city" else REGION_RADIUS_KM
    return haversine_distance_km(
        location["latitude"], location["longitude"], anchor["latitude"], anchor["longitude"],
    ) <= radius


def _anchor_area(anchor):
    """(country_code, viewbox or None) searched inside an anchor."""
    scale = _anchor_scale(anchor)
    if scale == "country":
        return anchor["country_code"], None
    half_size = _CITY_BOX_DEGREES if scale == "city" else _REGION_BOX_DEGREES
    return anchor["country_code"], (
        anchor["longitude"] - half_size, anchor["latitude"] - half_size,
        anchor["longitude"] + half_size, anchor["latitude"] + half_size,
    )


def _text_distance(occurrences_a, occurrences_b):
    """Smallest character distance between two mentions of the same text (inf if never together)."""
    distances = [
        max(a["start"], b["start"]) - min(a["end"], b["end"])
        for a in occurrences_a
        for b in occurrences_b
        if a["field"] == b["field"] and a.get("source_text") == b.get("source_text")
    ]
    return min(distances, default=math.inf)


def _contextual_search(name, context_names, context_locations, nominatim_client):
    """Search "name, context" with Nominatim and keep a result in the context's country."""
    if nominatim_client is None:
        return None
    for context_name, context_location in zip(context_names, context_locations):
        for location in nominatim_client.search(f"{name}, {context_name}"):
            if _is_relevant_result(location) and _agreement(location, context_location) > 0:
                return location
    return None


def _resolved(choice, options, nominatim_client, conflict, resolution):
    """Location returned by resolve_artwork_places: hierarchy, conflict flag, homonym statistics."""
    homonyms = [
        option for option in options
        if haversine_distance_km(
            choice["latitude"], choice["longitude"], option["latitude"], option["longitude"],
        ) > NEARBY_KM
    ]
    return {
        **_with_hierarchy(choice, nominatim_client),
        "context_conflict": conflict,
        "resolution": resolution,
        "homonym_count": len(homonyms),
        "importance_gap": (choice.get("importance") or 0.0) - max(
            (homonym.get("importance") or 0.0 for homonym in homonyms), default=0.0
        ),
    }


def resolve_artwork_places(mentions, conn, nominatim_client=None, alternatives_cache=None):
    """Resolve the places of one artwork from the largest to the most precise.

    `mentions` is a list of (name, occurrences) where occurrences carry
    `field`, `start`, `end` and `source_text`.

    1. Anchors: countries, regions and inhabited places whose name matches
       the mention (`_is_anchor`), anywhere in the artwork, except mentions
       that only follow an origin cue ("printed in London"). An anchor takes
       the alternative that best agrees with the other anchors (same
       country, nearby); if none agrees, "name, context" is searched, unless
       the first alternative is prominent (`_PROMINENT_IMPORTANCE`); if that
       fails too, the first alternative is kept and flagged `context_conflict`.
    2. Other mentions (monuments, streets, misread words): with anchors, the
       first alternative lying in an anchor's area (`_fits_anchor`, closest
       anchor in the text first); otherwise the name is searched inside the
       anchors' areas (`NominatimClient.search_within`), then as "name,
       anchor", keeping only results that carry the mention's words
       (`_names_the_mention`). If all fail and an anchor is right next to the
       mention ("Le Nys, Amsterdam"), the mention gets no location unless its
       first alternative is prominent: the artwork is then placed by its
       anchors ("Palazzo Vecchio, Florence" -> Florence) rather than by a
       homonym elsewhere. With only distant anchors, the first alternative is
       kept; both kept cases are flagged `context_conflict`. Without anchors,
       the first alternative, except for generic names (`_is_generic_name`:
       "Park", "Town Hall"), which get none. Prominent anchors are only
       revised by country or region anchors; generic names are never anchors.

    Returns {name: location or None}; locations include `context_conflict`,
    `resolution`, `homonym_count` (alternatives farther than NEARBY_KM from
    the choice) and `importance_gap` (importance of the choice minus that of
    the most important of those homonyms; the choice's importance if none).
    """
    alternatives_cache = {} if alternatives_cache is None else alternatives_cache

    def alternatives(name):
        """Cached `location_alternatives` of a name."""
        key = name.casefold().strip()
        if key not in alternatives_cache:
            alternatives_cache[key] = location_alternatives(name, conn, nominatim_client)
        return alternatives_cache[key]

    anchor_mentions = [
        (name, occurrences)
        for name, occurrences in mentions
        if alternatives(name)
        and _is_anchor(name, alternatives(name)[0])
        and not _is_generic_name(name)
        and not _origin_mentions_only(occurrences)
    ]
    anchor_names = {name for name, _ in anchor_mentions}
    resolved = {}

    for name, occurrences in anchor_mentions:
        options = alternatives(name)
        context = [
            (other_name, alternatives(other_name)[0])
            for other_name, _ in anchor_mentions
            if other_name != name
        ]
        choice, conflict = options[0], False
        # A prominent place (Paris, Berlin) is only revised by a country or a
        # region: a misread city-level anchor ("Red House" -> Red House,
        # Virginia) would otherwise drag it to a homonym.
        if (options[0].get("importance") or 0.0) >= _PROMINENT_IMPORTANCE:
            context = [(n, location) for n, location in context if _anchor_scale(location) != "city"]
        # A country name is not ambiguous enough to be revised by smaller places.
        if context and not _is_country(options[0]):
            scores = [sum(_agreement(option, location) for _, location in context) for option in options]
            best_index = max(range(len(options)), key=lambda index: (scores[index], -index))
            if scores[best_index] > 0:
                choice = options[best_index]
            elif (options[0].get("importance") or 0.0) >= _PROMINENT_IMPORTANCE:
                conflict = True
            else:
                contextual = _contextual_search(
                    name, [n for n, _ in context], [loc for _, loc in context], nominatim_client
                )
                choice, conflict = (contextual, False) if contextual else (options[0], True)
        resolved[name] = _resolved(choice, options, nominatim_client, conflict, "anchor")

    search_within = getattr(nominatim_client, "search_within", None)
    for name, occurrences in mentions:
        if name in anchor_names:
            continue
        options = alternatives(name)
        anchors = sorted(
            (
                (other_name, other_occurrences)
                for other_name, other_occurrences in anchor_mentions
                if resolved.get(other_name)
            ),
            key=lambda item: _text_distance(occurrences, item[1]),
        )
        anchor_locations = [resolved[other_name] for other_name, _ in anchors]
        if not anchor_locations:
            resolved[name] = (
                None if not options or _is_generic_name(name)
                else _resolved(options[0], options, nominatim_client, False, "global")
            )
            continue

        fitting = next(
            (option for anchor in anchor_locations for option in options if _fits_anchor(option, anchor)),
            None,
        )
        if fitting is not None:
            resolved[name] = _resolved(fitting, options, nominatim_client, False, "in_anchor")
            continue
        if search_within is not None:
            bounded = next(
                (
                    location
                    for anchor in anchor_locations
                    if anchor.get("country_code")
                    for location in search_within(name, *_anchor_area(anchor))
                    if _is_relevant_result(location)
                    and _fits_anchor(location, anchor)
                    and _names_the_mention(name, location)
                ),
                None,
            )
            if bounded is not None:
                resolved[name] = _resolved(bounded, [bounded, *options], nominatim_client, False, "searched_in_anchor")
                continue
        # A free-text "Grove of Trees, Versailles" search finds unrelated
        # memorials: generic names only use the bounded search above.
        contextual = None if _is_generic_name(name) else _contextual_search(
            name, [n for n, _ in anchors], anchor_locations, nominatim_client
        )
        if contextual is not None and _names_the_mention(name, contextual):
            resolved[name] = _resolved(contextual, [contextual, *options], nominatim_client, False, "name_and_anchor")
            continue
        # Not found in the anchors' areas. An anchor right next to the mention
        # ("Le Nys, Amsterdam") says where it is: no location rather than a
        # homonym elsewhere. A distant anchor (another sentence) may concern
        # something else: keep the best guess, flagged as a conflict.
        nearby_anchor = _text_distance(occurrences, anchors[0][1]) <= CONTEXT_WINDOW_CHARS
        prominent = (options[0].get("importance") or 0.0) >= _PROMINENT_IMPORTANCE if options else False
        keep = options and not _is_generic_name(name) and (prominent or not nearby_anchor)
        resolved[name] = (
            _resolved(options[0], options, nominatim_client, True, "outside_anchors") if keep else None
        )
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
