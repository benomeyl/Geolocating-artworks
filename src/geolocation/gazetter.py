import math
import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path

import requests


_LOGGER = logging.getLogger(__name__)
_NOMINATIM_ATTRIBUTION = "© OpenStreetMap contributors"


class NominatimClient:
    """Small, cached client for the public Nominatim search endpoint."""

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
        self.endpoint = endpoint or os.getenv(
            "NOMINATIM_URL",
            "https://nominatim.openstreetmap.org/search",
        )
        self.cache_endpoint = f"{self.endpoint}|address-hierarchy-v2"
        self.reverse_endpoint = reverse_endpoint or os.getenv(
            "NOMINATIM_REVERSE_URL",
            self.endpoint.rsplit("/", 1)[0] + "/reverse",
        )
        self.reverse_cache_endpoint = f"{self.reverse_endpoint}|address-hierarchy-v2"
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

    def _cached_response(self, query):
        with sqlite3.connect(self.cache_path) as connection:
            row = connection.execute(
                "SELECT response FROM nominatim_cache WHERE endpoint = ? AND query = ?",
                (self.cache_endpoint, query.casefold().strip()),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def _store_response(self, query, response):
        with sqlite3.connect(self.cache_path) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO nominatim_cache VALUES (?, ?, ?)",
                (self.cache_endpoint, query.casefold().strip(), json.dumps(response)),
            )

    def _normalize_result(self, result, fallback_name):
        try:
            latitude = float(result["lat"])
            longitude = float(result["lon"])
        except (KeyError, TypeError, ValueError):
            return None
        address = result.get("address") or {}
        return {
            "name": result.get("name") or result.get("display_name") or fallback_name,
            "country_code": (address.get("country_code") or "").upper() or None,
            "latitude": latitude,
            "longitude": longitude,
            "feature_code": result.get("type"),
            "place_class": result.get("class"),
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
                "country_code": (address.get("country_code") or "").upper() or None,
            },
            "population": None,
            "source": "Nominatim",
            "attribution": _NOMINATIM_ATTRIBUTION,
        }

    def search(self, query):
        cached = self._cached_response(query)
        if cached is not None:
            return cached

        with self._request_lock:
            cached = self._cached_response(query)
            if cached is not None:
                return cached

            elapsed = time.monotonic() - self._last_request_at
            if elapsed < self.minimum_interval:
                time.sleep(self.minimum_interval - elapsed)

            try:
                response = self.session.get(
                    self.endpoint,
                    params={"q": query, "format": "jsonv2", "limit": 5, "addressdetails": 1},
                    headers={"User-Agent": self.user_agent},
                    timeout=self.timeout,
                )
                self._last_request_at = time.monotonic()
                response.raise_for_status()
                results = response.json()
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
        query = f"reverse:{latitude:.5f},{longitude:.5f}"
        with sqlite3.connect(self.cache_path) as connection:
            row = connection.execute(
                "SELECT response FROM nominatim_cache WHERE endpoint = ? AND query = ?",
                (self.reverse_cache_endpoint, query),
            ).fetchone()
        if row:
            return json.loads(row[0])

        with self._request_lock:
            with sqlite3.connect(self.cache_path) as connection:
                row = connection.execute(
                    "SELECT response FROM nominatim_cache WHERE endpoint = ? AND query = ?",
                    (self.reverse_cache_endpoint, query),
                ).fetchone()
            if row:
                return json.loads(row[0])
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < self.minimum_interval:
                time.sleep(self.minimum_interval - elapsed)
            try:
                response = self.session.get(
                    self.reverse_endpoint,
                    params={
                        "lat": latitude,
                        "lon": longitude,
                        "format": "jsonv2",
                        "addressdetails": 1,
                    },
                    headers={"User-Agent": self.user_agent},
                    timeout=self.timeout,
                )
                self._last_request_at = time.monotonic()
                response.raise_for_status()
                result = response.json()
            except (requests.RequestException, ValueError) as error:
                _LOGGER.warning("Nominatim reverse lookup failed for %s: %s", query, error)
                return None

            location = self._normalize_result(result, query)
            if location is not None:
                with sqlite3.connect(self.cache_path) as connection:
                    connection.execute(
                        "INSERT OR REPLACE INTO nominatim_cache VALUES (?, ?, ?)",
                        (
                            self.reverse_cache_endpoint,
                            query,
                            json.dumps(location),
                        ),
                    )
            return location


def get_location_info(name, conn):
    """
    Get location information from the GeoNames database based on the provided name.

    Args:
        name (str): The name of the location to search for.
        conn (sqlite3.Connection): The SQLite connection object.
    Returns:
        list: A list of tuples containing the location information.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT name, country_code, latitude, longitude, feature_code, population
        FROM geonames
        WHERE name = ? OR asciiname = ?
        ORDER BY population DESC
        LIMIT 5
    """, (name, name))
    return cur.fetchall()


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
        + math.cos(latitude_a)
        * math.cos(latitude_b)
        * math.sin(longitude_delta / 2) ** 2
    )
    return 2 * earth_radius_km * math.asin(math.sqrt(haversine))


def resolve_location(
    name,
    conn,
    reference_coordinates=None,
    nominatim_client=None,
    include_hierarchy=False,
):
    """Try GeoNames first, then Nominatim if GeoNames has no matching place."""
    results = get_location_info(name, conn) if conn is not None else []
    if results:
        locations = [
            {
                "name": result[0],
                "country_code": result[1],
                "latitude": float(result[2]),
                "longitude": float(result[3]),
                "feature_code": result[4],
                "population": result[5],
                "source": "GeoNames",
                "hierarchy": {
                    "neighborhood": None,
                    "city": None,
                    "region": None,
                    "country": None,
                    "country_code": result[1],
                },
            }
            for result in results
        ]
    elif nominatim_client is not None:
        locations = nominatim_client.search(name)
    else:
        return None

    if not locations:
        return None

    if reference_coordinates is not None:
        reference_latitude, reference_longitude = reference_coordinates
        result = min(
            locations,
            key=lambda location: haversine_distance_km(
                reference_latitude,
                reference_longitude,
                location["latitude"],
                location["longitude"],
            ),
        )
    else:
        result = locations[0]
    if (
        include_hierarchy
        and result.get("source") == "GeoNames"
        and nominatim_client is not None
    ):
        reverse_result = nominatim_client.reverse(
            result["latitude"],
            result["longitude"],
        )
        if reverse_result:
            result = {
                **result,
                "hierarchy": reverse_result["hierarchy"],
                "hierarchy_source": "Nominatim",
            }
    return result


def geocode_candidates(results, conn=None, nominatim_client=None, limit_per_artwork=3):
    """Attach GeoNames/Nominatim resolution to the highest-ranked candidates."""
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
            location = location_cache[cache_key]
            if location is not None:
                candidate["geolocation"] = location
    return results


def geocode_top_candidates(results, conn=None, nominatim_client=None):
    """Backwards-compatible helper that geocodes only each artwork's top candidate."""
    return geocode_candidates(
        results,
        conn=conn,
        nominatim_client=nominatim_client,
        limit_per_artwork=1,
    )


def validate_location_candidate(loc_candidate, conn):
    """
    Validate the location candidate retrieved from the GeoNames database.

    Args:
        location_info (list): A list of tuples containing the location information.
        conn (sqlite3.Connection): The SQLite connection object.
    Returns:
        list: A list of validated location candidates with additional information.
    """
    results = get_location_info(loc_candidate, conn)

    if not results:
        return {"validate":False, "message": f"No location found for '{loc_candidate}'."}

    best = results[0]
    return {
        "validate": True,
        "name": best[0],
        "country_code": best[1],
        "latitude": best[2],
        "longitude": best[3],
        "feature_code": best[4]
        }

