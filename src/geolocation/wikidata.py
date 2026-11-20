"""
Wikidata lookup of monuments that OpenStreetMap does not name.

Nominatim only searches OpenStreetMap names (Wikidata/Wikipedia only feed its
importance ranking), so English descriptive names ("St. Ouen, Rouen",
"Church of St. Etienne du Mont", "Annunciation Bridge") are not found.
Wikidata labels and aliases in several languages often hold them.
`WikidataClient.search_places` returns the items with coordinates found by
a label/alias search (English, French, German) and a full-text search;
settlements (they are anchors, not monuments) and creative works (an
artwork's own item has coordinates) are excluded. The choice among them is
made by gazetter.resolve_artwork_places, inside the area of an anchor.

Wikimedia asks automated clients for a User-Agent with a contact: it is read
from WIKIDATA_USER_AGENT, else from the untracked file
data/processed/wikidata_user_agent.txt, else a generic name (more throttled).
Responses are cached in SQLite; requests are serial, at most one per
`minimum_interval` seconds, with maxlag=5 and Retry-After honoured.
"""

import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path

import requests

_LOGGER = logging.getLogger(__name__)
_API = "https://www.wikidata.org/w/api.php"
_DEFAULT_USER_AGENT = "GeolocatingArtworks/0.1 (EPFL research project)"
_USER_AGENT_FILE = Path("data/processed/wikidata_user_agent.txt")
SEARCH_LANGUAGES = ("en", "fr", "de")
_NAME_LANGUAGES = "en|fr|de|mul|it|es|nl"

# Instance-of classes of settlements and administrative units.
SETTLEMENT_CLASSES = {
    "Q515", "Q486972", "Q1549591", "Q3957", "Q532", "Q15284", "Q262166", "Q484170", "Q747074", "Q1093829",
    "Q5119", "Q6256", "Q3624078", "Q10864048", "Q123705", "Q2983893", "Q1637706", "Q200250", "Q1907114",
    "Q7930989", "Q35657", "Q1221156", "Q2989457", "Q3184121", "Q42744322", "Q1115575", "Q702492",
    "Q15921300", "Q22865", "Q1620908", "Q5084", "Q116457956", "Q1187811", "Q15127012", "Q533724",
    "Q13212489", "Q133442", "Q56061", "Q82794",
}
# Instance-of classes of creative works (paintings, photographs, prints...).
CREATIVE_WORK_CLASSES = {
    "Q3305213", "Q125191", "Q11060274", "Q838948", "Q93184", "Q860861", "Q4502142", "Q179700", "Q18761202",
    "Q15711026", "Q1028181",
}


def _user_agent(user_agent=None):
    """User-Agent: argument, environment, untracked contact file, or the generic default."""
    if user_agent:
        return user_agent
    if os.getenv("WIKIDATA_USER_AGENT"):
        return os.environ["WIKIDATA_USER_AGENT"]
    if _USER_AGENT_FILE.is_file():
        text = _USER_AGENT_FILE.read_text(encoding="utf-8").strip()
        if text:
            return text
    return _DEFAULT_USER_AGENT


class WikidataClient:
    """Cached and throttled client of the Wikidata search and entity API."""

    def __init__(self, *, user_agent=None, cache_path=None, minimum_interval=1.0, timeout=20, session=None):
        self.user_agent = _user_agent(user_agent)
        self.cache_path = Path(
            cache_path or os.getenv("WIKIDATA_CACHE_PATH", "data/processed/wikidata_cache.sqlite")
        )
        self.minimum_interval = minimum_interval
        self.timeout = timeout
        self.session = session or requests.Session()
        self._last_request_at = 0.0
        self._lock = threading.Lock()
        self._country_codes = {}
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.cache_path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS wikidata_cache (query TEXT PRIMARY KEY, response TEXT NOT NULL)"
            )

    def _get(self, params):
        """GET the API with `params` (cached); None if Wikidata keeps failing (not cached)."""
        params = {**params, "format": "json"}
        key = json.dumps(params, sort_keys=True)
        with sqlite3.connect(self.cache_path) as connection:
            row = connection.execute("SELECT response FROM wikidata_cache WHERE query = ?", (key,)).fetchone()
        if row:
            return json.loads(row[0])
        with self._lock:
            for attempt in range(6):
                elapsed = time.monotonic() - self._last_request_at
                if elapsed < self.minimum_interval:
                    time.sleep(self.minimum_interval - elapsed)
                try:
                    response = self.session.get(
                        _API, params={**params, "maxlag": 5}, headers={"User-Agent": self.user_agent},
                        timeout=self.timeout,
                    )
                    self._last_request_at = time.monotonic()
                    if response.status_code == 429 or response.status_code >= 500:
                        wait = int(response.headers.get("Retry-After") or 0) or 10 * (attempt + 1)
                        _LOGGER.warning("Wikidata HTTP %s, attente %s s", response.status_code, wait)
                        time.sleep(min(wait, 120))
                        continue
                    response.raise_for_status()
                    data = response.json()
                except (requests.RequestException, ValueError) as error:
                    self._last_request_at = time.monotonic()
                    _LOGGER.warning("Wikidata request failed (%s), retrying", error)
                    time.sleep(5 * (attempt + 1))
                    continue
                if data.get("error", {}).get("code") == "maxlag":
                    time.sleep(5 * (attempt + 1))
                    continue
                with sqlite3.connect(self.cache_path) as connection:
                    connection.execute("INSERT OR REPLACE INTO wikidata_cache VALUES (?, ?)", (key, json.dumps(data)))
                return data
        _LOGGER.warning("Wikidata unavailable for %s", params)
        return None

    def _search_ids(self, name):
        """Item ids found for `name` by label/alias search (SEARCH_LANGUAGES) and full-text search."""
        ids = []
        for language in SEARCH_LANGUAGES:
            data = self._get({
                "action": "wbsearchentities", "search": name, "language": language, "uselang": language,
                "type": "item", "limit": 10,
            }) or {}
            ids.extend(item["id"] for item in data.get("search", []))
        data = self._get({
            "action": "query", "list": "search", "srsearch": f"{name} haswbstatement:P625",
            "srnamespace": 0, "srlimit": 10,
        }) or {}
        ids.extend(item["title"] for item in data.get("query", {}).get("search", []))
        return list(dict.fromkeys(ids))

    def _entities(self, ids, props="claims|labels|aliases|sitelinks"):
        """Raw entities of `ids`, by id."""
        entities = {}
        for start in range(0, len(ids), 50):
            data = self._get({
                "action": "wbgetentities", "ids": "|".join(ids[start:start + 50]), "props": props,
                "languages": _NAME_LANGUAGES,
            }) or {}
            entities.update(data.get("entities", {}))
        return entities

    def _country_code(self, country_id):
        """ISO 3166-1 alpha-2 code (P297) of a country item, or None."""
        if country_id not in self._country_codes:
            entity = self._entities([country_id], props="claims").get(country_id, {})
            codes = [
                claim["mainsnak"]["datavalue"]["value"]
                for claim in entity.get("claims", {}).get("P297", [])
                if claim.get("mainsnak", {}).get("datavalue")
            ]
            self._country_codes[country_id] = codes[0].upper() if codes else None
        return self._country_codes[country_id]

    def search_places(self, name):
        """Location dicts of the items found for `name` that have coordinates and are not settlements or works.

        Each location carries `names` (labels and aliases in several
        languages) for the name check, `wikidata_id` and `sitelinks`.
        """
        ids = self._search_ids(name)
        if not ids:
            return []
        places = []
        entities = self._entities(ids)
        for qid in ids:
            entity = entities.get(qid) or {}
            claims = entity.get("claims", {})
            coordinates = next(
                (
                    claim["mainsnak"]["datavalue"]["value"]
                    for claim in claims.get("P625", [])
                    if claim.get("mainsnak", {}).get("datavalue")
                ),
                None,
            )
            if coordinates is None:
                continue
            classes = {
                claim["mainsnak"]["datavalue"]["value"]["id"]
                for claim in claims.get("P31", [])
                if claim.get("mainsnak", {}).get("datavalue")
            }
            if classes & SETTLEMENT_CLASSES or "P1082" in claims:
                continue
            if classes & CREATIVE_WORK_CLASSES or "P170" in claims:
                continue
            country_ids = [
                claim["mainsnak"]["datavalue"]["value"]["id"]
                for claim in claims.get("P17", [])
                if claim.get("mainsnak", {}).get("datavalue")
            ]
            labels = entity.get("labels", {})
            names = [value["value"] for value in labels.values()]
            names += [alias["value"] for aliases in entity.get("aliases", {}).values() for alias in aliases]
            label = (labels.get("en") or labels.get("fr") or labels.get("mul") or next(iter(labels.values()), {})).get(
                "value", name
            )
            places.append({
                "name": label,
                "names": list(dict.fromkeys(names)),
                "country_code": self._country_code(country_ids[0]) if country_ids else None,
                "latitude": float(coordinates["latitude"]),
                "longitude": float(coordinates["longitude"]),
                "feature_code": "wikidata_item",
                "place_class": "wikidata",
                "addresstype": None,
                "place_rank": 30,
                "importance": None,
                "wikidata_id": qid,
                "sitelinks": len(entity.get("sitelinks", {})),
                "population": None,
                "hierarchy": None,
                "source": "Wikidata",
                "attribution": "Wikidata (CC0)",
            })
        return places
