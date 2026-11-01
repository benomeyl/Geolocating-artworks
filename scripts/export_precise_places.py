"""Export the artworks of a prediction CSV whose candidates are precise places, for annotation.

The artworks of --input-csv (written by predict_sample_csv.py) are run
through NER, geocoding and the hierarchical model again (the Nominatim and
NLI caches make this fast), because the CSV only keeps one candidate per
artwork. Artworks with at least one candidate geocoded at the exact_place
level are kept, one row each, sorted by `p_exact`.

The hierarchical model abstains at the exact level (no threshold reaches its
precision target), so it has no exact-level probability. `p_exact` comes
from a logistic regression fitted like `_fit_level` on the exact-level
labels of the gold table, without threshold: it only orders the rows to
review and is not validated. `p_city` is the v5 model's probability that the
candidate is right at least to the city.

    python scripts/export_precise_places.py --input-csv results/test_keywords_500_v5.csv
"""

import argparse
import csv
import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from api.aic import enrich_records_from_dump  # noqa: E402
from data.filtering_logreg import (  # noqa: E402
    _hierarchical_candidate_features,
    extract_all_candidates,
    group_candidates_by_place,
    predict_with_hierarchical_model,
)
from data.nli import NliScorer  # noqa: E402
from geolocation.gazetter import NominatimClient  # noqa: E402
from predict_sample_csv import load_records  # noqa: E402
from run_pipeline import load_ner_pipeline  # noqa: E402

REVIEW_COLUMNS = [
    "review_place_status",
    "review_precision",
    "review_candidate_correct",
    "review_place_name",
    "review_latitude",
    "review_longitude",
    "review_notes",
]


def fit_exact_ranker(table_path, feature_columns):
    """Logistic regression of the exact level on the gold label table (C=1, as in the AP comparison)."""
    table = pd.read_pickle(table_path)
    usable = table[table["label_status"].isin(["negative", "match"])]
    labels = ((usable["label_status"] == "match") & (usable["label_level"] == "exact_place")).astype(int)
    X = usable[feature_columns].fillna(0).to_numpy(dtype=float)
    return LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(X, labels.to_numpy())


def precise_candidates(record, candidates, result, ranker, feature_columns, nli_scorer):
    """Exact-level candidates of one artwork with their p_exact, best first."""
    artwork = record.get("artwork", record)
    occurrences_by_place = group_candidates_by_place(candidates)
    precise = []
    for candidate in result.get("candidates", []):
        if candidate["candidate_level"] != "exact_place":
            continue
        occurrences = occurrences_by_place[candidate["place"].lower().strip()]
        features = _hierarchical_candidate_features(occurrences, candidate["geolocation"], artwork, nli_scorer)
        row = np.asarray([[features[column] for column in feature_columns]], dtype=float)
        precise.append({**candidate, "p_exact": float(ranker.predict_proba(row)[0][1])})
    return sorted(precise, key=lambda candidate: -candidate["p_exact"])


def annotation_row(record, result, precise):
    """CSV row of one artwork: its best precise candidate, links, context and empty review columns."""
    artwork = record.get("artwork", record)
    best = precise[0]
    location = best["geolocation"] or {}
    hierarchy = location.get("hierarchy") or {}
    latitude, longitude = location.get("latitude"), location.get("longitude")
    prediction = result.get("prediction")
    image_id = artwork.get("image_id")
    return {
        "p_exact": round(best["p_exact"], 3),
        "p_city": round(best["level_scores"].get("city", float("nan")), 3),
        "artwork_id": artwork.get("id"),
        "aic_url": f"https://www.artic.edu/artworks/{artwork.get('id')}",
        "image_url": f"https://www.artic.edu/iiif/2/{image_id}/full/843,/0/default.jpg" if image_id else "",
        "title": artwork.get("title"),
        "artist_display": " | ".join((artwork.get("artist_display") or "").splitlines()),
        "place_of_origin": artwork.get("place_of_origin"),
        "precise_place": best["place"],
        "found_in": ", ".join(best["champs"]),
        "resolved_name": location.get("name"),
        "osm_type": f"{location.get('place_class') or ''}/{location.get('feature_code') or ''}",
        "latitude": latitude,
        "longitude": longitude,
        "map_url": (
            f"https://www.openstreetmap.org/?mlat={latitude}&mlon={longitude}#map=17/{latitude}/{longitude}"
            if latitude is not None else ""
        ),
        "resolved_city": hierarchy.get("city"),
        "resolved_region": hierarchy.get("region"),
        "resolved_country": hierarchy.get("country"),
        "context_conflict": location.get("context_conflict"),
        "other_precise_places": " | ".join(
            f"{candidate['place']} -> {(candidate['geolocation'] or {}).get('name')} ({candidate['p_exact']:.2f})"
            for candidate in precise[1:]
        ),
        "all_candidates": " | ".join(
            f"{candidate['place']} [{candidate.get('candidate_level') or '?'}]"
            for candidate in result.get("candidates", [])
        ),
        "model_prediction": (
            f"{prediction['decision']['level']}: {prediction['place']}" if prediction else "abstention"
        ),
        **{column: "" for column in REVIEW_COLUMNS},
    }


def main():
    """Rerun the model on the artworks of the input CSV and write the precise places to annotate."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-csv", type=Path, default=Path("results/test_keywords_500_v5.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/annotation_precise_keywords_500.csv"))
    parser.add_argument("--records", type=Path, default=Path("data/raw/aic/aic_useful_fields.json"))
    parser.add_argument("--model", type=Path, default=Path("results/model_hierarchical_v5.joblib"))
    parser.add_argument("--table-cache", type=Path, default=Path("results/feature_table.pkl"))
    parser.add_argument("--gazetteer", default="data/gazetter/geonames.db")
    parser.add_argument("--aic-dump-dir", default=None)
    parser.add_argument("--nominatim-user-agent", default=None)
    parser.add_argument("--geocode-top-k", type=int, default=3)
    args = parser.parse_args()

    # All active levels, as predict_sample_csv.py --use-active-levels, to get p_city.
    model = {**joblib.load(args.model), "deployable_levels": None}
    feature_columns = model["feature_columns"]
    ranker = fit_exact_ranker(args.table_cache, feature_columns)

    with args.input_csv.open(encoding="utf-8-sig", newline="") as input_file:
        artwork_ids = [row["artwork_id"] for row in csv.DictReader(input_file)]
    records_by_id = {
        str((record.get("artwork", record)).get("id")): record for record in load_records(args.records)
    }
    records = [records_by_id[artwork_id] for artwork_id in artwork_ids if artwork_id in records_by_id]
    print(f"[info] {len(records)}/{len(artwork_ids)} œuvres du CSV retrouvées.")
    print(f"[info] {enrich_records_from_dump(records, args.aic_dump_dir)} œuvres enrichies depuis le dump AIC.")

    candidates_by_artwork = extract_all_candidates(records, load_ner_pipeline())
    nli_scorer = NliScorer()
    gazetteer_conn = sqlite3.connect(args.gazetteer) if Path(args.gazetteer).is_file() else None
    try:
        results = predict_with_hierarchical_model(
            records,
            candidates_by_artwork,
            model,
            gazetteer_conn=gazetteer_conn,
            nominatim_client=NominatimClient(user_agent=args.nominatim_user_agent),
            geocode_top_k=args.geocode_top_k,
            nli_scorer=nli_scorer,
        )
    finally:
        if gazetteer_conn is not None:
            gazetteer_conn.close()

    rows = []
    for record in records:
        artwork_id = (record.get("artwork", record)).get("id")
        result = results.get(artwork_id, {})
        precise = precise_candidates(
            record, candidates_by_artwork.get(artwork_id, []), result, ranker, feature_columns, nli_scorer
        )
        if precise:
            rows.append(annotation_row(record, result, precise))
    rows.sort(key=lambda row: -row["p_exact"])
    rows = [{"rank": rank, **row} for rank, row in enumerate(rows, start=1)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] {len(rows)} œuvres avec un lieu précis écrites dans {args.output}")


if __name__ == "__main__":
    main()
