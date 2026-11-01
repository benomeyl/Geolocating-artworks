"""Run the hierarchical model on a random AIC sample and export a review CSV.

With --keywords, the sample is drawn only among artworks whose title,
subjects or themes contain one of the keywords (config/keywords.json).
"""

import argparse
import csv
import json
import random
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api.aic import enrich_records_from_dump  # noqa: E402
from data.filtering_logreg import (  # noqa: E402
    extract_all_candidates,
    filter_by_keywords,
    model_uses_nli,
    predict_with_hierarchical_model,
)
from data.nli import NliScorer  # noqa: E402
from geolocation.gazetter import NominatimClient  # noqa: E402
from run_pipeline import load_ner_pipeline  # noqa: E402


def load_records(path):
    """Load the records of a JSON list or of a {"unique_artworks": [...]} envelope."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload.get("unique_artworks", []) if isinstance(payload, dict) else payload


def sample_records(records, size, seed, excluded_ids=()):
    """Draw a reproducible random sample of `size` records, excluding `excluded_ids`."""
    excluded = {str(artwork_id) for artwork_id in excluded_ids}
    eligible = [
        record for record in records
        if str((record.get("artwork", record)).get("id")) not in excluded
    ]
    eligible.sort(key=lambda record: str((record.get("artwork", record)).get("id")))
    return random.Random(seed).sample(eligible, min(size, len(eligible)))


KEYWORD_FIELDS = ("title", "alt_titles", "subject_titles", "theme_titles")


def _keyword_text(record):
    """Text searched for keywords: titles, subjects and themes (descriptions are too noisy)."""
    artwork = record.get("artwork", record)
    parts = []
    for field in KEYWORD_FIELDS:
        value = artwork.get(field)
        parts.extend(value if isinstance(value, list) else [value] if value else [])
    return " | ".join(str(part) for part in parts)


def keyword_matches(records, keywords):
    """{artwork_id: matched keywords} for the records matching at least one keyword."""
    texts = {(record.get("artwork", record)).get("id"): _keyword_text(record) for record in records}
    return {
        match["id"]: match["matched_keywords"]
        for match in filter_by_keywords(texts, keywords)["filtered_artworks"]
    }


def _best_abstention(candidates, model):
    """Candidate/level closest to its threshold, to show near misses."""
    best = None
    for candidate in candidates:
        for level, score in (candidate.get("level_scores") or {}).items():
            margin = score - model["levels"][level]["threshold"]
            if best is None or margin > best[0]:
                best = (margin, candidate, level, score)
    return best


def prediction_row(record, result, model, matched_keywords=None, validated_levels=None):
    """Flatten the prediction of one artwork into a CSV row, with links and review columns.

    `validated_levels` lists the levels that passed the outer validation; it
    is only needed when the model was run on all its active levels.
    """
    artwork = record.get("artwork", record)
    candidates = result.get("candidates", [])
    prediction = result.get("prediction")
    shown = prediction or (candidates[0] if candidates else None)
    geolocation = (shown or {}).get("geolocation") or {}
    hierarchy = geolocation.get("hierarchy") or {}
    decision = (prediction or {}).get("decision") or {}
    near_miss = None if prediction else _best_abstention(candidates, model)
    image_id = artwork.get("image_id")
    return {
        "artwork_id": artwork.get("id"),
        "aic_url": f"https://www.artic.edu/artworks/{artwork.get('id')}",
        "image_url": f"https://www.artic.edu/iiif/2/{image_id}/full/843,/0/default.jpg" if image_id else "",
        "title": artwork.get("title"),
        "place_of_origin": artwork.get("place_of_origin"),
        "artist_display": " | ".join((artwork.get("artist_display") or "").splitlines()),
        "has_description": bool(artwork.get("description")),
        "prediction_status": "depicted" if prediction else ("uncertain" if candidates else "no_candidate"),
        "predicted_level": decision.get("level"),
        "level_validated": (
            None if not decision else validated_levels is None or decision.get("level") in validated_levels
        ),
        "place": (shown or {}).get("place"),
        "candidate_level": (shown or {}).get("candidate_level"),
        "resolved_name": geolocation.get("name"),
        "latitude": geolocation.get("latitude"),
        "longitude": geolocation.get("longitude"),
        "resolved_city": hierarchy.get("city"),
        "resolved_region": hierarchy.get("region"),
        "resolved_country": hierarchy.get("country"),
        "model_score": decision.get("model_score"),
        "empirical_precision": decision.get("confidence"),
        "near_miss_place": near_miss[1]["place"] if near_miss else None,
        "near_miss_level": near_miss[2] if near_miss else None,
        "near_miss_margin": round(near_miss[0], 3) if near_miss else None,
        "candidate_count": len(candidates),
        "candidates": " | ".join(
            f"{candidate['place']} [{candidate.get('candidate_level') or '?'}]"
            for candidate in candidates
        ),
        "subject_titles": json.dumps(artwork.get("subject_titles") or [], ensure_ascii=False),
        "matched_keywords": ", ".join(matched_keywords or []),
        "review_is_correct": "",
        "review_notes": "",
    }


def main():
    """Sample artworks, run NER, geocoding and the model, and write the CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=Path("data/raw/aic/aic_useful_fields.json"))
    parser.add_argument("--model", type=Path, default=Path("results/model_hierarchical.joblib"))
    parser.add_argument("--gold-json", type=Path, default=Path("data/raw/gold_standard/gold_dataset.json"))
    parser.add_argument("--output", type=Path, default=Path("results/test_sample_500.csv"))
    parser.add_argument("--size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gazetteer", default="data/gazetter/geonames.db")
    parser.add_argument("--aic-dump-dir", default=None)
    parser.add_argument("--nominatim-user-agent", default=None)
    parser.add_argument("--geocode-top-k", type=int, default=3)
    parser.add_argument(
        "--keywords",
        type=Path,
        default=None,
        help="JSON de mots-clés (ex. config/keywords.json): ne tirer que les œuvres dont le titre, "
        "les sujets ou les thèmes en contiennent un.",
    )
    parser.add_argument(
        "--use-active-levels",
        action="store_true",
        help="Utiliser tous les niveaux actifs, même non validés par la validation externe "
        "(colonne level_validated). Pour exploration uniquement.",
    )
    args = parser.parse_args()

    model = joblib.load(args.model)
    validated_levels = None
    if args.use_active_levels:
        validated_levels = list(model.get("deployable_levels") or [])
        model = {**model, "deployable_levels": None}
    gold_ids = [
        (record.get("artwork", record)).get("id") for record in load_records(args.gold_json)
    ]
    records = load_records(args.records)
    matches = {}
    if args.keywords:
        matches = keyword_matches(records, json.loads(args.keywords.read_text(encoding="utf-8")))
        records = [record for record in records if (record.get("artwork", record)).get("id") in matches]
        print(f"[info] {len(records)} œuvres contiennent au moins un mot-clé.")
    records = sample_records(records, args.size, args.seed, gold_ids)
    print(f"[info] {len(records)} œuvres tirées (seed={args.seed}), hors gold standard.")
    print(f"[info] {enrich_records_from_dump(records, args.aic_dump_dir)} œuvres enrichies depuis le dump AIC.")

    candidates_by_artwork = extract_all_candidates(records, load_ner_pipeline())
    gazetteer_conn = sqlite3.connect(args.gazetteer) if Path(args.gazetteer).is_file() else None
    try:
        results = predict_with_hierarchical_model(
            records,
            candidates_by_artwork,
            model,
            gazetteer_conn=gazetteer_conn,
            nominatim_client=NominatimClient(user_agent=args.nominatim_user_agent),
            geocode_top_k=args.geocode_top_k,
            nli_scorer=NliScorer() if model_uses_nli(model) else None,
        )
    finally:
        if gazetteer_conn is not None:
            gazetteer_conn.close()

    rows = [
        prediction_row(
            record,
            results.get((record.get("artwork", record)).get("id"), {}),
            model,
            matches.get((record.get("artwork", record)).get("id")),
            validated_levels,
        )
        for record in records
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[info] Statuts: {dict(Counter(row['prediction_status'] for row in rows))}")
    print(f"[info] Niveaux prédits: {dict(Counter(row['predicted_level'] for row in rows if row['predicted_level']))}")
    print(f"[ok] Résultats écrits dans {args.output}")


if __name__ == "__main__":
    main()
