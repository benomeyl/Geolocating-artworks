"""Create a reproducible, stratified manual-review CSV from predictions."""

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import joblib

GEOGRAPHIC_LEVELS = [
    "exact_place",
    "neighborhood",
    "city",
    "region",
    "country",
]


DEFAULT_SAMPLE_SIZES = {
    "no_ner_candidate": 30,
    "near_threshold": 30,
    "mid_threshold": 20,
    "far_threshold": 20,
    "candidate_ungeocoded": 20,
    "geocoded_level_unclassified": 20,
    "threshold_met_but_not_deployable": 20,
}
MANUAL_REVIEW_FIELDS = [
    "actual_place_status",
    "actual_precision",
    "actual_place_name",
    "actual_latitude",
    "actual_longitude",
    "actual_city",
    "actual_region",
    "actual_country",
    "review_outcome",
    "review_notes",
]
REVIEW_OUTCOMES = [
    "correct_at_predicted_level",
    "correct_but_too_general",
    "wrong_location",
    "false_positive",
    "missed_depicted_place",
    "correct_abstention",
    "not_sure",
]


def load_json(path):
    with Path(path).open(encoding="utf-8") as input_file:
        return json.load(input_file)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _threshold_margin(candidate, model):
    location_level = candidate.get("candidate_level")
    if location_level not in GEOGRAPHIC_LEVELS:
        return None, None, False
    specificity = {level: index for index, level in enumerate(GEOGRAPHIC_LEVELS)}
    deployable_levels = set(model.get("deployable_levels", []))
    margins = []
    passed_undeployable = False
    for level, score in (candidate.get("level_scores") or {}).items():
        level_model = model.get("levels", {}).get(level, {})
        if level_model.get("status") != "active":
            continue
        if specificity[location_level] > specificity[level]:
            continue
        margin = float(score) - float(level_model["threshold"])
        if margin >= 0 and level not in deployable_levels:
            passed_undeployable = True
        if level in deployable_levels:
            margins.append((margin, level, float(score)))
    if not margins:
        return None, None, passed_undeployable
    margin, level, score = max(margins)
    return (margin, level, score), margins, passed_undeployable


def classify_abstention(record, model):
    candidates = record.get("all_candidates") or []
    if not candidates:
        return "no_ner_candidate", None, None, None

    geocoded = [candidate for candidate in candidates if candidate.get("geolocation")]
    if not geocoded:
        return "candidate_ungeocoded", candidates[0], None, None

    scored = [candidate for candidate in geocoded if candidate.get("level_scores")]
    if not scored:
        return "geocoded_level_unclassified", geocoded[0], None, None

    best = None
    passed_undeployable = False
    for candidate in scored:
        margin_info, _, undeployable = _threshold_margin(candidate, model)
        passed_undeployable |= undeployable
        if margin_info is not None and (best is None or margin_info[0] > best[0][0]):
            best = (margin_info, candidate)

    if passed_undeployable:
        matching = next(
            (
                candidate for candidate in scored
                if _threshold_margin(candidate, model)[2]
            ),
            scored[0],
        )
        return "threshold_met_but_not_deployable", matching, None, None

    if best is None:
        return "geocoded_level_unclassified", geocoded[0], None, None
    margin, level, score = best[0]
    candidate = best[1]
    if margin >= -0.10:
        stratum = "near_threshold"
    elif margin >= -0.30:
        stratum = "mid_threshold"
    else:
        stratum = "far_threshold"
    return stratum, candidate, level, score


def _record_context(records_by_id, record_id):
    record = records_by_id.get(record_id, {})
    artwork = record.get("artwork", record)
    return {
        "artwork_source": artwork.get("source"),
        "image_id": artwork.get("image_id"),
        "description": artwork.get("description"),
        "short_description": artwork.get("short_description"),
        "subject_titles": json.dumps(artwork.get("subject_titles") or [], ensure_ascii=False),
        "theme_titles": json.dumps(artwork.get("theme_titles") or [], ensure_ascii=False),
        "style_titles": json.dumps(artwork.get("style_titles") or [], ensure_ascii=False),
        "matched_queries": json.dumps(record.get("matched_queries") or [], ensure_ascii=False),
    }


def _candidate_summary(candidate):
    if not candidate:
        return {}
    geolocation = candidate.get("geolocation") or {}
    hierarchy = geolocation.get("hierarchy") or {}
    decision = candidate.get("decision") or {}
    return {
        "candidate_place": candidate.get("place"),
        "candidate_level": candidate.get("candidate_level"),
        "score_ner_max": candidate.get("score_ner_max"),
        "model_score": decision.get("model_score"),
        "predicted_level": decision.get("level"),
        "empirical_precision": decision.get("confidence"),
        "precision_support_groups": decision.get("confidence_support"),
        "threshold": decision.get("threshold"),
        "geocoder_source": geolocation.get("source"),
        "latitude": geolocation.get("latitude"),
        "longitude": geolocation.get("longitude"),
        "resolved_name": geolocation.get("name"),
        "resolved_city": hierarchy.get("city"),
        "resolved_region": hierarchy.get("region"),
        "resolved_country": hierarchy.get("country"),
        "fields": json.dumps(candidate.get("champs") or [], ensure_ascii=False),
        "all_level_scores": json.dumps(candidate.get("level_scores") or {}, ensure_ascii=False),
    }


def _location_group_key(candidate):
    if not candidate:
        return None
    geolocation = candidate.get("geolocation") or {}
    osm_type = geolocation.get("osm_type")
    osm_id = geolocation.get("osm_id")
    if osm_type and osm_id:
        return f"osm:{osm_type}:{osm_id}"
    latitude = geolocation.get("latitude")
    longitude = geolocation.get("longitude")
    if latitude is not None and longitude is not None:
        return f"coordinates:{float(latitude):.3f},{float(longitude):.3f}"
    place = " ".join(str(candidate.get("place") or "").casefold().split())
    return f"name:{place}" if place else None


def build_review_sample(
    predictions,
    model,
    records,
    *,
    seed=42,
    sample_sizes=None,
):
    sample_sizes = {**DEFAULT_SAMPLE_SIZES, **(sample_sizes or {})}
    records_by_id = {
        (record.get("artwork", record)).get("id"): record
        for record in records
    }
    strata = {name: [] for name in DEFAULT_SAMPLE_SIZES}
    accepted = []

    for prediction in predictions:
        if prediction.get("place_status_predicted") == "depicted":
            accepted.append((prediction, "accepted_prediction", prediction.get("top_candidate")))
        else:
            stratum, review_candidate, _, _ = classify_abstention(prediction, model)
            strata.setdefault(stratum, []).append((prediction, stratum, review_candidate))

    rng = random.Random(seed)
    selected = list(accepted)
    population_counts = {"accepted_prediction": len(accepted)}
    sample_counts = {"accepted_prediction": len(accepted)}
    for stratum in sorted(strata):
        population = sorted(strata[stratum], key=lambda item: str(item[0].get("id")))
        population_counts[stratum] = len(population)
        requested = sample_sizes.get(stratum, 0)
        count = min(requested, len(population))
        chosen = rng.sample(population, count) if count else []
        selected.extend(chosen)
        sample_counts[stratum] = count

    selected.sort(key=lambda item: (item[1], str(item[0].get("id"))))
    location_group_counts = Counter(
        key
        for _, _, candidate in selected
        if (key := _location_group_key(candidate)) is not None
    )

    rows = []
    for review_number, (prediction, stratum, candidate) in enumerate(selected, start=1):
        candidate_info = _candidate_summary(candidate)
        location_group = _location_group_key(candidate)
        rows.append({
            "review_id": f"R{review_number:04d}",
            "sample_stratum": stratum,
            "stratum_population": population_counts.get(stratum, 0),
            "stratum_sample_size": sample_counts.get(stratum, 0),
            "location_group": location_group,
            "same_location_sample_count": location_group_counts.get(location_group, 0),
            "artwork_id": prediction.get("id"),
            "title": prediction.get("title"),
            "prediction_status": prediction.get("place_status_predicted"),
            **candidate_info,
            **_record_context(records_by_id, prediction.get("id")),
            "actual_place_status": "",
            "actual_precision": "",
            "actual_place_name": "",
            "actual_latitude": "",
            "actual_longitude": "",
            "actual_city": "",
            "actual_region": "",
            "actual_country": "",
            "review_outcome": "",
            "review_notes": "",
        })

    metadata = {
        "seed": seed,
        "prediction_count": len(predictions),
        "accepted_predictions_included": len(accepted),
        "sample_sizes_requested": sample_sizes,
        "population_by_stratum": population_counts,
        "sample_by_stratum": sample_counts,
        "selected_count": len(rows),
        "stratum_definitions": {
            "accepted_prediction": "All predictions with place_status_predicted=depicted; all included.",
            "no_ner_candidate": "No candidate extracted by NER.",
            "candidate_ungeocoded": "Candidates exist, but none has a geolocation.",
            "geocoded_level_unclassified": "A geolocation exists, but no candidate has a classifiable geographic level and scores.",
            "threshold_met_but_not_deployable": "A model level threshold is crossed, but that level did not pass outer-CV deployment criteria.",
            "near_threshold": "Best eligible deployable score is within 0.10 below its level threshold.",
            "mid_threshold": "Best eligible deployable score is 0.10 to 0.30 below its level threshold.",
            "far_threshold": "Best eligible deployable score is more than 0.30 below its level threshold.",
        },
        "weighting_note": (
            "Strata are sampled at different rates. Use population_by_stratum / "
            "sample_by_stratum as the design weight when estimating aggregate rates."
        ),
        "review_outcomes": REVIEW_OUTCOMES,
        "actual_place_status_values": ["depicted", "none", "mentioned_only", "undeterminable"],
        "actual_precision_values": ["exact_place", "neighborhood", "city", "region", "country", "unknown"],
    }
    return rows, metadata


def write_review_files(rows, metadata, csv_path, metadata_path):
    csv_path = Path(csv_path)
    metadata_path = Path(metadata_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["review_id", "sample_stratum", "artwork_id", "title"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("results/predictions_echantillon.json"))
    parser.add_argument("--records", type=Path, default=Path("data/raw/aic/search_multiple_queries_all_artworks_20260917T125145Z.json"))
    parser.add_argument("--model", type=Path, default=Path("results/model_hierarchical.joblib"))
    parser.add_argument("--output", type=Path, default=Path("results/manual_review_sample.csv"))
    parser.add_argument("--metadata-output", type=Path, default=Path("results/manual_review_sample_metadata.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-no-ner", type=int, default=30)
    parser.add_argument("--sample-near-threshold", type=int, default=30)
    parser.add_argument("--sample-mid-threshold", type=int, default=20)
    parser.add_argument("--sample-far-threshold", type=int, default=20)
    parser.add_argument("--sample-ungeocoded", type=int, default=20)
    parser.add_argument("--sample-unclassified", type=int, default=20)
    parser.add_argument("--sample-nondeployable", type=int, default=20)
    args = parser.parse_args()

    predictions = load_json(args.predictions)
    payload = load_json(args.records)
    records = payload.get("unique_artworks", payload.get("artworks", [])) if isinstance(payload, dict) else payload
    model = joblib.load(args.model)
    sample_sizes = {
        "no_ner_candidate": args.sample_no_ner,
        "near_threshold": args.sample_near_threshold,
        "mid_threshold": args.sample_mid_threshold,
        "far_threshold": args.sample_far_threshold,
        "candidate_ungeocoded": args.sample_ungeocoded,
        "geocoded_level_unclassified": args.sample_unclassified,
        "threshold_met_but_not_deployable": args.sample_nondeployable,
    }
    rows, metadata = build_review_sample(
        predictions,
        model,
        records,
        seed=args.seed,
        sample_sizes=sample_sizes,
    )
    metadata["predictions_file"] = str(args.predictions)
    metadata["predictions_sha256"] = sha256_file(args.predictions)
    metadata["model_file"] = str(args.model)
    metadata["model_sha256"] = sha256_file(args.model)
    metadata["source_records_file"] = str(args.records)
    metadata["source_records_sha256"] = sha256_file(args.records)
    write_review_files(rows, metadata, args.output, args.metadata_output)
    print(f"[ok] {len(rows)} œuvres exportées vers {args.output}")
    print(f"[ok] Plan d’échantillonnage écrit dans {args.metadata_output}")
    print(f"[info] Population par strate: {metadata['population_by_stratum']}")
    print(f"[info] Échantillon par strate: {metadata['sample_by_stratum']}")


if __name__ == "__main__":
    main()