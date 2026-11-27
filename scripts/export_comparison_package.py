"""Export the gold and the model's out-of-fold predictions for an external comparison.

A colleague tests another approach (from the images) on the same artworks.
For a fair comparison, each gold artwork must be predicted by a model that
did not see it during training: the predictions exported here come from
the outer grouped cross-validation (5 folds, stable partition of seed 0,
the same as the first of the 20 partitions of compare_tables.py), each fold
model being trained on the other folds only.

For every model two decisions are exported:
- "all levels": the most specific level whose threshold is crossed, among
  every level that has a threshold in the fold model (what the CV metrics
  measure);
- "published": the same choice restricted to the levels that pass rule D
  (read from --deployment-json, the compare_tables.py output), i.e. what
  would actually be published (today: country only).

Output (--output-dir): gold_and_predictions.csv (one row per artwork),
gold_and_predictions.json (same content) and README.md (fields, protocol,
summary metrics).

    python scripts/export_comparison_package.py
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.filtering_logreg import (  # noqa: E402
    _SPECIFICITY,
    _select_hierarchical_level,
    evaluate_hierarchical_grouped_cv,
)
from geolocation.gazetter import NominatimClient  # noqa: E402
from run_pipeline import load_artwork_records  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_tables import TRAINING_OPTIONS  # noqa: E402

AIC_IMAGE = "https://www.artic.edu/iiif/2/{}/full/843,/0/default.jpg"
AIC_PAGE = "https://www.artic.edu/artworks/{}"


def _origin(artwork_id):
    """Collection the gold record comes from."""
    artwork_id = str(artwork_id)
    if artwork_id.isdigit():
        return "aic"
    if artwork_id.startswith("Q"):
        return "wikidata"
    return "dresden_csv" if "dresden" in artwork_id else ("venice_csv" if "venice" in artwork_id else "other")


def _target(record):
    """Annotated depicted place of a record, or None."""
    annotations = record.get("annotations", {})
    targets = annotations.get("depicted_places") or [annotations.get("depicted_place")]
    return next((target for target in targets if target), None)


def collect_predictions(records, table, options, seed):
    """{artwork_id: (fold, test rows, scores_by_row, levels)} from the outer CV of one model."""
    collected = {}
    fold_counter = [0]

    def keep(test_table, scores_by_row, model):
        fold_counter[0] += 1
        for artwork_id, rows in test_table.groupby("artwork_id"):
            collected[str(artwork_id)] = (fold_counter[0], rows, scores_by_row, model["levels"])

    metrics = evaluate_hierarchical_grouped_cv(
        records, table, n_splits=5, training_options=options, shuffle_seed=seed, fold_callback=keep
    )
    # Test artworks without any candidate: no prediction, but they were tested.
    for fold in metrics["folds"]:
        for artwork_id in fold["test_artwork_ids"]:
            collected.setdefault(str(artwork_id), (fold["fold"], None, None, None))
    return collected, metrics


def decide(rows, scores_by_row, levels, allowed_levels=None):
    """(level, row) chosen for one artwork, as in evaluate_hierarchical_work_predictions, or (None, None)."""
    if rows is None or levels is None:
        return None, None
    if allowed_levels is not None:
        levels = {
            level: entry if level in allowed_levels else {**entry, "status": "not_deployed"}
            for level, entry in levels.items()
        }
    decisions = []
    # The test table of a fold is re-indexed from 0: its index is the row
    # position that _score_rows uses as key.
    for index, row in rows.iterrows():
        row_scores = scores_by_row.get(index, {})
        level = _select_hierarchical_level(row.get("candidate_level"), row_scores, levels, _SPECIFICITY)
        if level is not None:
            decisions.append((_SPECIFICITY[level], -row_scores[level], index, level))
    if not decisions:
        return None, None
    _, _, index, level = min(decisions)
    return level, rows.loc[index]


def _correct(row, level):
    """True if the chosen candidate is right at the chosen level (same rule as the CV metrics)."""
    return bool(
        row is not None
        and row.get("label_status") == "match"
        and row.get("label_level") in _SPECIFICITY
        and _SPECIFICITY[row.get("label_level")] <= _SPECIFICITY[level]
    )


def main():
    """Run the outer CV of each model, then write the CSV, JSON and README."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", default="data/raw/gold_standard/gold_dataset.json")
    parser.add_argument("--table", default="results/feature_table_v15_llm.pkl")
    parser.add_argument("--hybrid-options", default="results/llm/hybrid_options.json")
    parser.add_argument("--deployment-json", default="results/compare_v15_hybrid.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/export_comparison"))
    args = parser.parse_args()

    records = load_artwork_records(args.records)
    table = pd.read_pickle(args.table)
    deployment = json.loads(Path(args.deployment_json).read_text(encoding="utf-8"))
    models = {
        "v15": (dict(TRAINING_OPTIONS), deployment["v15"]["summary"]["deployed"]),
        "hybrid": (
            {**TRAINING_OPTIONS, **json.loads(Path(args.hybrid_options).read_text(encoding="utf-8"))},
            deployment["v15+llm-fin"]["summary"]["deployed"],
        ),
    }

    predictions, summaries = {}, {}
    for name, (options, deployed) in models.items():
        print(f"[info] validation croisée externe de {name} (graine {args.seed})...", flush=True)
        predictions[name], summaries[name] = collect_predictions(records, table, options, args.seed)
        summaries[name]["deployed_levels"] = deployed

    reverse = NominatimClient()
    rows = []
    for record in records:
        artwork = record["artwork"]
        artwork_id = str(artwork.get("id"))
        annotations = record.get("annotations", {})
        source = annotations.get("ground_truth_source") or {}
        target = _target(record)
        hierarchy = {}
        if target and target.get("latitude") is not None:
            hierarchy = (reverse.reverse(target["latitude"], target["longitude"]) or {}).get("hierarchy") or {}
        origin = _origin(artwork_id)
        row = {
            "artwork_id": artwork_id,
            "collection": origin,
            "title": artwork.get("title"),
            "artist_display": artwork.get("artist_display"),
            "date_display": artwork.get("date_display"),
            "artwork_type": artwork.get("artwork_type_title"),
            "place_of_origin": artwork.get("place_of_origin"),
            "image_url": AIC_IMAGE.format(artwork["image_id"]) if origin == "aic" and artwork.get("image_id") else artwork.get("image_url"),
            "page_url": AIC_PAGE.format(artwork_id) if origin == "aic" else (
                f"https://www.wikidata.org/wiki/{artwork_id}" if origin == "wikidata" else artwork.get("wikidata_url")
            ),
            "is_public_domain": artwork.get("is_public_domain"),
            "gold_place_status": annotations.get("place_status"),
            "gold_place_name": target.get("name") if target else None,
            "gold_precision": target.get("precision") if target else None,
            "gold_latitude": target.get("latitude") if target else None,
            "gold_longitude": target.get("longitude") if target else None,
            "gold_city": hierarchy.get("city"),
            "gold_region": hierarchy.get("region"),
            "gold_country_code": hierarchy.get("country_code"),
            "gold_sample_stratum": source.get("sample_stratum"),
        }
        for name, (_, deployed) in models.items():
            fold, artwork_rows, scores_by_row, levels = predictions[name].get(artwork_id, (None, None, None, None))
            row[f"{name}_cv_fold"] = fold
            row[f"{name}_evaluated"] = fold is not None and annotations.get("place_status") != "undeterminable"
            for suffix, allowed in (("all_levels", None), ("published", set(deployed))):
                level, chosen = decide(artwork_rows, scores_by_row, levels, allowed)
                location = (chosen.get("candidate_geolocation") if chosen is not None else None) or {}
                row[f"{name}_{suffix}_level"] = level
                row[f"{name}_{suffix}_place"] = chosen.get("place") if chosen is not None else None
                row[f"{name}_{suffix}_resolved_name"] = location.get("name")
                row[f"{name}_{suffix}_latitude"] = location.get("latitude")
                row[f"{name}_{suffix}_longitude"] = location.get("longitude")
                row[f"{name}_{suffix}_correct"] = _correct(chosen, level) if level else None
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "gold_and_predictions.csv", index=False)
    frame.to_json(args.output_dir / "gold_and_predictions.json", orient="records", force_ascii=False, indent=1)

    print("\n=== Résumé (une partition, graine", args.seed, ") ===")
    for name in models:
        evaluated = frame[frame[f"{name}_evaluated"]]
        for suffix in ("all_levels", "published"):
            predicted = evaluated[evaluated[f"{name}_{suffix}_level"].notna()]
            correct = predicted[f"{name}_{suffix}_correct"].astype(bool) & (predicted["gold_place_status"] == "depicted")
            depicted = (evaluated["gold_place_status"] == "depicted").sum()
            print(f"  {name:6s} {suffix:10s} prédictions={len(predicted):3d} justes={int(correct.sum()):3d} "
                  f"précision={correct.mean() if len(predicted) else float('nan'):.3f} rappel={correct.sum() / depicted:.3f} "
                  f"niveaux={dict(Counter(predicted[f'{name}_{suffix}_level']))}")
    print(f"[ok] {args.output_dir}/gold_and_predictions.csv ({len(frame)} œuvres)")


if __name__ == "__main__":
    main()
