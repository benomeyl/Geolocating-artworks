"""Compare label tables or training options on the same outer partitions.

Each variant is a cached label table (results/feature_table_*.pkl, built by
`run_pipeline.py train` or scripts/compare_feature_sets.py) with optional
training options. Every variant is evaluated with evaluate_repeated_grouped_cv
on the same seeds; stable_group_folds gives the same partition to two
tables that differ by a few labels, so the metrics are compared partition by
partition with the first variant. False positives come first: published
predictions must be right, so a variant is only better if it does not add
false positives.

    python scripts/compare_tables.py \
        --variant v11 results/feature_table_v11.pkl \
        --variant v12 results/feature_table_v12.pkl \
        --variant v12-margin10 results/feature_table_v12.pkl '{"threshold_margin": 0.10}'
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.filtering_logreg import (  # noqa: E402
    GEOGRAPHIC_LEVELS,
    GEOGRAPHIC_TARGET_PRECISION,
    evaluate_repeated_grouped_cv,
    wilson_interval,
)
from run_pipeline import _repeated_level_is_deployable, load_artwork_records  # noqa: E402

# Same defaults as `run_pipeline.py train`.
TRAINING_OPTIONS = {
    "n_splits": 5,
    "regularization_values": (0.1, 1.0, 10.0),
    "minimum_recall": 0.5,
    "minimum_positive_groups": 5,
    "minimum_prediction_groups": 5,
    "minimum_predictions": 5,
    "threshold_margin": 0.05,
    "prediction_model": "fold_average",
    "inner_repeats": 5,
    "pooled_levels": True,
}
WORST_CASE_MARGIN = 0.05


def published_counts(run, levels):
    """True and false positives of one partition restricted to the deployed levels."""
    tp = sum(run["metrics_by_level"][level]["true_positives"] for level in levels)
    fp = sum(run["metrics_by_level"][level]["false_positives"] for level in levels)
    return tp, fp


def summarize(name, repeated, minimum_predictions):
    """Work metrics, per-level precision and rule-D deployable levels of one variant."""
    runs = repeated["runs"]
    deployed = [
        level for level in GEOGRAPHIC_LEVELS
        if _repeated_level_is_deployable(
            level, GEOGRAPHIC_TARGET_PRECISION[level], repeated, minimum_predictions, WORST_CASE_MARGIN
        )
    ]
    published = [published_counts(run, deployed) for run in runs]
    published_tp = sum(tp for tp, _ in published)
    published_fp = sum(fp for _, fp in published)
    print(f"\n=== {name} ===")
    for metric in ("precision", "recall", "true_positives", "false_positives"):
        values = [run[metric] for run in runs]
        print(f"  {metric:16s} {np.mean(values):7.3f} ± {np.std(values):.3f}  (min {min(values):.3f}, max {max(values):.3f})")
    print("  niveau          précision moy. (min)   prédictions/découpage   FP total")
    for level in GEOGRAPHIC_LEVELS:
        precisions = [run["metrics_by_level"][level]["precision"] for run in runs]
        known = [value for value in precisions if value is not None]
        predictions = [run["metrics_by_level"][level]["predictions"] for run in runs]
        false_positives = sum(run["metrics_by_level"][level]["false_positives"] for run in runs)
        mean = f"{np.mean(known):.3f} ({min(known):.3f})" if known else "n/a"
        flag = "  déployé" if level in deployed else ""
        print(f"  {level:14s}  {mean:20s}  {np.mean(predictions):6.1f}                 {false_positives:4d}{flag}")
    if deployed:
        low, high = wilson_interval(published_tp, published_tp + published_fp)
        print(
            f"  publié (niveaux {', '.join(deployed)}): {published_tp} TP / {published_fp} FP sur {len(runs)} découpages, "
            f"précision {published_tp / (published_tp + published_fp):.3f} [{low:.2f} – {high:.2f}]"
        )
    else:
        print("  publié: aucun niveau déployable (règle D)")
    return {"deployed": deployed, "published_tp": published_tp, "published_fp": published_fp}


def compare(reference_name, reference, name, repeated):
    """Partition-by-partition differences with the reference variant."""
    print(f"\n--- {name} − {reference_name} (apparié, {len(repeated['runs'])} découpages) ---")
    for metric, better_if_higher in (
        ("false_positives", False), ("precision", True), ("true_positives", True), ("recall", True)
    ):
        a = np.array([run[metric] for run in reference["runs"]], dtype=float)
        b = np.array([run[metric] for run in repeated["runs"]], dtype=float)
        difference = b - a
        better = int(((difference > 0) if better_if_higher else (difference < 0)).sum())
        worse = int(((difference < 0) if better_if_higher else (difference > 0)).sum())
        print(f"  {metric:16s} {difference.mean():+7.3f} ± {difference.std():.3f}   mieux {better}/10, moins bien {worse}/10"
              .replace("/10", f"/{len(difference)}"))


def main():
    """Evaluate every variant on the same seeds and print the paired comparison."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--variant", nargs="+", action="append", required=True, metavar=("NAME TABLE", "OPTIONS"),
        help="Nom, table pickle et, en option, un JSON d'options d'entraînement (texte ou fichier .json) qui remplacent les défauts.",
    )
    parser.add_argument("--records", default="data/raw/gold_standard/gold_dataset.json")
    parser.add_argument("--aic-dump-dir", default=None)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None, help="JSON des métriques par découpage.")
    args = parser.parse_args()

    records = load_artwork_records(args.records, args.aic_dump_dir)
    results = {}
    for variant in args.variant:
        if len(variant) not in (2, 3):
            parser.error("--variant attend NOM TABLE [OPTIONS_JSON]")
        name, table_path = variant[:2]
        extra = {}
        if len(variant) == 3:
            # Options as a JSON string, or as the path of a JSON file (easier to quote on Windows).
            extra = json.loads(Path(variant[2]).read_text(encoding="utf-8")) if variant[2].endswith(".json") else json.loads(variant[2])
        options = {**TRAINING_OPTIONS, **extra}
        table = pd.read_pickle(table_path)
        repeated = evaluate_repeated_grouped_cv(
            records, table, repeats=args.repeats, n_jobs=args.jobs, n_splits=5, training_options=options
        )
        summary = summarize(name, repeated, options["minimum_predictions"])
        results[name] = {"table": table_path, "options": options, "summary": summary, "runs": repeated["runs"]}
        if len(results) > 1:
            reference_name = next(iter(results))
            compare(reference_name, results[reference_name], name, repeated)
    if args.output:
        args.output.write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
        print(f"\n[ok] Métriques par découpage dans {args.output}")


if __name__ == "__main__":
    main()
