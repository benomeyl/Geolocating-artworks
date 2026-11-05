"""Compare feature sets of the hierarchical model on the same label table.

The NER, geocoding and labels are computed once; each feature set is then
trained and evaluated exactly like `run_pipeline.py train` (outer grouped
CV, thresholds tuned inside the outer training folds). Because the
thresholded outer metrics rest on a few dozen predictions, the script also
reports a threshold-free score per level: the out-of-fold average precision
(area under the precision-recall curve), averaged over several CV seeds.
Every set is also evaluated on the same --outer-repeats shuffled partitions
(evaluate_repeated_grouped_cv), and compared partition by partition with
the first set: a difference that has the same sign on most partitions is
larger than the noise of the split.

    python scripts/compare_feature_sets.py --output results/compare_title.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.filtering_logreg import (  # noqa: E402
    GEOGRAPHIC_FEATURE_COLUMNS,
    GEOGRAPHIC_LEVELS,
    MENTION_ROLE_COLUMNS,
    _levels_up_to,
    evaluate_hierarchical_grouped_cv,
    evaluate_repeated_grouped_cv,
    extract_all_candidates,
    train_hierarchical_model,
)
from data.nli import NliScorer  # noqa: E402
from run_pipeline import (  # noqa: E402
    add_geocoder_arguments,
    add_label_arguments,
    build_label_table,
    deployable_levels,
    load_artwork_records,
    load_ner_pipeline,
    make_nominatim_client,
)

# Each set adds one feature group to the previous one.
FEATURE_SETS = {
    "v7": GEOGRAPHIC_FEATURE_COLUMNS,
    "v7 + rôle de la mention": GEOGRAPHIC_FEATURE_COLUMNS + MENTION_ROLE_COLUMNS,
}
# Columns whose weights and discriminative power are reported (the new group).
NEW_COLUMNS = MENTION_ROLE_COLUMNS
TRAINING_OPTIONS = {
    "n_splits": 5,
    "regularization_values": (0.1, 1.0, 10.0),
    "minimum_recall": 0.5,
    "minimum_positive_groups": 5,
    "minimum_prediction_groups": 5,
    "minimum_predictions": 5,
}
AP_SEEDS = range(5)


def build_or_load_table(args):
    """Label table of the gold records, rebuilt unless --reuse-table finds a cached one."""
    records = load_artwork_records(args.records, args.aic_dump_dir)
    if args.reuse_table and args.table_cache.is_file():
        table = pd.read_pickle(args.table_cache)
        print(f"[info] Table relue depuis {args.table_cache} ({len(table)} candidats).")
        return records, table
    candidates_by_artwork = extract_all_candidates(records, load_ner_pipeline())
    nominatim_client = None if args.no_nominatim else make_nominatim_client(args)
    table = build_label_table(args, records, candidates_by_artwork, nominatim_client, NliScorer())
    args.table_cache.parent.mkdir(parents=True, exist_ok=True)
    table.to_pickle(args.table_cache)
    print(f"[info] {len(table)} candidats étiquetés, table sauvegardée dans {args.table_cache}.")
    return records, table


def level_rows(table, level):
    """Rows usable for `level` with their cumulative labels and CV groups (as in _fit_level)."""
    usable = table[table["label_status"].isin(["negative", "match"])]
    labels = (
        (usable["label_status"] == "match") & usable["label_level"].isin(_levels_up_to(level))
    ).astype(int).to_numpy()
    return usable, labels, usable["cv_group"].astype(str).to_numpy()


def threshold_free_scores(table, columns):
    """Out-of-fold average precision and ROC AUC per level, mean and std over AP_SEEDS."""
    scores = {}
    for level in GEOGRAPHIC_LEVELS:
        usable, labels, groups = level_rows(table, level)
        n_splits = min(5, len(set(groups[labels == 1])), len(set(groups[labels == 0])))
        if n_splits < 2:
            scores[level] = None
            continue
        X = usable[columns].fillna(0).to_numpy(dtype=float)
        average_precisions, aucs = [], []
        for seed in AP_SEEDS:
            oof = np.zeros(len(labels))
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            for train_idx, test_idx in cv.split(X, labels, groups):
                classifier = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
                classifier.fit(X[train_idx], labels[train_idx])
                oof[test_idx] = classifier.predict_proba(X[test_idx])[:, 1]
            average_precisions.append(average_precision_score(labels, oof))
            aucs.append(roc_auc_score(labels, oof))
        scores[level] = {
            "positives": int(labels.sum()),
            "rows": len(labels),
            "average_precision": float(np.mean(average_precisions)),
            "average_precision_std": float(np.std(average_precisions)),
            "roc_auc": float(np.mean(aucs)),
        }
    return scores


def evaluate_feature_set(records, table, columns, repeats, n_jobs):
    """Outer-CV metrics (fixed and repeated partitions), deployable levels and threshold-free scores."""
    options = {**TRAINING_OPTIONS, "feature_columns": columns}
    model = train_hierarchical_model(table, **options)
    nested = evaluate_hierarchical_grouped_cv(records, table, n_splits=5, training_options=options)
    repeated = evaluate_repeated_grouped_cv(
        records, table, repeats=repeats, n_jobs=n_jobs, n_splits=5, training_options=options
    )
    deployable = deployable_levels(
        model, nested, TRAINING_OPTIONS["minimum_prediction_groups"], TRAINING_OPTIONS["minimum_predictions"]
    )
    return {
        "feature_count": len(columns),
        "outer_precision": nested["precision"],
        "outer_recall": nested["recall"],
        "outer_false_positives": nested["false_positives"],
        "outer_false_negatives": nested["false_negatives"],
        "outer_mislocated": nested["mislocated_count"],
        "outer_by_level": {
            level: {
                key: nested["metrics_by_level"][level][key]
                for key in ("precision", "predictions", "prediction_groups", "false_positives")
            }
            for level in GEOGRAPHIC_LEVELS
        },
        "active_levels": [level for level in GEOGRAPHIC_LEVELS if model["levels"][level]["status"] == "active"],
        "deployable_levels": deployable,
        "threshold_free": threshold_free_scores(table, columns),
        "repeated": {
            "precision": [run["precision"] for run in repeated["runs"]],
            "recall": [run["recall"] for run in repeated["runs"]],
            "by_level": {
                level: {key: repeated["metrics_by_level"][level][key] for key in ("precision", "predictions")}
                for level in GEOGRAPHIC_LEVELS
            },
        },
    }


def new_feature_summary(table, columns):
    """Mean of each new feature on correct and wrong candidates, its AUC alone and its weights."""
    usable = table[table["label_status"].isin(["negative", "match"])]
    is_match = (usable["label_status"] == "match").to_numpy()
    summary = {}
    for column in NEW_COLUMNS:
        values = usable[column].fillna(0).to_numpy(dtype=float)
        summary[column] = {
            "mean_match": float(values[is_match].mean()),
            "mean_negative": float(values[~is_match].mean()),
            "mean_negative_not_depicted": float(
                values[(~is_match) & (usable["place_status"] != "depicted").to_numpy()].mean()
            ),
            "auc_alone": float(roc_auc_score(is_match, values)),
        }
    for level in ("city", "country"):
        _, labels, _ = level_rows(table, level)
        X = usable[columns].fillna(0).to_numpy(dtype=float)
        classifier = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(X, labels)
        for column, weight in zip(columns, classifier.coef_[0]):
            if column in summary:
                summary[column][f"weight_{level}"] = float(weight)
    return summary


def print_report(results, summary, base_rate):
    """Print the comparison table, the threshold-free scores and the new features."""
    print("\n=== Validation externe (mêmes folds, seuils réglés dans chaque fold) ===")
    print(f"  {'jeu':<22s} {'préc.':>6s} {'rappel':>6s} {'FP':>4s} {'FN':>4s}  ville / région / pays (préc., groupes)")
    for name, result in results.items():
        levels = " / ".join(
            f"{_fmt(result['outer_by_level'][level]['precision'])} ({result['outer_by_level'][level]['prediction_groups']})"
            for level in ("city", "region", "country")
        )
        print(
            f"  {name:<22s} {_fmt(result['outer_precision']):>6s} {_fmt(result['outer_recall']):>6s} "
            f"{result['outer_false_positives']:>4d} {result['outer_false_negatives']:>4d}  {levels}"
            f"  déployés={result['deployable_levels'] or 'aucun'}"
        )
    reference_name, reference = next(iter(results.items()))
    print(f"\n=== Validation externe répétée ({len(reference['repeated']['precision'])} découpages identiques pour tous les jeux) ===")
    for name, result in results.items():
        repeated = result["repeated"]
        line = (
            f"  {name:<28s} préc.={np.mean(repeated['precision']):.3f}±{np.std(repeated['precision']):.3f} "
            f"rappel={np.mean(repeated['recall']):.3f}±{np.std(repeated['recall']):.3f}"
        )
        if name != reference_name:
            for metric in ("precision", "recall"):
                differences = np.subtract(repeated[metric], reference["repeated"][metric])
                line += (
                    f"  Δ{metric[:4]}={differences.mean():+.3f} "
                    f"(mieux {int((differences > 0).sum())}/{len(differences)})"
                )
        print(line)
        print("      " + " / ".join(
            f"{level} {_fmt(repeated['by_level'][level]['precision'][0])} ({repeated['by_level'][level]['predictions'][0] or 0:.0f} préd.)"
            for level in ("city", "region", "country")
        ))
    print("\n=== Précision moyenne hors-fold par niveau (sans seuil, moyenne sur 5 graines) ===")
    print(f"  {'jeu':<22s} " + " ".join(f"{level:>14s}" for level in GEOGRAPHIC_LEVELS))
    for name, result in results.items():
        cells = []
        for level in GEOGRAPHIC_LEVELS:
            score = result["threshold_free"][level]
            cells.append(f"{score['average_precision']:.3f}±{score['average_precision_std']:.3f}" if score else "n/a")
        print(f"  {name:<22s} " + " ".join(f"{cell:>14s}" for cell in cells))
    print(f"\n=== Nouvelles features (taux de candidats justes global: {base_rate:.2f}) ===")
    for column, stats in summary.items():
        print(
            f"  {column:<18s} moyenne justes={stats['mean_match']:.2f} faux={stats['mean_negative']:.2f} "
            f"(œuvres sans lieu {stats['mean_negative_not_depicted']:.2f}) AUC seule={stats['auc_alone']:.3f} "
            f"poids ville={stats.get('weight_city', 0):+.2f} pays={stats.get('weight_country', 0):+.2f}"
        )


def _fmt(value):
    """Format an optional ratio."""
    return f"{value:.3f}" if value is not None else "n/a"


def main():
    """Parse the command line, build the table once and compare every feature set."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", default="data/raw/gold_standard/gold_dataset.json")
    parser.add_argument("--output", type=Path, default=Path("results/compare_feature_sets.json"))
    parser.add_argument("--table-cache", type=Path, default=Path("results/feature_table.pkl"))
    parser.add_argument(
        "--reuse-table",
        action="store_true",
        help="Relire la table en cache au lieu de relancer NER et géocodage (si le code des features n'a pas changé).",
    )
    parser.add_argument("--outer-repeats", type=int, default=10)
    parser.add_argument("--outer-jobs", type=int, default=8)
    add_label_arguments(parser)
    add_geocoder_arguments(parser)
    args = parser.parse_args()

    records, table = build_or_load_table(args)
    print(f"[info] Statuts: {table['label_status'].value_counts().to_dict()}")
    results = {}
    for name, columns in FEATURE_SETS.items():
        print(f"[info] Évaluation de « {name} » ({len(columns)} features)...")
        results[name] = evaluate_feature_set(records, table, columns, args.outer_repeats, args.outer_jobs)
    largest = list(FEATURE_SETS.values())[-1]
    summary = new_feature_summary(table, largest)
    usable = table[table["label_status"].isin(["negative", "match"])]
    print_report(results, summary, float((usable["label_status"] == "match").mean()))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"feature_sets": results, "new_features": summary}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\n[ok] Résultats écrits dans {args.output}")


if __name__ == "__main__":
    main()
