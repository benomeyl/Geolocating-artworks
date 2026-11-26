"""Train and save a hierarchical model from a cached label table.

Same training, outer validation and deployment rule (rule D over repeated
partitions) as `run_pipeline.py train`, but the table (candidates, features,
labels) is read from a pickle instead of being rebuilt: no NER, geocoding,
NLI or LLM is needed. Options are those of scripts/compare_tables.py, plus a
JSON file or string that replaces them (e.g. results/llm/hybrid_options.json
for the hybrid LLM model).

    python scripts/train_from_table.py --table results/feature_table_v15_llm.pkl \
        --options results/llm/hybrid_options.json --model-out results/model_hierarchical_v15_hybrid.joblib
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_tables import TRAINING_OPTIONS  # noqa: E402
from data.filtering_logreg import (  # noqa: E402
    evaluate_hierarchical_grouped_cv,
    evaluate_repeated_grouped_cv,
    train_hierarchical_model,
)
from run_pipeline import (  # noqa: E402
    deployable_levels,
    file_sha256,
    load_artwork_records,
    print_repeated_cv_report,
    print_training_report,
)


def main():
    """Train on the table, validate on outer partitions and save the model."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--options", default=None, help="Options d'entraînement : fichier .json ou texte JSON.")
    parser.add_argument("--records", type=Path, default=Path("data/raw/gold_standard/gold_dataset.json"))
    parser.add_argument("--aic-dump-dir", default=None)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--outer-repeats", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    extra = {}
    if args.options:
        extra = json.loads(Path(args.options).read_text(encoding="utf-8")) if args.options.endswith(".json") \
            else json.loads(args.options)
    options = {**TRAINING_OPTIONS, **extra}
    records = load_artwork_records(args.records, args.aic_dump_dir)
    table = pd.read_pickle(args.table)
    print(f"[info] {len(table)} candidats lus depuis {args.table} ; options : {options}")

    model = train_hierarchical_model(table, **options)
    nested = evaluate_hierarchical_grouped_cv(records, table, n_splits=args.outer_splits, training_options=options)
    model["nested_cv_metrics"] = nested
    repeated = None
    if args.outer_repeats > 1:
        repeated = evaluate_repeated_grouped_cv(
            records, table, repeats=args.outer_repeats, n_jobs=args.jobs, n_splits=args.outer_splits,
            training_options=options,
        )
        model["repeated_cv_metrics"] = repeated
    model["deployable_levels"] = deployable_levels(
        model, nested, options["minimum_prediction_groups"], options["minimum_predictions"], repeated
    )
    model["deployment_rule"] = (
        "mean precision over shuffled partitions >= target and pooled Wilson lower bound >= target - 0.05"
        if repeated is not None else "fixed-partition precision >= target"
    )
    model["training_metadata"] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "records_file": str(args.records),
        "records_sha256": file_sha256(args.records),
        "record_count": len(records),
        "table_file": str(args.table),
        "table_sha256": file_sha256(args.table),
        "candidate_count": len(table),
        "training_options": options,
        "label_strategy": "hierarchical_geographic_labels",
        "geographic_labels_used": True,
        "target_precisions": model["target_precisions"],
        "minimum_work_recall": options["minimum_recall"],
    }

    print_training_report(model, nested, options["minimum_recall"])
    if repeated is not None:
        print_repeated_cv_report(repeated)
        print(f"[info] Règle de déploiement: {model['deployment_rule']}")
    print(f"[info] Niveaux actifs (seuil trouvé): {model['active_levels']}")
    print(f"[info] Niveaux activés: {model['deployable_levels'] or 'aucun (abstention)'}")
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_out)
    print(f"\n[ok] Modèle sauvegardé dans {args.model_out}")


if __name__ == "__main__":
    main()
