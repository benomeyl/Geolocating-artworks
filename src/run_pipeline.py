"""
Command-line interface of the place-detection pipeline (data/filtering_logreg.py).

Commands (run from src/, or adapt the paths):

  train         Train and evaluate the hierarchical geographic model:
                    python src/run_pipeline.py train --records gold_dataset.json \\
                        --model-out results/model_hierarchical.joblib
  audit-labels  Export the hierarchical label of every candidate, without training.
  predict       Apply a trained model to new, unannotated artworks:
                    python src/run_pipeline.py predict --records artworks.json \\
                        --model-in model.joblib --output predictions.json
  train-legacy  Reproduce the older classifier trained on fuzzy name labels.

Input JSON files are either a list of {"artwork": {...}, "annotations": {...}}
records or a {"unique_artworks": [...]} envelope; "annotations" is optional
for predict.
"""

import argparse
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib

from api.aic import enrich_records_from_dump
from geolocation.gazetter import NominatimClient, geocode_candidates
from data.filtering_logreg import (
    extract_all_candidates,
    build_training_table,
    build_geographic_label_table,
    evaluate_hierarchical_grouped_cv,
    train_hierarchical_model,
    predict_with_hierarchical_model,
    train_logistic_model,
    predict_with_model,
    evaluate_binary_predictions,
    evaluate_place_identity,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_records(path):
    """Load a JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_artwork_records(path, aic_dump_dir=None):
    """Load the artwork records of a JSON file and enrich AIC records from the dump."""
    payload = load_records(path)
    records = payload.get("unique_artworks", []) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("The input JSON must contain a list of artwork records.")
    print(f"[info] {len(records)} œuvres chargées depuis {path}.")
    print(f"[info] {enrich_records_from_dump(records, aic_dump_dir)} œuvres enrichies depuis le dump AIC.")
    return records


def file_sha256(path):
    """SHA-256 of a file, recorded in model metadata to trace the training data."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ner_pipeline():
    """Load the multilingual NER pipeline shared by every command."""
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

    model_name = "Babelscape/wikineural-multilingual-ner"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    # No truncation=/max_length= here: recent transformers versions reject them
    # (TypeError) and the pipeline already truncates to
    # tokenizer.model_max_length. To force a shorter limit, set
    # tokenizer.model_max_length before building the pipeline.
    return pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        # "first" labels each word from its first sub-token; "simple" split
        # words into fragments ("##kirche", "sonnen").
        aggregation_strategy="first",
    )


def make_nominatim_client(args):
    """Build the Nominatim client from the geocoder command-line options."""
    return NominatimClient(
        endpoint=args.nominatim_url,
        reverse_endpoint=args.nominatim_reverse_url,
        user_agent=args.nominatim_user_agent,
        cache_path=args.nominatim_cache,
    )


@contextmanager
def open_gazetteer(path, missing_message=None):
    """Yield a connection to the GeoNames database, or None if the file is missing."""
    if not Path(path).is_file():
        if missing_message:
            print(missing_message)
        yield None
        return
    connection = sqlite3.connect(path)
    try:
        yield connection
    finally:
        connection.close()


def build_label_table(args, records, candidates_by_artwork, nominatim_client):
    """Geocode and label the candidates of annotated records (train and audit-labels)."""
    missing_message = f"[avertissement] Gazetteer introuvable: {args.gazetteer}." if args.no_nominatim else None
    with open_gazetteer(args.gazetteer, missing_message) as gazetteer_conn:
        return build_geographic_label_table(
            records,
            candidates_by_artwork,
            gazetteer_conn=gazetteer_conn,
            nominatim_client=nominatim_client,
            exact_tolerance_km=args.exact_tolerance_km,
            neighborhood_tolerance_km=args.neighborhood_tolerance_km,
            city_tolerance_km=args.city_tolerance_km,
        )


def _format_optional(value):
    """Format a metric that may be missing."""
    return f"{value:.3f}" if value is not None else "n/a"


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def deployable_levels(model, nested_metrics, minimum_prediction_groups, minimum_predictions):
    """Copy outer-CV metrics into each level and return the levels that can be deployed.

    A level is deployable when it is active and its outer-CV precision reaches
    its target with enough predictions and prediction groups.
    """
    levels = []
    for level, level_model in model["levels"].items():
        outer = nested_metrics["metrics_by_level"][level]
        level_model["outer_cv_precision"] = outer["precision"]
        level_model["outer_cv_recall"] = outer["recall"]
        level_model["outer_cv_prediction_groups"] = outer["prediction_groups"]
        level_model["outer_cv_predictions"] = outer["predictions"]
        level_model["outer_cv_false_positives"] = outer["false_positives"]
        if (
            level_model["status"] == "active"
            and outer["precision"] is not None
            and outer["precision"] >= level_model["target_precision"]
            and outer["prediction_groups"] >= minimum_prediction_groups
            and outer["predictions"] >= minimum_predictions
        ):
            levels.append(level)
    return levels


def print_training_report(model, nested_metrics, minimum_work_recall):
    """Print thresholds, tuning metrics, deployability and outer-CV metrics."""
    print("\n=== Seuils par niveau (précision empirique OOF) ===")
    for level in model["specificity_order"]:
        level_model = model["levels"][level]
        if level_model["status"] == "active":
            print(
                f"  {level:<16s} seuil={level_model['threshold']:.3f} "
                f"précision OOF={level_model['oof_precision']:.3f} "
                f"rappel OOF={level_model['oof_recall']:.3f} "
                f"support={level_model['oof_support']} "
                f"groupes+={level_model['positive_groups']}"
            )
        else:
            print(
                f"  {level:<16s} abstention: {level_model['status']} "
                f"(groupes+={level_model.get('positive_groups', 0)})"
            )

    work_metrics = model["threshold_tuning_work_metrics"]
    print("\n=== Décision hiérarchique par œuvre (OOF de réglage, indicatif) ===")
    print(
        f"  précision={work_metrics['precision']:.3f} "
        f"rappel={work_metrics['recall']:.3f} "
        f"FP={work_metrics['false_positives']} "
        f"FN={work_metrics['false_negatives']} "
        f"lieux mal prédits={work_metrics['mislocated_count']}"
    )
    print(f"  rappel minimal {minimum_work_recall:.2f} atteint: {model['minimum_work_recall_met']}")
    print(
        "  Note: les seuils OOF sont réglés sur ces folds; ces métriques ne sont "
        "pas une estimation finale indépendante."
    )

    print("\n=== Déployabilité par niveau (validation externe) ===")
    for level in model["specificity_order"]:
        level_metrics = nested_metrics["metrics_by_level"][level]
        print(
            f"  {level:<16s} précision={_format_optional(level_metrics['precision'])} "
            f"groupes={level_metrics['prediction_groups']} "
            f"déployé={level in model['deployable_levels']}"
        )
    print(f"[info] Niveaux activés: {model['deployable_levels'] or 'aucun (abstention)'}")

    print("\n=== Validation externe groupée par cible (estimation hors-fold) ===")
    print(
        f"  précision={nested_metrics['precision']:.3f} "
        f"rappel={nested_metrics['recall']:.3f} "
        f"FP={nested_metrics['false_positives']} "
        f"FN={nested_metrics['false_negatives']} "
        f"lieux mal prédits={nested_metrics['mislocated_count']} "
        f"folds={nested_metrics['fold_count']}"
    )
    if nested_metrics["recall"] < minimum_work_recall:
        print(
            "[avertissement] Le rappel externe est sous le minimum demandé; "
            "les cibles de précision n'ont pas été relâchées."
        )


def cmd_train(args):
    """Train the hierarchical model, evaluate it on outer folds and save it."""
    records = load_artwork_records(args.records, args.aic_dump_dir)
    print(
        "[info] Training hiérarchique: labels issus des coordonnées et de la "
        "hiérarchie; validation groupée par œuvre et cible géographique."
    )
    print("[info] Chargement du modèle NER et extraction des candidats...")
    candidates_by_artwork = extract_all_candidates(records, load_ner_pipeline())
    candidate_count = sum(len(candidates) for candidates in candidates_by_artwork.values())
    print(f"[info] {candidate_count} candidats NER bruts.")

    nominatim_client = None if args.no_nominatim else make_nominatim_client(args)
    table = build_label_table(args, records, candidates_by_artwork, nominatim_client)
    print(f"[info] {len(table)} candidats avec étiquettes géographiques.")
    if table.empty:
        raise ValueError("No candidate labels were generated.")
    print(f"[info] Niveaux: {table['label_level'].value_counts().to_dict()}")
    print(f"[info] Statuts: {table['label_status'].value_counts().to_dict()}")

    training_options = {
        "n_splits": args.n_splits,
        "regularization_values": args.regularization_values,
        "minimum_recall": args.minimum_work_recall,
        "minimum_positive_groups": args.minimum_positive_groups,
        "minimum_prediction_groups": args.minimum_prediction_groups,
        "minimum_predictions": args.minimum_predictions,
    }
    model = train_hierarchical_model(table, **training_options)
    nested_metrics = evaluate_hierarchical_grouped_cv(
        records,
        table,
        n_splits=args.outer_splits,
        training_options=training_options,
    )
    model["nested_cv_metrics"] = nested_metrics
    model["deployable_levels"] = deployable_levels(
        model, nested_metrics, args.minimum_prediction_groups, args.minimum_predictions
    )
    model["training_metadata"] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "records_file": str(args.records),
        "records_sha256": file_sha256(args.records),
        "record_count": len(records),
        "candidate_count": len(table),
        "label_strategy": "hierarchical_geographic_labels",
        "geographic_labels_used": True,
        "geocoder": "GeoNames+Nominatim" if nominatim_client else "GeoNames only",
        "target_precisions": model["target_precisions"],
        "minimum_work_recall": args.minimum_work_recall,
    }

    print_training_report(model, nested_metrics, args.minimum_work_recall)
    joblib.dump(model, args.model_out)
    print(f"\n[ok] Modèle hiérarchique sauvegardé dans {args.model_out}")


# ---------------------------------------------------------------------------
# train-legacy
# ---------------------------------------------------------------------------

def print_identity_report(identity):
    """Print the name and distance metrics of the legacy model."""
    print("\n=== Identité et proximité géographique (in-sample — indicatif) ===")
    print(f"  identity_accuracy (noms flous): {identity['identity_accuracy']:.3f}")
    print(f"  erreurs d'identité: {identity['wrong_identity_count']} / {identity['total_depicted']}")
    if identity["coordinate_accuracy"] is not None:
        print(
            f"  coordonnées à <= {identity['coordinate_tolerance_km']:.1f} km: "
            f"{identity['coordinate_accuracy']:.3f} "
            f"({identity['coordinate_evaluable']} œuvres; "
            f"couverture géocodeur {identity['coordinate_candidate_coverage']:.3f})"
        )
    if identity["city_proximity_accuracy"] is not None:
        print(
            f"  proximité du centre-ville annoté (<= {identity['city_tolerance_km']:.1f} km): "
            f"{identity['city_proximity_accuracy']:.3f} "
            f"({identity['city_evaluable']} œuvres; "
            f"couverture géocodeur {identity['city_candidate_coverage']:.3f})"
        )
    if identity["wrong_identity"]:
        print("  Exemples d'erreurs :")
        for err in identity["wrong_identity"][:5]:
            distance = err["distance_km"]
            distance_text = f"{distance:.1f} km" if distance is not None else "indisponible"
            print(
                f"    - {err['id']}: prédit '{err['predicted']}', "
                f"réel '{err['actual']}', distance: {distance_text}"
            )


def cmd_train_legacy(args):
    """Train the legacy classifier on fuzzy name labels (no dump enrichment, as before)."""
    print(f"[info] Chargement de {args.records}...")
    records = load_records(args.records).get("unique_artworks")
    print(f"[info] {len(records)} œuvres annotées chargées.")
    place_status_counts = Counter(
        record.get("annotations", {}).get("place_status", "missing") for record in records
    )
    print(f"[info] Répartition des annotations: {dict(sorted(place_status_counts.items()))}")
    print(
        "[info] Labels d'entraînement: correspondance floue des noms avec "
        "depicted_place (pas de labels géographiques hiérarchiques)."
    )

    print("[info] Chargement du modèle NER et extraction des candidats...")
    candidates_by_artwork = extract_all_candidates(records, load_ner_pipeline())
    print(f"[info]   -> {sum(len(v) for v in candidates_by_artwork.values())} candidats bruts extraits.")

    table = build_training_table(records, candidates_by_artwork)
    if len(table) == 0:
        print("[erreur] Table vide : vérifie que le NER détecte bien des LOC sur tes textes.", file=sys.stderr)
        sys.exit(1)
    print(f"[info]   -> {len(table)} lignes candidates, {int(table['label'].sum())} labels positifs par nom.")

    print("[info] Entraînement (validation croisée groupée par œuvre)...")
    model = train_logistic_model(table, n_splits=args.n_splits, minimum_recall=args.minimum_recall)
    model["training_metadata"] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "records_file": str(args.records),
        "records_sha256": file_sha256(args.records),
        "record_count": len(records),
        "place_status_counts": dict(place_status_counts),
        "label_strategy": "fuzzy_name_match",
        "geographic_labels_used": False,
        "candidate_count": len(table),
        "positive_label_count": int(table["label"].sum()),
    }

    print("\n=== Classifieur binaire (labels par nom) ===")
    print(f"C retenu           : {model['C']}")
    print(f"Seuil d'acceptation : {model['threshold']:.3f}")
    print("Coefficients du classifieur textuel :")
    for feature, coef in sorted(model["coefficients"].items(), key=lambda kv: -abs(kv[1])):
        print(f"  {feature:<24s} {coef:+.3f}")
    print(f"Intercept           : {model['intercept']:+.3f}")
    print(
        "\nMétriques OOF au niveau candidat (labels par nom; utilisées pour "
        "choisir C et le seuil, pas un test final indépendant) :"
    )
    for key, value in model["cv_metrics"].items():
        print(f"  {key}: {value}")

    # Artwork-level and identity metrics on the whole annotated set: in-sample,
    # indicative only (the out-of-fold cv_metrics above are the honest ones).
    results = predict_with_model(records, candidates_by_artwork, model)
    detection = evaluate_binary_predictions(records, results, threshold=model["threshold"])
    missing_message = f"[avertissement] Gazetteer introuvable: {args.gazetteer}; Nominatim sera utilisé seul."
    with open_gazetteer(args.gazetteer, missing_message) as gazetteer_conn:
        identity = evaluate_place_identity(
            records,
            results,
            gazetteer_conn=gazetteer_conn,
            nominatim_client=make_nominatim_client(args),
            coordinate_tolerance_km=args.coordinate_tolerance_km,
            city_tolerance_km=args.city_tolerance_km,
        )

    print("\n=== Détection par œuvre (in-sample, labels par nom — indicatif) ===")
    for key in ("precision", "recall", "f1", "accuracy"):
        print(f"  {key}: {detection[key]:.3f}")
    print_identity_report(identity)

    joblib.dump(model, args.model_out)
    print(f"\n[ok] Modèle sauvegardé dans {args.model_out} (métadonnées de données et de labels incluses).")


# ---------------------------------------------------------------------------
# audit-labels
# ---------------------------------------------------------------------------

def cmd_audit_labels(args):
    """Export the hierarchical label of every candidate to JSON, without training."""
    records = load_artwork_records(args.records, args.aic_dump_dir)
    print("[info] Extraction des candidats NER...")
    candidates_by_artwork = extract_all_candidates(records, load_ner_pipeline())
    nominatim_client = None if args.no_nominatim else make_nominatim_client(args)
    table = build_label_table(args, records, candidates_by_artwork, nominatim_client)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "records_file": str(args.records),
            "records_sha256": file_sha256(args.records),
            "record_count": len(records),
            "label_strategy": "hierarchical_geographic_audit",
            "used_for_model_training": False,
            "geocoder": "GeoNames+Nominatim" if nominatim_client else "GeoNames only",
            "tolerances_km": {
                "exact_place": args.exact_tolerance_km,
                "neighborhood": args.neighborhood_tolerance_km,
                "city": args.city_tolerance_km,
            },
        },
        "label_counts": table["label_level"].value_counts().to_dict(),
        "status_counts": table["label_status"].value_counts().to_dict(),
        "labels": json.loads(table.to_json(orient="records", force_ascii=False)),
    }
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, ensure_ascii=False, indent=2)

    print(f"[info] Répartition des niveaux: {report['label_counts']}")
    print(f"[info] Répartition des statuts: {report['status_counts']}")
    print(f"[ok] Étiquettes exportées dans {output_path}")


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

def print_model_provenance(model):
    """Print what a saved model was trained on, from its metadata."""
    training_metadata = model.get("training_metadata")
    if not training_metadata:
        print(
            "[avertissement] Ce modèle ne contient pas de métadonnées de training; "
            "son jeu de données et sa stratégie de labels sont inconnus."
        )
        return
    print(
        "[info] Modèle entraîné avec "
        f"{training_metadata.get('label_strategy', 'labels inconnus')} sur "
        f"{training_metadata.get('record_count', 'n')} œuvres; "
        "labels géographiques utilisés: "
        f"{training_metadata.get('geographic_labels_used', 'inconnu')}; "
        f"jeu SHA-256 {training_metadata.get('records_sha256', 'indisponible')[:12]}."
    )


def predict_hierarchical(args, model, records, candidates_by_artwork):
    """Predict with a hierarchical model and write one entry per artwork."""
    missing_message = f"[avertissement] Gazetteer introuvable: {args.gazetteer}; Nominatim sera utilisé seul."
    with open_gazetteer(args.gazetteer, missing_message) as gazetteer_conn:
        results = predict_with_hierarchical_model(
            records,
            candidates_by_artwork,
            model,
            gazetteer_conn=gazetteer_conn,
            nominatim_client=make_nominatim_client(args),
            geocode_top_k=args.geocode_top_k,
        )

    output = []
    for record in records:
        artwork = record.get("artwork", record)
        result = results.get(artwork.get("id"), {})
        prediction = result.get("prediction")
        output.append({
            "id": artwork.get("id"),
            "title": artwork.get("title"),
            "place_status_predicted": "depicted" if prediction else "uncertain",
            "top_candidate": prediction,
            "all_candidates": result.get("candidates", []),
        })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    predicted_count = sum(item["place_status_predicted"] == "depicted" for item in output)
    print(
        f"[ok] {predicted_count}/{len(output)} œuvres ont une prédiction "
        "hiérarchique au-dessus du seuil; les autres sont incertaines."
    )
    print(f"[ok] Résultats écrits dans {output_path}")


def predict_legacy(args, model, records, candidates_by_artwork):
    """Predict with a legacy name-based model, geocode the top candidates and write the output."""
    print("[info] Application du modèle...")
    results = predict_with_model(records, candidates_by_artwork, model)
    missing_message = f"[avertissement] Gazetteer introuvable: {args.gazetteer}; Nominatim sera utilisé seul."
    with open_gazetteer(args.gazetteer, missing_message) as gazetteer_conn:
        geocode_candidates(
            results,
            conn=gazetteer_conn,
            nominatim_client=make_nominatim_client(args),
            limit_per_artwork=args.geocode_top_k,
        )

    threshold = model["threshold"]
    output = []
    for record in records:
        artwork = record.get("artwork", record)
        places = results.get(artwork.get("id"), [])
        accepted = [p for p in places if p["score_heuristique"] >= threshold]
        output.append({
            "id": artwork.get("id"),
            "title": artwork.get("title"),
            "place_status_predicted": "depicted" if accepted else "none_or_mentioned_only",
            # Best candidate, accepted or not: useful to review near misses.
            "top_candidate": places[0] if places else None,
            "all_candidates": places,
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    n_depicted = sum(1 for o in output if o["place_status_predicted"] == "depicted")
    print(f"\n[ok] {len(output)} œuvres traitées, {n_depicted} avec un lieu accepté.")
    print(f"[ok] Résultats écrits dans {args.output}")
    print(
        f"\n[rappel] Les {args.geocode_top_k} meilleurs candidats maximum par œuvre sont géocodés. "
        "Les résultats Nominatim incluent l'attribution OpenStreetMap et peuvent être mis en cache."
    )


def cmd_predict(args):
    """Apply a saved model (hierarchical or legacy) to new artworks."""
    print(f"[info] Chargement du modèle depuis {args.model_in}...")
    model = joblib.load(args.model_in)
    print_model_provenance(model)

    records = load_artwork_records(args.records, args.aic_dump_dir)
    print("[info] Chargement du modèle NER et extraction des candidats...")
    candidates_by_artwork = extract_all_candidates(records, load_ner_pipeline())

    if model.get("model_type") == "hierarchical_geographic_logistic_regression":
        predict_hierarchical(args, model, records, candidates_by_artwork)
    else:
        predict_legacy(args, model, records, candidates_by_artwork)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def add_geocoder_arguments(parser):
    """Add the AIC dump and Nominatim options shared by every command."""
    parser.add_argument(
        "--aic-dump-dir",
        default=None,
        help="Dump AIC local pour les champs de contexte (défaut: data/raw/artic-api-data/json/artworks).",
    )
    parser.add_argument("--nominatim-url", default=None, help="Endpoint Nominatim configurable pour pouvoir changer de service.")
    parser.add_argument("--nominatim-reverse-url", default=None, help="Endpoint reverse Nominatim (déduit de --nominatim-url par défaut).")
    parser.add_argument("--nominatim-user-agent", default=None, help="User-Agent identifiant l'application; configurer un contact pour l'usage réel.")
    parser.add_argument("--nominatim-cache", default=None, help="Cache SQLite persistant (défaut: data/processed/nominatim_cache.sqlite).")


def add_label_arguments(parser):
    """Add the gazetteer, label tolerance and --no-nominatim options of train/audit-labels."""
    parser.add_argument("--gazetteer", default="data/gazetter/geonames.db", help="Base SQLite GeoNames.")
    parser.add_argument("--exact-tolerance-km", type=float, default=0.25)
    parser.add_argument("--neighborhood-tolerance-km", type=float, default=2.0)
    parser.add_argument("--city-tolerance-km", type=float, default=25.0)
    parser.add_argument(
        "--no-nominatim",
        action="store_true",
        help="N'utiliser que GeoNames local; les candidats non résolus restent non évaluables.",
    )


def main():
    """Parse the command line and run the chosen command."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Entraîne le modèle sur un jeu annoté.")
    train_parser.add_argument("--records", required=True, help="JSON des œuvres annotées.")
    train_parser.add_argument(
        "--model-out",
        default="results/model_hierarchical.joblib",
        help="Fichier de sortie du modèle hiérarchique.",
    )
    train_parser.add_argument("--n-splits", type=int, default=5, help="Nombre de folds pour la validation croisée.")
    train_parser.add_argument("--outer-splits", type=int, default=5, help="Nombre de folds externes d’évaluation.")
    train_parser.add_argument(
        "--minimum-work-recall",
        "--minimum-recall",
        dest="minimum_work_recall",
        type=float,
        default=0.5,
        help="Rappel OOF minimal visé au niveau œuvre (défaut: 0.50).",
    )
    train_parser.add_argument(
        "--regularization-values",
        type=float,
        nargs="+",
        default=(0.1, 1.0, 10.0),
        help="Valeurs C testées pour chaque niveau.",
    )
    train_parser.add_argument("--minimum-positive-groups", type=int, default=5)
    train_parser.add_argument("--minimum-prediction-groups", type=int, default=5)
    train_parser.add_argument("--minimum-predictions", type=int, default=5)
    add_label_arguments(train_parser)
    add_geocoder_arguments(train_parser)
    train_parser.set_defaults(func=cmd_train)

    legacy_train_parser = subparsers.add_parser(
        "train-legacy",
        help="Reproduit l’ancien classifieur à labels par noms flous.",
    )
    legacy_train_parser.add_argument("--records", required=True)
    legacy_train_parser.add_argument("--model-out", default="results/model_name_labels_legacy.joblib")
    legacy_train_parser.add_argument("--n-splits", type=int, default=5)
    legacy_train_parser.add_argument("--minimum-recall", type=float, default=0.8)
    legacy_train_parser.add_argument("--gazetteer", default="data/gazetter/geonames.db")
    legacy_train_parser.add_argument("--coordinate-tolerance-km", type=float, default=5.0)
    legacy_train_parser.add_argument("--city-tolerance-km", type=float, default=30.0)
    add_geocoder_arguments(legacy_train_parser)
    legacy_train_parser.set_defaults(func=cmd_train_legacy)

    labels_parser = subparsers.add_parser(
        "audit-labels",
        help="Génère les étiquettes géographiques hiérarchiques pour inspection.",
    )
    labels_parser.add_argument("--records", required=True, help="JSON du gold standard.")
    labels_parser.add_argument(
        "--output",
        default="results/geographic_candidate_labels.json",
        help="Fichier JSON de sortie pour les étiquettes auditables.",
    )
    add_label_arguments(labels_parser)
    add_geocoder_arguments(labels_parser)
    labels_parser.set_defaults(func=cmd_audit_labels)

    predict_parser = subparsers.add_parser("predict", help="Applique un modèle entraîné à de nouvelles œuvres.")
    predict_parser.add_argument("--records", required=True, help="JSON des œuvres à traiter (annotations optionnelles).")
    predict_parser.add_argument("--model-in", required=True, help="Modèle entraîné (fichier .joblib).")
    predict_parser.add_argument("--output", default="predictions.json", help="Fichier de sortie des prédictions.")
    predict_parser.add_argument("--gazetteer", default="data/gazetter/geonames.db", help="Base SQLite GeoNames.")
    predict_parser.add_argument("--geocode-top-k", type=int, default=3, help="Nombre maximal de candidats géocodés par œuvre.")
    add_geocoder_arguments(predict_parser)
    predict_parser.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
