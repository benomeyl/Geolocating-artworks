"""
Script d'utilisation du pipeline régression logistique (filtering_logreg.py).

Deux usages couverts, à lancer séparément :

  1. Entraînement sur le jeu annoté :
     python run_pipeline.py train --records annotated.json --model-out model.joblib

  2. Application à de nouvelles œuvres NON annotées (production) :
     python run_pipeline.py predict --records nouvelles_oeuvres.json \\
         --model-in model.joblib --output predictions.json

Les fichiers JSON d'entrée suivent le même format que celui utilisé tout au
long du projet : une liste d'objets {"artwork": {...}, "annotations": {...}}
(la clé "annotations" est optionnelle pour le mode predict).
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import joblib
import pandas as pd

from geolocation.gazetter import NominatimClient, geocode_candidates
from data.filtering_logreg import (
    extract_all_candidates,
    build_training_table,
    build_geographic_label_table,
    train_logistic_model,
    predict_with_model,
    evaluate_binary_predictions,
    evaluate_place_identity,
)


def load_records(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_ner_pipeline():
    """
    Charge le modèle NER utilisé depuis le début du projet. Centralisé ici
    pour ne pas le réécrire dans train/predict.
    """
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

    model_name = "Babelscape/wikineural-multilingual-ner"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    # Pas de truncation=/max_length= ici : TokenClassificationPipeline les
    # refuse dans les versions récentes de transformers (TypeError), et elle
    # tronque déjà d'elle-même à tokenizer.model_max_length. Pour forcer une
    # limite plus courte (ex: 256), fixer tokenizer.model_max_length = 256
    # avant de créer la pipeline.
    return pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
    )


def cmd_train(args):
    print(f"[info] Chargement de {args.records}...")
    records = load_records(args.records).get("unique_artworks")
    print(f"[info] {len(records)} œuvres annotées chargées.")

    print("[info] Chargement du modèle NER...")
    ner = load_ner_pipeline()

    print("[info] Extraction des candidats (NER sur tous les champs)...")
    candidates_by_artwork = extract_all_candidates(records, ner)
    n_candidates = sum(len(v) for v in candidates_by_artwork.values())
    print(f"[info]   -> {n_candidates} candidats bruts extraits.")

    print("[info] Construction de la table d'entraînement...")
    table = build_training_table(records, candidates_by_artwork)
    print(f"[info]   -> {len(table)} lignes ({int(table['label'].sum())} positifs).")
    if len(table) == 0:
        print("[erreur] Table vide : vérifie que le NER détecte bien des LOC sur tes textes.", file=sys.stderr)
        sys.exit(1)

    print("[info] Entraînement (validation croisée groupée par œuvre)...")
    model = train_logistic_model(
        table,
        n_splits=args.n_splits,
        minimum_recall=args.minimum_recall,
    )

    print("\n=== Résultat de l'entraînement ===")
    print(f"C retenu           : {model['C']}")
    print(f"Seuil d'acceptation : {model['threshold']:.3f}")
    print("Coefficients appris (comparer avec les anciens poids à la main) :")
    for feature, coef in sorted(model["coefficients"].items(), key=lambda kv: -abs(kv[1])):
        print(f"  {feature:<24s} {coef:+.3f}")
    print(f"Intercept           : {model['intercept']:+.3f}")
    print("\nMétriques de validation croisée (hors-échantillon, au niveau candidat) :")
    for key, value in model["cv_metrics"].items():
        print(f"  {key}: {value}")

    # Évaluation au niveau œuvre + identité du lieu, sur TOUT le jeu annoté
    # (indicatif — les vraies métriques de généralisation sont celles de
    # cv_metrics ci-dessus, calculées hors-échantillon).
    results = predict_with_model(records, candidates_by_artwork, model)
    detection = evaluate_binary_predictions(records, results, threshold=model["threshold"])
    gazetteer_conn = None
    if Path(args.gazetteer).is_file():
        gazetteer_conn = sqlite3.connect(args.gazetteer)
    else:
        print(f"[avertissement] Gazetteer introuvable: {args.gazetteer}; Nominatim sera utilisé seul.")
    try:
        identity = evaluate_place_identity(
            records,
            results,
            gazetteer_conn=gazetteer_conn,
            nominatim_client=NominatimClient(
                endpoint=args.nominatim_url,
                reverse_endpoint=args.nominatim_reverse_url,
                user_agent=args.nominatim_user_agent,
                cache_path=args.nominatim_cache,
            ),
            coordinate_tolerance_km=args.coordinate_tolerance_km,
            city_tolerance_km=args.city_tolerance_km,
        )
    finally:
        if gazetteer_conn is not None:
            gazetteer_conn.close()

    print("\n=== Détection (au niveau œuvre, in-sample — indicatif) ===")
    for key in ("precision", "recall", "f1", "accuracy"):
        print(f"  {key}: {detection[key]:.3f}")

    print("\n=== Identité du lieu (parmi les œuvres 'depicted') ===")
    print(f"  identity_accuracy: {identity['identity_accuracy']:.3f}")
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

    joblib.dump(model, args.model_out)
    print(f"\n[ok] Modèle sauvegardé dans {args.model_out}")


def cmd_audit_labels(args):
    payload = load_records(args.records)
    records = payload.get("unique_artworks", []) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("The input JSON must contain a list of artwork records.")

    print(f"[info] {len(records)} œuvres chargées; extraction des candidats NER...")
    candidates_by_artwork = extract_all_candidates(records, load_ner_pipeline())
    gazetteer_conn = None
    if Path(args.gazetteer).is_file():
        gazetteer_conn = sqlite3.connect(args.gazetteer)
    elif args.no_nominatim:
        print(f"[avertissement] Gazetteer introuvable: {args.gazetteer}.")

    nominatim_client = None
    if not args.no_nominatim:
        nominatim_client = NominatimClient(
            endpoint=args.nominatim_url,
            reverse_endpoint=args.nominatim_reverse_url,
            user_agent=args.nominatim_user_agent,
            cache_path=args.nominatim_cache,
        )

    try:
        table = build_geographic_label_table(
            records,
            candidates_by_artwork,
            gazetteer_conn=gazetteer_conn,
            nominatim_client=nominatim_client,
            exact_tolerance_km=args.exact_tolerance_km,
            neighborhood_tolerance_km=args.neighborhood_tolerance_km,
            city_tolerance_km=args.city_tolerance_km,
        )
    finally:
        if gazetteer_conn is not None:
            gazetteer_conn.close()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "label_counts": table["label_level"].value_counts().to_dict(),
        "status_counts": table["label_status"].value_counts().to_dict(),
        "labels": json.loads(table.to_json(orient="records", force_ascii=False)),
    }
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, ensure_ascii=False, indent=2)

    print(f"[info] Répartition des niveaux: {report['label_counts']}")
    print(f"[info] Répartition des statuts: {report['status_counts']}")
    print(f"[ok] Étiquettes exportées dans {output_path}")


def cmd_predict(args):
    print(f"[info] Chargement du modèle depuis {args.model_in}...")
    model = joblib.load(args.model_in)

    print(f"[info] Chargement de {args.records}...")
    records = load_records(args.records)
    print(f"[info] {len(records)} œuvres à traiter.")

    print("[info] Chargement du modèle NER...")
    ner = load_ner_pipeline()

    print("[info] Extraction des candidats...")
    candidates_by_artwork = extract_all_candidates(records, ner)

    print("[info] Application du modèle...")
    results = predict_with_model(records, candidates_by_artwork, model)

    gazetteer_conn = None
    if Path(args.gazetteer).is_file():
        gazetteer_conn = sqlite3.connect(args.gazetteer)
    else:
        print(f"[avertissement] Gazetteer introuvable: {args.gazetteer}; Nominatim sera utilisé seul.")
    try:
        geocode_candidates(
            results,
            conn=gazetteer_conn,
            nominatim_client=NominatimClient(
                endpoint=args.nominatim_url,
                reverse_endpoint=args.nominatim_reverse_url,
                user_agent=args.nominatim_user_agent,
                cache_path=args.nominatim_cache,
            ),
            limit_per_artwork=args.geocode_top_k,
        )
    finally:
        if gazetteer_conn is not None:
            gazetteer_conn.close()

    threshold = model["threshold"]
    output = []
    for record in records:
        artwork = record.get("artwork", record)
        artwork_id = artwork.get("id")
        places = results.get(artwork_id, [])
        accepted = [p for p in places if p["score_heuristique"] >= threshold]

        output.append({
            "id": artwork_id,
            "title": artwork.get("title"),
            "place_status_predicted": "depicted" if accepted else "none_or_mentioned_only",
            # Le meilleur candidat, qu'il soit accepté ou non : utile pour
            # une relecture manuelle des cas proches du seuil.
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
        "Les résultats Nominatim "
        "incluent l'attribution OpenStreetMap et peuvent être mis en cache."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Entraîne le modèle sur un jeu annoté.")
    train_parser.add_argument("--records", required=True, help="JSON des œuvres annotées.")
    train_parser.add_argument("--model-out", default="model.joblib", help="Fichier de sortie pour le modèle entraîné.")
    train_parser.add_argument("--n-splits", type=int, default=5, help="Nombre de folds pour la validation croisée.")
    train_parser.add_argument("--minimum-recall", type=float, default=0.8, help="Rappel minimum visé pour le choix du seuil.")
    train_parser.add_argument("--gazetteer", default="data/gazetter/geonames.db", help="Base SQLite GeoNames.")
    train_parser.add_argument("--coordinate-tolerance-km", type=float, default=5.0, help="Tolérance de proximité du point annoté, en km.")
    train_parser.add_argument("--city-tolerance-km", type=float, default=30.0, help="Rayon de proximité du centre-ville annoté, en km.")
    add_geocoder_arguments(train_parser)
    train_parser.set_defaults(func=cmd_train)

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
    labels_parser.add_argument("--gazetteer", default="data/gazetter/geonames.db")
    labels_parser.add_argument("--exact-tolerance-km", type=float, default=0.25)
    labels_parser.add_argument("--neighborhood-tolerance-km", type=float, default=2.0)
    labels_parser.add_argument("--city-tolerance-km", type=float, default=25.0)
    labels_parser.add_argument(
        "--no-nominatim",
        action="store_true",
        help="N'utiliser que GeoNames local; les candidats non résolus restent non évaluables.",
    )
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


def add_geocoder_arguments(parser):
    parser.add_argument("--nominatim-url", default=None, help="Endpoint Nominatim configurable pour pouvoir changer de service.")
    parser.add_argument("--nominatim-reverse-url", default=None, help="Endpoint reverse Nominatim (déduit de --nominatim-url par défaut).")
    parser.add_argument("--nominatim-user-agent", default=None, help="User-Agent identifiant l'application; configurer un contact pour l'usage réel.")
    parser.add_argument("--nominatim-cache", default=None, help="Cache SQLite persistant (défaut: data/processed/nominatim_cache.sqlite).")


if __name__ == "__main__":
    main()
