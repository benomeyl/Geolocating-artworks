"""Compare local LLMs (Ollama) for reading the places of an artwork.

Two groups of gold artworks are read by each model (data/llm_places.py):

- "exact_perdu": artworks annotated with an exact place that never reaches
  the model as an exact candidate today (no candidate of the label table
  matches at the exact level). Question: does the LLM name the monument
  well enough for Wikidata/Nominatim to find it (verified place within
  0.5 km of the annotation)?
- "sans_lieu": artworks annotated as depicting no place but with at least
  one place candidate (portraits, objects, "made in" places): these give
  the model's confident false positives. Question: how often does the LLM
  say a place is depicted, and how often is such a place verified (it
  would then become a false positive candidate)?

Speed (seconds per artwork, CPU) is reported too. Results per artwork go to
--output-csv, the summary to stdout and --output-json. Answers are cached,
so an interrupted run resumes where it stopped.

    ollama pull qwen3:4b && ollama pull gemma3:4b
    python scripts/compare_local_llms.py --models qwen3:4b gemma3:4b
    python scripts/compare_local_llms.py --models gemma3:4b qwen3:4b qwen3:4b+think --per-group 20
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.llm_places import DEFAULT_OLLAMA_URL, LocalLlm, distance_km, verify_place  # noqa: E402
from geolocation.gazetter import NominatimClient  # noqa: E402
from geolocation.wikidata import WikidataClient  # noqa: E402
from run_pipeline import load_artwork_records  # noqa: E402

EXACT_KM = 0.5   # verified place counted as the annotated monument
CITY_KM = 25.0   # verified place at least in the right city


def _targets(record):
    """Annotated depicted places of a record (with coordinates)."""
    annotations = record.get("annotations", {})
    targets = annotations.get("depicted_places") or [annotations.get("depicted_place")]
    return [target for target in targets if target and target.get("latitude") is not None]


def build_sample(records, table, size_without_place, seed):
    """(artwork id, group, record) for the two groups, in a fixed order."""
    exact_matches = set(
        table[(table["label_status"] == "match") & (table["label_level"] == "exact_place")]["artwork_id"].astype(str)
    )
    candidates = set(table["artwork_id"].astype(str))
    lost, without_place = [], []
    for record in records:
        artwork_id = str(record["artwork"].get("id"))
        status = record.get("annotations", {}).get("place_status")
        if status == "depicted" and any(t.get("precision") == "exact_place" for t in _targets(record)):
            if artwork_id not in exact_matches:
                lost.append((artwork_id, "exact_perdu", record))
        elif status == "none" and artwork_id in candidates:
            without_place.append((artwork_id, "sans_lieu", record))
    random.Random(seed).shuffle(without_place)
    return lost + without_place[:size_without_place]


def evaluate(llm, sample, nominatim_client, wikidata_client):
    """One row per artwork: the model's answer and what the gazetteers confirm.

    Places of every role are verified: the "_any_role" columns measure how
    well the model names monuments, independently of its depicted/mentioned
    judgement (a model that calls everything "mentioned" still names it).
    """
    rows = []
    for position, (artwork_id, group, record) in enumerate(sample, start=1):
        answer, seconds = llm.read_places(record["artwork"])
        places = answer.get("places", [])
        verify_started = time.monotonic()
        checked = [(place, verify_place(place, nominatim_client, wikidata_client)) for place in places]
        verify_seconds = time.monotonic() - verify_started
        checked_precise = [
            (place, location) for place, location in checked
            if location is not None and place.get("type") not in ("city", "region", "country")
        ]
        depicted = [place for place in places if place.get("role") == "depicted"]
        verified = [(p, loc) for p, loc in checked if loc is not None and p.get("role") == "depicted"]
        precise = [(p, loc) for p, loc in checked_precise if p.get("role") == "depicted"]
        row = {
            "model": llm.label, "group": group, "artwork_id": artwork_id,
            "title": record["artwork"].get("title"), "seconds": round(seconds or 0.0, 1),
            "verify_seconds": round(verify_seconds, 2), "places_checked": len(places),
            "json_error": "error" in answer,
            "depicts_place": bool(answer.get("depicts_place")),
            "depicted": " | ".join(f"{p.get('name')} [{p.get('type')}; {p.get('city')}, {p.get('country')}]" for p in depicted),
            "verified": " | ".join(f"{loc.get('name')} ({loc['verified_by']})" for _, loc in verified),
            "verified_precise_count": len(precise),
        }
        targets = _targets(record)
        if group == "exact_perdu" and targets:
            target = next(t for t in targets if t.get("precision") == "exact_place")
            row["target"] = target.get("name")
            distances = [distance_km(location, target["latitude"], target["longitude"]) for _, location in precise]
            nearest = min(distances, default=None)
            row["nearest_km"] = round(nearest, 2) if nearest is not None else None
            row["exact_found"] = nearest is not None and nearest <= EXACT_KM
            row["precise_wrong"] = sum(1 for distance in distances if distance > CITY_KM)
            any_role = [distance_km(loc, target["latitude"], target["longitude"]) for _, loc in checked_precise]
            row["exact_found_any_role"] = bool(any_role) and min(any_role) <= EXACT_KM
        rows.append(row)
        print(f"  [{llm.label}] {position}/{len(sample)} {group:11s} {seconds or 0:5.1f} s  {str(row['title'])[:50]}",
              flush=True)
    return rows


def summarize(results):
    """Per-model summary of the two groups."""
    summary = {}
    for model, rows in results.groupby("model"):
        lost = rows[rows["group"] == "exact_perdu"]
        none = rows[rows["group"] == "sans_lieu"]
        aic = lost[lost["artwork_id"].str.isdigit()]
        summary[model] = {
            "secondes_par_oeuvre": round(rows["seconds"].mean(), 1),
            "verification_secondes_par_oeuvre": round(rows["verify_seconds"].mean(), 2),
            "json_invalide": int(rows["json_error"].sum()),
            "exact_perdu_oeuvres": len(lost),
            "exact_perdu_lieu_represente": int(lost["depicts_place"].sum()),
            "exact_perdu_monument_retrouve": int(lost["exact_found"].sum()),
            "exact_perdu_monument_retrouve_AIC": f"{int(aic['exact_found'].sum())}/{len(aic)}",
            "exact_perdu_monument_nomme_tout_role": int(lost["exact_found_any_role"].sum()),
            "exact_perdu_lieux_precis_faux": int(lost["precise_wrong"].sum()),
            "sans_lieu_oeuvres": len(none),
            "sans_lieu_dit_represente": int(none["depicts_place"].sum()),
            "sans_lieu_lieu_precis_verifie": int((none["verified_precise_count"] > 0).sum()),
        }
    return summary


def main():
    """Read the sample with each model and print the comparison."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["qwen3:4b", "gemma3:4b"])
    parser.add_argument("--records", default="data/raw/gold_standard/gold_dataset.json")
    parser.add_argument("--table", default="results/feature_table_v12.pkl",
                        help="Table d'étiquettes qui définit les lieux exacts perdus (défaut: géocodage v12).")
    parser.add_argument("--without-place", type=int, default=60, help="Œuvres « sans lieu » tirées au hasard.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="N'évaluer que les N premières œuvres (essai).")
    parser.add_argument("--per-group", type=int, default=None,
                        help="Sous-échantillon : N œuvres tirées au hasard dans chaque groupe (même graine).")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--output-csv", type=Path, default=Path("results/compare_local_llms.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("results/compare_local_llms.json"))
    args = parser.parse_args()

    try:
        available = {model["name"] for model in requests.get(f"{args.ollama_url}/api/tags", timeout=5).json()["models"]}
    except requests.RequestException:
        sys.exit(f"Ollama ne répond pas sur {args.ollama_url} : lancer `ollama serve`.")
    names = [model.removesuffix("+think") for model in args.models]
    missing = [model for model in names if model not in available and f"{model}:latest" not in available]
    if missing:
        sys.exit(f"Modèles absents d'Ollama : {missing} (`ollama pull <modèle>`).")

    records = load_artwork_records(args.records)
    sample = build_sample(records, pd.read_pickle(args.table), args.without_place, args.seed)
    if args.per_group:
        rng = random.Random(args.seed)
        sample = [
            item for group in ("exact_perdu", "sans_lieu")
            for item in rng.sample([s for s in sample if s[1] == group], min(args.per_group, sum(s[1] == group for s in sample)))
        ]
    sample = sample[: args.limit]
    print(f"[info] {len(sample)} œuvres : "
          f"{sum(g == 'exact_perdu' for _, g, _ in sample)} exact_perdu, {sum(g == 'sans_lieu' for _, g, _ in sample)} sans_lieu")
    nominatim_client, wikidata_client = NominatimClient(), WikidataClient()

    rows = []
    for model in args.models:
        # "qwen3:4b+think" turns Qwen3's reasoning mode on; plain "qwen3:4b" turns it off.
        name, think = (model[: -len("+think")], True) if model.endswith("+think") else (model, None)
        if think is None and name.startswith("qwen3"):
            think = False
        llm = LocalLlm(name, url=args.ollama_url, think=think)
        rows.extend(evaluate(llm, sample, nominatim_client, wikidata_client))
    results = pd.DataFrame(rows)
    results.to_csv(args.output_csv, index=False)
    summary = summarize(results)
    args.output_json.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\n" + pd.DataFrame(summary).to_string())
    print(f"\n[ok] Détail par œuvre : {args.output_csv}")


if __name__ == "__main__":
    main()
