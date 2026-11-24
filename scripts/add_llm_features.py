"""Add the LLM features (data/llm_places.LLM_FEATURE_COLUMNS) to a label table.

Reads the cached answers of one model (data/processed/llm_cache.sqlite,
filled by compare_local_llms.py or run_llm_places.py import) and writes two
tables for scripts/compare_tables.py:

- <output>: the input table plus the LLM columns, to train the model with
  them as features;
- <output>_veto.pkl: the same table where the candidates of artworks the
  LLM calls "no depicted place" are never scored (candidate_level set to
  None), i.e. the LLM can veto a publication but never add one.

    python scripts/add_llm_features.py --table results/feature_table_v15.pkl \
        --output results/feature_table_v15_llm.pkl
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.llm_places import LLM_FEATURE_COLUMNS, LocalLlm, artwork_prompt, candidate_llm_features  # noqa: E402
from run_pipeline import load_artwork_records  # noqa: E402


def main():
    """Build the LLM-feature and veto tables."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", default="results/feature_table_v15.pkl")
    parser.add_argument("--records", default="data/raw/gold_standard/gold_dataset.json")
    parser.add_argument("--model", default="gemma3:4b")
    parser.add_argument("--output", type=Path, default=Path("results/feature_table_v15_llm.pkl"))
    args = parser.parse_args()

    llm = LocalLlm(args.model, think=False if args.model.startswith("qwen3") else None)
    answers = {}
    for record in load_artwork_records(args.records):
        cached = llm.cached(artwork_prompt(record["artwork"]))
        answers[str(record["artwork"].get("id"))] = cached[0] if cached else None
    missing = sum(answer is None for answer in answers.values())
    print(f"[info] {len(answers) - missing} réponses {args.model} en cache, {missing} manquantes (features à 0)")

    table = pd.read_pickle(args.table)
    features = pd.DataFrame([
        candidate_llm_features(place, answers.get(str(artwork_id)))
        for place, artwork_id in zip(table["place"], table["artwork_id"])
    ], index=table.index)
    with_llm = table.copy()
    with_llm[LLM_FEATURE_COLUMNS] = features
    with_llm.attrs = dict(table.attrs)
    with_llm.to_pickle(args.output)

    veto = with_llm.copy()
    veto.attrs = dict(table.attrs)
    vetoed = veto["llm_depicts_place"] == 0
    veto.loc[vetoed, "candidate_level"] = None
    veto_path = args.output.with_name(args.output.stem + "_veto.pkl")
    veto.to_pickle(veto_path)

    print(features.mean().round(3).to_string())
    print(f"[ok] {args.output} ; {veto_path} ({int(vetoed.sum())} candidats sur {len(veto)} mis hors décision)")


if __name__ == "__main__":
    main()
