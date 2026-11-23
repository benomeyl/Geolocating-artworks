"""Run the place-reading LLM on another machine (GPU) and bring the answers back.

The prompts depend on the local AIC dump (data/raw, not in git), so they are
built here once and shipped as a file; the GPU machine only needs this
repository, `requests` and Ollama. Three steps:

1. On the machine that has the AIC dump:
       python scripts/run_llm_places.py export
   writes results/llm/inputs_gold.jsonl (one prompt per gold artwork, built by
   data/llm_places.artwork_prompt) and results/llm/sample40_ids.txt (the 20 +
   20 artworks of the Qwen3 reasoning trial, same draw as compare_local_llms.py
   --per-group 20).
2. On the GPU machine (Ollama running, model pulled):
       python scripts/run_llm_places.py infer --model gemma3:4b
       python scripts/run_llm_places.py infer --model qwen3:4b --think --ids results/llm/sample40_ids.txt
   writes one answer per line to results/llm/answers_<model>.jsonl as it goes;
   a rerun skips the prompts already answered, so it can be stopped at any time.
3. Back on the first machine:
       python scripts/run_llm_places.py import
   puts every answers_*.jsonl into the local cache (data/processed/
   llm_cache.sqlite); compare_local_llms.py and later steps then read them
   without calling any model.
"""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.llm_places import DEFAULT_OLLAMA_URL, LocalLlm, artwork_prompt  # noqa: E402

LLM_DIR = Path("results/llm")
INPUTS = LLM_DIR / "inputs_gold.jsonl"
SAMPLE_IDS = LLM_DIR / "sample40_ids.txt"


def _prompt_hash(prompt):
    """Short hash identifying a prompt (an answer is only reused for the very same prompt)."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _answers_path(model, think):
    """results/llm/answers_<model>[_think].jsonl, with ":" replaced."""
    return LLM_DIR / f"answers_{model.replace(':', '_')}{'_think' if think else ''}.jsonl"


def _think_flag(model, think):
    """Reasoning flag sent to Ollama: on with --think, off for Qwen3 otherwise, absent for Gemma 3."""
    if think:
        return True
    return False if model.startswith("qwen3") else None


def cmd_export(args):
    """Write the gold prompts and the ids of the 40-artwork trial sample."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pandas as pd
    from compare_local_llms import build_sample
    from run_pipeline import load_artwork_records

    records = load_artwork_records(args.records)
    LLM_DIR.mkdir(parents=True, exist_ok=True)
    with INPUTS.open("w", encoding="utf-8") as handle:
        for record in records:
            prompt = artwork_prompt(record["artwork"])
            handle.write(json.dumps({
                "artwork_id": str(record["artwork"].get("id")), "prompt_hash": _prompt_hash(prompt), "prompt": prompt,
            }, ensure_ascii=False) + "\n")
    sample = build_sample(records, pd.read_pickle(args.table), 60, 0)
    rng = random.Random(0)
    chosen = [
        item for group in ("exact_perdu", "sans_lieu")
        for item in rng.sample([s for s in sample if s[1] == group], 20)
    ]
    SAMPLE_IDS.write_text("\n".join(artwork_id for artwork_id, _, _ in chosen) + "\n", encoding="utf-8")
    print(f"[ok] {len(records)} prompts -> {INPUTS} ; {len(chosen)} ids -> {SAMPLE_IDS}")


def cmd_infer(args):
    """Answer every exported prompt (or those of --ids) with one Ollama model, resuming if interrupted."""
    try:
        requests.get(f"{args.ollama_url}/api/tags", timeout=5).raise_for_status()
    except requests.RequestException:
        sys.exit(f"Ollama ne répond pas sur {args.ollama_url}.")
    inputs = [json.loads(line) for line in INPUTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.ids:
        wanted = set(Path(args.ids).read_text(encoding="utf-8").split())
        inputs = [item for item in inputs if item["artwork_id"] in wanted]
    output = _answers_path(args.model, args.think)
    done = set()
    if output.is_file():
        done = {json.loads(line)["prompt_hash"] for line in output.read_text(encoding="utf-8").splitlines() if line.strip()}
    todo = [item for item in inputs if item["prompt_hash"] not in done][: args.limit]
    print(f"[info] {args.model}{' +think' if args.think else ''}: {len(done)} déjà faits, {len(todo)} à faire -> {output}")

    # A throw-away cache: on this machine the answers live in the JSONL file.
    llm = LocalLlm(args.model, url=args.ollama_url, think=_think_flag(args.model, args.think),
                   cache_path=LLM_DIR / "infer_cache.sqlite", timeout=args.timeout)
    started = time.monotonic()
    with output.open("a", encoding="utf-8") as handle:
        for position, item in enumerate(todo, start=1):
            try:
                answer, seconds = llm.read_prompt(item["prompt"])
            except requests.RequestException as error:
                print(f"  [erreur] {item['artwork_id']}: {error}", flush=True)
                continue
            handle.write(json.dumps({
                "artwork_id": item["artwork_id"], "prompt_hash": item["prompt_hash"], "model": args.model,
                "think": bool(args.think), "seconds": round(seconds, 2), "answer": answer,
            }, ensure_ascii=False) + "\n")
            handle.flush()
            elapsed = time.monotonic() - started
            print(f"  {position}/{len(todo)}  {seconds:5.1f} s  (moyenne {elapsed / position:4.1f} s)  {item['artwork_id']}",
                  flush=True)


def cmd_import(args):
    """Load every results/llm/answers_*.jsonl into the local LLM cache."""
    prompts = {
        item["prompt_hash"]: item["prompt"]
        for item in map(json.loads, INPUTS.read_text(encoding="utf-8").splitlines())
    }
    for path in sorted(LLM_DIR.glob("answers_*.jsonl")):
        imported = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            prompt = prompts.get(item["prompt_hash"])
            if prompt is None:
                continue
            llm = LocalLlm(item["model"], think=_think_flag(item["model"], item["think"]))
            llm.store(prompt, item["answer"], item["seconds"])
            imported += 1
        print(f"[ok] {path.name}: {imported} réponses importées dans le cache")


def main():
    """Command line: export / infer / import."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Écrire les prompts du gold (machine avec le dump AIC).")
    export.add_argument("--records", default="data/raw/gold_standard/gold_dataset.json")
    export.add_argument("--table", default="results/feature_table_v15.pkl")
    infer = commands.add_parser("infer", help="Interroger le modèle Ollama (machine GPU).")
    infer.add_argument("--model", required=True)
    infer.add_argument("--think", action="store_true", help="Mode réflexion (Qwen3).")
    infer.add_argument("--ids", default=None, help="Fichier d'identifiants à traiter (un par ligne).")
    infer.add_argument("--limit", type=int, default=None, help="N'en traiter que N (mesure de vitesse).")
    infer.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    infer.add_argument("--timeout", type=int, default=1800, help="Délai maximal par œuvre, en secondes.")
    commands.add_parser("import", help="Charger les réponses dans le cache local.")
    args = parser.parse_args()
    {"export": cmd_export, "infer": cmd_infer, "import": cmd_import}[args.command](args)


if __name__ == "__main__":
    main()
