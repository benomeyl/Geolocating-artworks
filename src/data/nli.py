"""
Zero-shot natural language inference (NLI) scores for place candidates.

A multilingual NLI model reads a premise (the artwork's title and the text
around a mention) and a hypothesis ("This artwork depicts Venice.") and
returns the probability that the premise entails the hypothesis. The model
was never trained on this task: it uses its general understanding of
language. The probabilities become features of the hierarchical model
(filtering_logreg.nli_features).

Scores are cached in SQLite, keyed by model, premise and hypothesis, so a
pair is never computed twice across training, comparison and prediction.
"""

import hashlib
import sqlite3
from pathlib import Path

DEFAULT_NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
DEFAULT_NLI_CACHE = Path("data/processed/nli_cache.sqlite")


class NliScorer:
    """Entailment probabilities of (premise, hypothesis) pairs, with a persistent cache."""

    def __init__(self, model_name=DEFAULT_NLI_MODEL, cache_path=DEFAULT_NLI_CACHE, batch_size=8, max_length=512):
        self.model_name = model_name
        self.cache_path = Path(cache_path)
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None
        self._tokenizer = None
        self._entailment_index = None
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.cache_path) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS nli_cache (key TEXT PRIMARY KEY, entailment REAL)")

    def _key(self, premise, hypothesis):
        """Cache key of a pair for this model."""
        return hashlib.sha1(f"{self.model_name}\n{premise}\n{hypothesis}".encode("utf-8")).hexdigest()

    def _load(self):
        """Load the model on first use (scores already cached need no model)."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.eval()
        labels = {label.lower(): index for index, label in self._model.config.id2label.items()}
        self._entailment_index = labels["entailment"]

    def _compute(self, pairs):
        """Entailment probabilities of `pairs`, computed by the model in batches."""
        import torch

        if self._model is None:
            self._load()
        probabilities = []
        for offset in range(0, len(pairs), self.batch_size):
            batch = pairs[offset:offset + self.batch_size]
            inputs = self._tokenizer(
                [premise for premise, _ in batch],
                [hypothesis for _, hypothesis in batch],
                truncation="only_first",
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = self._model(**inputs).logits
            probabilities.extend(torch.softmax(logits, dim=-1)[:, self._entailment_index].tolist())
        return probabilities

    def entailment(self, pairs):
        """Probability that each premise entails its hypothesis, in the order of `pairs`."""
        keys = [self._key(premise, hypothesis) for premise, hypothesis in pairs]
        with sqlite3.connect(self.cache_path) as connection:
            cached = {}
            for key in set(keys):
                row = connection.execute("SELECT entailment FROM nli_cache WHERE key = ?", (key,)).fetchone()
                if row is not None:
                    cached[key] = row[0]
            missing = [(key, pair) for key, pair in zip(keys, pairs) if key not in cached]
            if missing:
                unique = dict(missing)
                scores = self._compute(list(unique.values()))
                cached.update(zip(unique, scores))
                connection.executemany("INSERT OR REPLACE INTO nli_cache VALUES (?, ?)", zip(unique, scores))
        return [cached[key] for key in keys]
