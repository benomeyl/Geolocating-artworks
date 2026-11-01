from src.data.filtering_logreg import NLI_FEATURE_COLUMNS, nli_features, nli_premise
from src.data.nli import NliScorer


def _occurrence(text, place, field, score=0.9):
    start = text.index(place)
    return {"place": place, "field": field, "source_text": text, "start": start, "end": start + len(place), "score_ner": score}


def test_premise_uses_the_passage_around_a_mention_outside_the_title():
    description = "<p>" + "x" * 400 + " Painted in Rome after a trip.</p>"
    occurrences = [_occurrence(description, "Rome", "description")]
    premise = nli_premise(occurrences, {"title": "View of the Rialto"}, window=20)
    assert premise.startswith("Title: View of the Rialto. Description: ")
    assert "Painted in Rome after a trip." in premise
    assert "<p>" not in premise and "x" * 30 not in premise


def test_premise_of_a_title_mention_adds_the_beginning_of_the_description():
    occurrences = [_occurrence("Vase", "Vase", "title")]
    assert nli_premise(occurrences, {"title": "Vase", "description": "A <em>black-figure</em> jar."}) == (
        "Title: Vase. Description: A black-figure jar."
    )
    assert nli_premise(occurrences, {"title": "Vase"}) == "Title: Vase."


class _RecordingScorer:
    def __init__(self):
        self.pairs = []

    def entailment(self, pairs):
        self.pairs.extend(pairs)
        return [0.1 * (index + 1) for index in range(len(pairs))]


def test_one_feature_per_hypothesis_with_the_place_name():
    scorer = _RecordingScorer()
    features = nli_features([_occurrence("Colorado", "Colorado", "title")], {"title": "Colorado"}, scorer)
    assert list(features) == NLI_FEATURE_COLUMNS
    assert scorer.pairs[0] == ("Title: Colorado.", "This artwork shows a view of the place Colorado.")
    assert features["nli_scene"] == 0.2


def test_scores_are_cached_and_computed_once(tmp_path, monkeypatch):
    scorer = NliScorer(cache_path=tmp_path / "nli.sqlite")
    calls = []

    def fake_compute(pairs):
        calls.append(list(pairs))
        return [0.5] * len(pairs)

    monkeypatch.setattr(scorer, "_compute", fake_compute)
    assert scorer.entailment([("p", "h1"), ("p", "h2"), ("p", "h1")]) == [0.5, 0.5, 0.5]
    assert scorer.entailment([("p", "h2")]) == [0.5]
    assert calls == [[("p", "h1"), ("p", "h2")]]
