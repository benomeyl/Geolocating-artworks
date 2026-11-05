import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_pipeline import deployable_levels  # noqa: E402

LEVELS = ["exact_place", "neighborhood", "city", "region", "country"]


def _outer(precision=None, predictions=0, true_positives=0):
    return {
        "precision": precision, "recall": None, "predictions": predictions, "prediction_groups": predictions,
        "false_positives": predictions - true_positives, "true_positives": true_positives,
    }


def _model():
    levels = {level: {"status": "abstained", "target_precision": 0.9} for level in LEVELS}
    levels["region"] = {"status": "active", "target_precision": 0.9}
    levels["country"] = {"status": "active", "target_precision": 0.9}
    return {"levels": levels}


def _repeated(region, country):
    runs = []
    for region_precision, country_precision in zip(region, country):
        by_level = {level: _outer() for level in LEVELS}
        by_level["region"] = _outer(region_precision, 20)
        by_level["country"] = _outer(country_precision, 40)
        runs.append({"metrics_by_level": by_level})
    return {"runs": runs}


def test_rule_d_needs_the_mean_and_every_partition():
    nested = {"metrics_by_level": {level: _outer() for level in LEVELS}}
    # region: mean 0.91 but one partition at 0.80; country: mean 0.905, worst 0.86.
    repeated = _repeated(region=[0.95, 0.98, 0.80, 0.91], country=[0.92, 0.86, 0.94, 0.90])

    assert deployable_levels(_model(), nested, 5, 5, repeated) == ["country"]


def test_without_repeats_the_fixed_partition_decides():
    nested = {"metrics_by_level": {level: _outer() for level in LEVELS}}
    nested["metrics_by_level"]["region"] = _outer(0.92, 25, 23)

    assert deployable_levels(_model(), nested, 5, 5) == ["region"]
