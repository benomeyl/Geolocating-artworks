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


def _repeated(region, country, region_predictions=20, country_predictions=40):
    """Partitions given as true positives per partition, with a fixed number of predictions."""
    runs = []
    for region_tp, country_tp in zip(region, country):
        by_level = {level: _outer() for level in LEVELS}
        by_level["region"] = _outer(region_tp / region_predictions, region_predictions, region_tp)
        by_level["country"] = _outer(country_tp / country_predictions, country_predictions, country_tp)
        runs.append({"metrics_by_level": by_level})
    return {"runs": runs}


def test_rule_d_needs_the_mean_and_the_pooled_wilson_bound():
    nested = {"metrics_by_level": {level: _outer() for level in LEVELS}}
    # region: 73/80, mean 0.9125 but Wilson lower bound 0.83 < 0.85;
    # country: 145/160, mean 0.906, Wilson lower bound 0.851.
    repeated = _repeated(region=[19, 19, 16, 19], country=[37, 34, 38, 36])

    assert deployable_levels(_model(), nested, 5, 5, repeated) == ["country"]


def test_one_bad_partition_among_many_does_not_reject_a_well_supported_level():
    nested = {"metrics_by_level": {level: _outer() for level in LEVELS}}
    # 19 partitions at 55/60 and one at 50/60 (0.833 < 0.85): 1095/1200 pooled.
    repeated = _repeated(region=[0] * 20, country=[55] * 19 + [50], country_predictions=60)

    assert deployable_levels(_model(), nested, 5, 5, repeated) == ["country"]


def test_without_repeats_the_fixed_partition_decides():
    nested = {"metrics_by_level": {level: _outer() for level in LEVELS}}
    nested["metrics_by_level"]["region"] = _outer(0.92, 25, 23)

    assert deployable_levels(_model(), nested, 5, 5) == ["region"]
