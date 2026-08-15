"""Drift and calibration monitoring tests."""

import unittest

import numpy as np
import pandas as pd

from ml.features import feature_statistics
from ml.monitoring import calibration_report, detect_drift, population_stability_index


def reference_for(values, field="total_visits"):
    frame = pd.DataFrame({field: values})
    return feature_statistics(frame)[field]


class TestPopulationStabilityIndex(unittest.TestCase):
    def test_identical_data_does_not_report_drift(self) -> None:
        """
        A discrete feature must not drift against its own training data.

        Reference quantiles tie on low-cardinality integers (three zeros for
        "visits"), and collapsing them while assuming equal mass per bin
        reported PSI 0.25 for data that had not moved at all.
        """
        rng = np.random.default_rng(0)
        values = pd.Series(rng.integers(0, 15, 500))

        psi = population_stability_index(reference_for(values), values)

        self.assertIsNotNone(psi)
        self.assertLess(psi, 0.1, "identical data must stay under the warn threshold")

    def test_heavily_tied_feature_does_not_report_drift(self) -> None:
        """60% of the mass on a single value must still score as no drift."""
        values = pd.Series([0] * 300 + [1] * 120 + list(range(2, 82)))

        psi = population_stability_index(reference_for(values, "time_on_site_seconds"), values)

        self.assertIsNotNone(psi)
        self.assertLess(psi, 0.1)

    def test_real_shift_is_detected(self) -> None:
        rng = np.random.default_rng(1)
        baseline = pd.Series(rng.integers(0, 15, 500))
        reference = reference_for(baseline)

        self.assertGreater(population_stability_index(reference, baseline + 25), 0.25)

    def test_continuous_feature_round_trips(self) -> None:
        rng = np.random.default_rng(2)
        values = pd.Series(rng.normal(500, 120, 800))

        reference = reference_for(values, "time_on_site_seconds")

        self.assertLess(population_stability_index(reference, values), 0.05)

    def test_categorical_shift_is_detected(self) -> None:
        reference = {"kind": "categorical", "proportions": {"a": 0.8, "b": 0.2}}
        moved = pd.Series(["b"] * 90 + ["a"] * 10)

        self.assertGreater(population_stability_index(reference, moved), 0.25)


class TestDetectDrift(unittest.TestCase):
    def test_insufficient_data_is_reported_not_guessed(self) -> None:
        result = detect_drift({}, pd.DataFrame({"total_visits": [1, 2, 3]}))

        self.assertEqual(result["status"], "insufficient_data")

    def test_unmoved_traffic_is_clean(self) -> None:
        rng = np.random.default_rng(3)
        frame = pd.DataFrame(
            {
                "total_visits": rng.integers(0, 15, 400),
                "time_on_site_seconds": rng.normal(500, 100, 400),
            }
        )

        result = detect_drift(feature_statistics(frame), frame)

        self.assertEqual(result["status"], "ok", result["signals"])


class TestCalibration(unittest.TestCase):
    def test_well_calibrated_scores_pass(self) -> None:
        rng = np.random.default_rng(4)
        probabilities = rng.uniform(0, 1, 3000)
        outcomes = (rng.uniform(0, 1, 3000) < probabilities).astype(int)

        report = calibration_report(probabilities, outcomes)

        self.assertEqual(report["status"], "ok")
        self.assertLess(report["expected_calibration_error"], 0.05)

    def test_overconfident_scores_are_flagged(self) -> None:
        rng = np.random.default_rng(5)
        probabilities = rng.uniform(0.8, 1.0, 1000)
        outcomes = (rng.uniform(0, 1, 1000) < 0.2).astype(int)

        report = calibration_report(probabilities, outcomes)

        self.assertEqual(report["status"], "alert")

    def test_single_class_outcomes_are_reported(self) -> None:
        report = calibration_report([0.5] * 100, [1] * 100)

        self.assertEqual(report["status"], "insufficient_data")


if __name__ == "__main__":
    unittest.main()
