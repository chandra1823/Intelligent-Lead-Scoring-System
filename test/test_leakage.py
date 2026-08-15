"""
Leakage detector tests.

The bundled dataset is the reference case: Phase 1 found `Tags` by hand, and
the detector has to find it automatically without also flagging the legitimate
behavioural features.
"""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ml.leakage import detect_leakage, safe_features

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "Lead Scoring.csv"


class TestLeakageDetector(unittest.TestCase):
    def test_flags_a_perfect_predictor(self) -> None:
        rng = np.random.default_rng(0)
        target = pd.Series(rng.integers(0, 2, 500))
        frame = pd.DataFrame(
            {
                "honest": rng.normal(size=500),
                "leak": target.map({1: "won", 0: "lost"}),
            }
        )

        report = detect_leakage(frame, target)

        self.assertIn("leak", report.quarantined)
        self.assertNotIn("honest", report.quarantined)

    def test_flags_a_numeric_copy_of_the_target(self) -> None:
        rng = np.random.default_rng(1)
        target = pd.Series(rng.integers(0, 2, 500))
        frame = pd.DataFrame({"score_copy": target + rng.normal(0, 0.01, 500)})

        report = detect_leakage(frame, target)

        self.assertIn("score_copy", report.quarantined)
        self.assertEqual(report.findings[0].reason, "solo_auc")

    def test_clean_features_pass(self) -> None:
        rng = np.random.default_rng(2)
        target = pd.Series(rng.integers(0, 2, 600))
        frame = pd.DataFrame(
            {
                "weak_numeric": rng.normal(size=600) + target * 0.3,
                "weak_categorical": rng.choice(["a", "b", "c"], 600),
            }
        )

        report = detect_leakage(frame, target)

        self.assertTrue(report.is_clean, report.summary())

    def test_ignores_tiny_categories(self) -> None:
        """Three leads at 100% is noise, not a leak."""
        rng = np.random.default_rng(3)
        target = pd.Series(rng.integers(0, 2, 800))
        values = rng.choice(["a", "b"], 800).astype(object)
        values[:3] = "rare"
        target.iloc[:3] = 1

        report = detect_leakage(pd.DataFrame({"mostly_fine": values}), target)

        self.assertTrue(report.is_clean, report.summary())

    def test_safe_features_respects_overrides(self) -> None:
        rng = np.random.default_rng(4)
        target = pd.Series(rng.integers(0, 2, 400))
        frame = pd.DataFrame({"leak": target.map({1: "won", 0: "lost"})})

        blocked, _ = safe_features(frame, target, ["leak"])
        self.assertEqual(blocked, [])

        approved, report = safe_features(frame, target, ["leak"], overrides=["leak"])
        self.assertEqual(approved, ["leak"])
        # The override lets it through but the finding is still reported.
        self.assertFalse(report.is_clean)


@unittest.skipUnless(DATASET.exists(), "bundled dataset not available")
class TestAgainstBundledDataset(unittest.TestCase):
    """The regression test for Phase 1's original finding."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = pd.read_csv(DATASET)
        cls.target = cls.raw["Converted"].astype(int)

    def test_rediscovers_the_tags_leak(self) -> None:
        report = detect_leakage(self.raw, self.target, ["Tags"])

        self.assertIn("Tags", report.quarantined)
        finding = report.findings[0]
        self.assertEqual(finding.severity, "critical")

    def test_flags_lead_quality(self) -> None:
        report = detect_leakage(self.raw, self.target, ["Lead Quality"])
        self.assertIn("Lead Quality", report.quarantined)

    def test_does_not_flag_behavioural_features(self) -> None:
        """The features the model legitimately depends on must survive."""
        honest = [
            "Total Time Spent on Website",
            "TotalVisits",
            "Page Views Per Visit",
            "Lead Source",
            "Last Activity",
            "What is your current occupation",
            "City",
            "Country",
        ]

        report = detect_leakage(self.raw, self.target, honest)

        self.assertEqual(report.quarantined, [], report.summary())


if __name__ == "__main__":
    unittest.main()
