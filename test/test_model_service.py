"""Inference service tests: fallback behaviour, ensemble blending, explanations."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from app.core.config import settings
from ml.canonical import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES
from ml.features import LEAKY_COLUMNS, X_EDUCATION_MAPPING, clean_frame
from ml.model_service import LeadScoringModel, band_for

ENGAGED_LEAD = {
    "time_on_site_seconds": 1200.0,
    "page_views_per_visit": 5.0,
    "total_visits": 8.0,
}

COLD_LEAD = {
    "time_on_site_seconds": 5.0,
    "page_views_per_visit": 1.0,
    "total_visits": 1.0,
}


class TestFallbackBehaviour(unittest.TestCase):
    """Behaviour when no artifacts are present on disk."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.model = LeadScoringModel(Path(self._tmpdir.name))

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_falls_back_to_heuristic(self) -> None:
        result = self.model.predict(ENGAGED_LEAD)

        self.assertEqual(result.model_name, "heuristic_fallback")
        self.assertEqual(result.centralized_output["inference_mode"], "heuristic_fallback")
        self.assertGreaterEqual(result.probability, 0.0)
        self.assertLessEqual(result.probability, 1.0)

    def test_fallback_reports_no_components(self) -> None:
        """Absent models must not be backfilled with the blended score."""
        result = self.model.predict(ENGAGED_LEAD)

        self.assertEqual(result.centralized_output["components"], {})
        self.assertEqual(result.centralized_output["available_models"], [])

    def test_fallback_clamps_probability(self) -> None:
        result = self.model.predict(
            {"time_on_site_seconds": 1e7, "page_views_per_visit": 1e4, "total_visits": 1e4}
        )

        self.assertLessEqual(result.probability, 1.0)

    def test_explain_is_empty_without_artifacts(self) -> None:
        result = self.model.explain(ENGAGED_LEAD)

        self.assertEqual(result.contributions, [])
        self.assertIn("heuristic", self.model.summarize(result))

    def test_batch_path_works_without_artifacts(self) -> None:
        frame = clean_frame(pd.DataFrame([ENGAGED_LEAD, COLD_LEAD]))

        scores = self.model.predict_frame(frame)

        self.assertEqual(len(scores), 2)
        self.assertGreater(scores[0], scores[1])


class TestBands(unittest.TestCase):
    def test_bands_partition_the_range(self) -> None:
        self.assertEqual(band_for(0.9), "hot")
        self.assertEqual(band_for(0.6), "warm")
        self.assertEqual(band_for(0.3), "cool")
        self.assertEqual(band_for(0.1), "cold")


class TestFeatureDefinitions(unittest.TestCase):
    def test_leaky_columns_are_not_in_the_bundled_mapping(self) -> None:
        """The bundled dataset's post-outcome columns must never be mapped."""
        for column in LEAKY_COLUMNS:
            self.assertNotIn(column, X_EDUCATION_MAPPING)

    def test_clean_frame_maps_placeholder_to_missing(self) -> None:
        frame = clean_frame(pd.DataFrame([{"specialization": "Select", "city": "Mumbai"}]))

        self.assertTrue(frame["specialization"].isna().all())
        self.assertEqual(frame["city"].iloc[0], "Mumbai")

    def test_clean_frame_adds_missing_columns(self) -> None:
        frame = clean_frame(pd.DataFrame([{"total_visits": 3}]))

        self.assertEqual(list(frame.columns), FEATURE_COLUMNS)
        self.assertEqual(frame["total_visits"].iloc[0], 3)

    def test_numeric_and_categorical_do_not_overlap(self) -> None:
        self.assertEqual(set(NUMERIC_FEATURES) & set(CATEGORICAL_FEATURES), set())


@unittest.skipUnless(
    LeadScoringModel(settings.base_artifact_dir).models,
    "base model not trained; run scripts/train_model.py",
)
class TestTrainedModel(unittest.TestCase):
    """Behaviour once the base model exists."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = LeadScoringModel(settings.base_artifact_dir)

    def test_uses_ensemble_inference(self) -> None:
        result = self.model.predict(ENGAGED_LEAD)

        self.assertIn(
            result.centralized_output["inference_mode"],
            {"hybrid_ensemble", "recalibrated_ensemble"},
        )
        self.assertTrue(result.centralized_output["components"])

    def test_components_match_loaded_models(self) -> None:
        result = self.model.predict(ENGAGED_LEAD)

        self.assertEqual(set(result.centralized_output["components"]), set(self.model.models))

    def test_weights_sum_to_one(self) -> None:
        self.assertAlmostEqual(sum(self.model.weights.values()), 1.0, places=6)

    def test_engaged_lead_outscores_cold_lead(self) -> None:
        self.assertGreater(
            self.model.predict(ENGAGED_LEAD).probability,
            self.model.predict(COLD_LEAD).probability,
        )

    def test_partial_payload_scores_without_error(self) -> None:
        result = self.model.predict({"total_visits": 4})

        self.assertGreaterEqual(result.probability, 0.0)
        self.assertLessEqual(result.probability, 1.0)

    def test_batch_matches_single_scoring(self) -> None:
        single = self.model.predict(ENGAGED_LEAD).probability
        batch = self.model.predict_frame(clean_frame(pd.DataFrame([ENGAGED_LEAD])))[0]

        self.assertAlmostEqual(single, batch, places=6)

    def test_explain_returns_ranked_signed_contributions(self) -> None:
        result = self.model.explain(
            {**ENGAGED_LEAD, "origin": "Lead Add Form", "channel": "Reference"}
        )

        self.assertTrue(result.contributions, "expected at least one non-baseline field")
        for item in result.contributions:
            self.assertIn(item["direction"], ("increases", "decreases"))

        impacts = [abs(item["impact"]) for item in result.contributions]
        self.assertEqual(impacts, sorted(impacts, reverse=True), "contributions must be ranked")

    def test_summary_mentions_probability(self) -> None:
        result = self.model.explain(ENGAGED_LEAD)

        self.assertIn("%", self.model.summarize(result))

    def test_training_captured_reference_statistics(self) -> None:
        """Drift monitoring depends on these being written at training time."""
        self.assertTrue(self.model.feature_statistics)


if __name__ == "__main__":
    unittest.main()
