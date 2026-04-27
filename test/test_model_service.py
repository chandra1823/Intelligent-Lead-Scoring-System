import tempfile
import unittest

from ml.model_service import LeadScoringModel


class TestLeadScoringModel(unittest.TestCase):
    def test_fallback_prediction_range_and_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = LeadScoringModel(f"{tmpdir}/missing_model.pkl")
            result = model.predict(
                total_time_spent_on_website=180.0,
                page_views_per_visit=2.4,
                total_visits=5.0,
            )

        self.assertIn(result.prediction, (0, 1))
        self.assertGreaterEqual(result.probability, 0.0)
        self.assertLessEqual(result.probability, 1.0)
        self.assertIn(result.model_name, ("hybrid_fallback", "hybrid_partial_fallback", "hybrid_rf_xgb_lgbm"))

        self.assertIn("centralized_output", result.__dict__)
        self.assertIn("components", result.centralized_output)
        self.assertIn("inference_mode", result.centralized_output)
        self.assertIn("random_forest", result.centralized_output["components"])
        self.assertIn("xgboost", result.centralized_output["components"])
        self.assertIn("lightgbm", result.centralized_output["components"])

    def test_fallback_clamps_probability_for_large_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = LeadScoringModel(f"{tmpdir}/missing_model.pkl")
            result = model.predict(
                total_time_spent_on_website=10_000.0,
                page_views_per_visit=500.0,
                total_visits=1_000.0,
            )

        self.assertEqual(result.probability, 1.0)
        self.assertEqual(result.prediction, 1)


if __name__ == "__main__":
    unittest.main()
