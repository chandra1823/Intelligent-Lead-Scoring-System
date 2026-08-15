"""
Phase 1 compatibility tests.

The scoring endpoints predate the platform layer and are consumed by the
dashboard, so their contract is pinned here separately from the newer
platform tests.
"""

import unittest

try:
    from fastapi.testclient import TestClient

    from app.main import app
except Exception:  # pragma: no cover - dependency/environment guard
    TestClient = None
    app = None


def _require_test_database() -> None:
    """
    Fail loudly rather than writing to the developer's real database.

    test/__init__.py points the engine at a temporary file, but unittest only
    imports it when `test` is treated as a package. Run the suite as:

        python -m unittest discover -s test -t .

    Without -t the start directory becomes the top-level directory, the package
    __init__ never runs, and these tests would sync leads and train models
    straight into data/leadscoring.db.
    """
    from app.core.config import settings

    if "leadscoring-tests" not in settings.database_url:
        raise RuntimeError(
            "Tests are pointed at a non-test database "
            f"({settings.database_url}). Run: python -m unittest discover -s test -t ."
        )


_require_test_database()

LEGACY_PAYLOAD = {
    "total_time_spent_on_website": 1200,
    "page_views_per_visit": 5.0,
    "total_visits": 8,
}

CANONICAL_PAYLOAD = {
    "time_on_site_seconds": 1200,
    "page_views_per_visit": 5.0,
    "total_visits": 8,
    "origin": "Lead Add Form",
    "channel": "Reference",
    "last_activity": "SMS Sent",
    "occupation": "Working Professional",
    "city": "Mumbai",
}


@unittest.skipIf(TestClient is None or app is None, "FastAPI/TestClient dependencies not available")
class TestApiRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)

    def test_health_endpoint(self) -> None:
        data = self.client.get("/health").json()

        self.assertEqual(data["status"], "ok")
        self.assertIn("model", data)
        self.assertIn("hybrid_weights", data)
        self.assertIn("artifacts_loaded", data)
        self.assertIn("model_tier", data)

    def test_metrics_endpoint(self) -> None:
        data = self.client.get("/metrics").json()

        self.assertIn("metrics", data)
        self.assertIn("model_version", data)
        self.assertIn("leakage_report", data)

    def test_predict_with_legacy_payload(self) -> None:
        response = self.client.post("/predict", json=LEGACY_PAYLOAD)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn(data["prediction"], (0, 1))
        self.assertGreaterEqual(data["probability"], 0.0)
        self.assertLessEqual(data["probability"], 1.0)
        self.assertIn(data["label"], ("likely_to_convert", "unlikely_to_convert"))
        self.assertIn("centralized_output", data)

    def test_predict_with_canonical_payload(self) -> None:
        response = self.client.post("/predict", json=CANONICAL_PAYLOAD)

        self.assertEqual(response.status_code, 200)
        self.assertIn(response.json()["prediction"], (0, 1))

    def test_predict_rejects_negative_values(self) -> None:
        response = self.client.post(
            "/predict", json={**LEGACY_PAYLOAD, "total_time_spent_on_website": -5}
        )

        self.assertEqual(response.status_code, 422)

    def test_predict_accepts_a_partial_payload(self) -> None:
        """
        Every field is optional in 2.0 — omitted fields fall back to the
        training baseline rather than failing the request.
        """
        response = self.client.post("/predict", json={"total_visits": 3})

        self.assertEqual(response.status_code, 200)

    def test_predict_rejects_a_non_numeric_value(self) -> None:
        response = self.client.post("/predict", json={"total_visits": "many"})

        self.assertEqual(response.status_code, 422)

    def test_explain_endpoint(self) -> None:
        data = self.client.post("/explain", json=CANONICAL_PAYLOAD).json()

        self.assertIn("summary", data)
        self.assertIsInstance(data["contributions"], list)
        for item in data["contributions"]:
            self.assertIn("feature", item)
            self.assertIn(item["direction"], ("increases", "decreases"))

    def test_explain_summary_reflects_this_lead(self) -> None:
        data = self.client.post("/explain", json=CANONICAL_PAYLOAD).json()

        self.assertIn(f"{data['probability'] * 100:.1f}%", data["summary"])

    def test_ui_routes_serve_html(self) -> None:
        for path in ("/ui", "/ui/dashboard", "/ui/roadmap"):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertIn("text/html", response.headers["content-type"])

    def test_static_assets_are_mounted(self) -> None:
        self.assertEqual(self.client.get("/frontend/dashboard.js").status_code, 200)


if __name__ == "__main__":
    unittest.main()
