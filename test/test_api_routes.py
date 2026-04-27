import unittest

try:
    from fastapi.testclient import TestClient
    from app.main import app
except Exception:  # pragma: no cover - dependency/environment guard
    TestClient = None
    app = None


@unittest.skipIf(TestClient is None or app is None, "FastAPI/TestClient dependencies not available")
class TestApiRoutes(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health_endpoint(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("model", data)
        self.assertIn("hybrid_weights", data)

    def test_predict_endpoint(self) -> None:
        payload = {
            "total_time_spent_on_website": 180,
            "page_views_per_visit": 2.4,
            "total_visits": 5,
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("prediction", data)
        self.assertIn("probability", data)
        self.assertIn("label", data)
        self.assertIn("model", data)
        self.assertIn("centralized_output", data)
        self.assertIn(data["prediction"], (0, 1))

        centralized = data["centralized_output"]
        self.assertIn("components", centralized)
        self.assertIn("inference_mode", centralized)

    def test_ui_pages(self) -> None:
        landing = self.client.get("/ui")
        dashboard = self.client.get("/ui/dashboard")
        self.assertEqual(landing.status_code, 200)
        self.assertEqual(dashboard.status_code, 200)
        self.assertIn("text/html", landing.headers.get("content-type", ""))
        self.assertIn("text/html", dashboard.headers.get("content-type", ""))


if __name__ == "__main__":
    unittest.main()
