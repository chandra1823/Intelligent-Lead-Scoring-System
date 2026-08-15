"""
End-to-end platform tests.

Each test class runs against a throwaway SQLite database so runs cannot
contaminate each other or the developer's real data.
"""

import csv
import random
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The test database is configured in test/__init__.py, which unittest imports
# before this module — settings must already point at it by the time app.main
# builds the engine.
try:
    from fastapi.testclient import TestClient

    from app.main import app
except Exception:  # pragma: no cover - dependency guard
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


def write_fake_export(path: Path, rows: int = 300) -> Path:
    """A CSV whose column names deliberately differ from the canonical schema."""
    random.seed(11)
    records = []
    for index in range(rows):
        engaged = random.random() < 0.4
        converted = 1 if (engaged and random.random() < 0.7) else (1 if random.random() < 0.1 else 0)
        records.append(
            {
                "Record ID": f"C{index:04d}",
                "Time on Site": random.randint(600, 2500) if engaged else random.randint(0, 180),
                "Number of Sessions": random.randint(4, 12) if engaged else random.randint(0, 2),
                "Pages Per Session": (
                    round(random.uniform(3, 8), 1) if engaged else round(random.uniform(0, 2), 1)
                ),
                "Original Source": random.choice(["Google", "Reference", "Direct Traffic"]),
                "Job Title": random.choice(["Working Professional", "Unemployed", "Student"]),
                "Amount": random.choice([500, 1500, 5000]),
                "Is Converted": converted,
            }
        )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return path


@unittest.skipIf(TestClient is None or app is None, "FastAPI/TestClient not available")
class TestConnectorLifecycle(unittest.TestCase):
    """The clone-to-first-score journey, in the order a user performs it."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._files = tempfile.TemporaryDirectory()
        cls.csv_path = write_fake_export(Path(cls._files.name) / "export.csv")
        cls.client = TestClient(app)
        cls.client.__enter__()

        response = cls.client.post(
            "/v1/sources",
            json={"name": "Test CRM", "kind": "csv", "config": {"path": str(cls.csv_path)}},
        )
        assert response.status_code == 201, response.text
        cls.source_id = response.json()["id"]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)
        cls._files.cleanup()

    def test_01_connectors_are_listed(self) -> None:
        kinds = {c["kind"] for c in self.client.get("/v1/connectors").json()["connectors"]}

        self.assertIn("csv", kinds)
        self.assertIn("hubspot", kinds)

    def test_02_sync_is_blocked_before_mapping_is_confirmed(self) -> None:
        """Importing thousands of rows under a guessed mapping is costly to undo."""
        response = self.client.post(f"/v1/sources/{self.source_id}/sync", json={"limit": 100})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "blocked")

    def test_03_inspect_proposes_a_mapping(self) -> None:
        payload = self.client.post(f"/v1/sources/{self.source_id}/inspect").json()

        self.assertEqual(payload["status"], "ok")
        mapping = payload["proposal"]["mapping"]
        self.assertEqual(mapping.get("Time on Site"), "time_on_site_seconds")
        self.assertEqual(mapping.get("Is Converted"), "converted")
        self.assertEqual(mapping.get("Amount"), "deal_value")

    def test_04_mapping_can_be_confirmed(self) -> None:
        response = self.client.post(
            f"/v1/sources/{self.source_id}/mapping",
            json={"mapping": {}, "accept_proposal": True},
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["coverage"]["has_target"])

    def test_05_sync_imports_and_scores(self) -> None:
        payload = self.client.post(
            f"/v1/sources/{self.source_id}/sync", json={"limit": 1000}
        ).json()

        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["created"], 300)
        self.assertEqual(payload["scored"], 300)

    def test_06_resync_updates_rather_than_duplicates(self) -> None:
        payload = self.client.post(
            f"/v1/sources/{self.source_id}/sync", json={"limit": 1000}
        ).json()

        self.assertEqual(payload["created"], 0, "re-syncing created duplicate leads")

    def test_07_priority_queue_respects_capacity(self) -> None:
        payload = self.client.get("/v1/leads/priority?limit=10").json()

        self.assertEqual(len(payload["leads"]), 10)
        self.assertEqual(payload["summary"]["capacity"], 10)

        ranks = [lead["rank"] for lead in payload["leads"]]
        self.assertEqual(ranks, sorted(ranks))

        priorities = [lead["priority"] for lead in payload["leads"]]
        self.assertEqual(priorities, sorted(priorities, reverse=True))

    def test_08_expected_value_ranking_differs_from_probability(self) -> None:
        by_value = self.client.get("/v1/leads/priority?limit=20&strategy=expected_value").json()
        by_probability = self.client.get("/v1/leads/priority?limit=20&strategy=probability").json()

        top_value = [lead["external_id"] for lead in by_value["leads"]]
        top_probability = [lead["external_id"] for lead in by_probability["leads"]]

        self.assertNotEqual(top_value, top_probability, "deal value had no effect on ranking")

    def test_09_sync_history_is_recorded(self) -> None:
        runs = self.client.get(f"/v1/sources/{self.source_id}/runs").json()["runs"]

        self.assertGreaterEqual(len(runs), 2)
        self.assertTrue(any(run["status"] == "success" for run in runs))

    def test_10_stats_report_a_training_plan(self) -> None:
        stats = self.client.get("/v1/leads/stats").json()

        self.assertEqual(stats["total"], 300)
        self.assertEqual(stats["labelled"], 300)
        self.assertIn(stats["training_plan"]["action"], {"recalibrate", "train_tenant_model"})

    def test_11_training_produces_a_promoted_model(self) -> None:
        payload = self.client.post("/v1/train", json={}).json()

        self.assertEqual(payload["status"], "trained")
        self.assertTrue(payload["promoted"], payload.get("reason"))
        self.assertEqual(payload["training_rows"], 300)

    def test_12_health_reports_the_tenant_model(self) -> None:
        health = self.client.get("/health").json()

        self.assertEqual(health["model_source"], "tenant")
        self.assertIn(health["model_tier"], {"recalibrated", "tenant", "continuous"})

    def test_13_monitoring_returns_drift_and_calibration(self) -> None:
        payload = self.client.get("/v1/monitoring").json()

        self.assertIn("drift", payload)
        self.assertIn("calibration", payload)
        self.assertIn(payload["health"]["status"], {"ok", "warn", "alert", "insufficient_data"})

    def test_14_lift_table_is_computed(self) -> None:
        payload = self.client.get("/v1/monitoring/lift").json()

        self.assertEqual(payload["status"], "ok")
        self.assertGreater(payload["top_decile_lift"], 1.0, "top decile should beat the base rate")

    def test_15_read_only_sources_reject_writeback(self) -> None:
        response = self.client.post(f"/v1/sources/{self.source_id}/push-scores")

        self.assertEqual(response.status_code, 400)
        self.assertIn("read-only", response.json()["detail"])

    def test_16_outcomes_can_be_recorded(self) -> None:
        lead = self.client.get("/v1/leads/priority?limit=1").json()["leads"][0]

        payload = self.client.post(
            "/v1/leads/outcome",
            json={"lead_id": lead["lead_id"], "converted": True, "deal_value": 4200},
        ).json()

        self.assertTrue(payload["converted"])
        self.assertEqual(payload["deal_value"], 4200)

    def test_17_unknown_lead_outcome_is_404(self) -> None:
        response = self.client.post(
            "/v1/leads/outcome", json={"lead_id": "does-not-exist", "converted": True}
        )

        self.assertEqual(response.status_code, 404)

    def test_18_duplicate_source_name_is_rejected(self) -> None:
        response = self.client.post(
            "/v1/sources",
            json={"name": "Test CRM", "kind": "csv", "config": {"path": str(self.csv_path)}},
        )

        self.assertEqual(response.status_code, 409)

    def test_19_bad_connector_config_is_rejected(self) -> None:
        response = self.client.post(
            "/v1/sources",
            json={"name": "Broken", "kind": "csv", "config": {"path": "/nope/missing.csv"}},
        )

        self.assertEqual(response.status_code, 400)

    def test_20_unknown_connector_kind_is_rejected(self) -> None:
        response = self.client.post(
            "/v1/sources", json={"name": "Nope", "kind": "telepathy", "config": {}}
        )

        self.assertEqual(response.status_code, 400)


@unittest.skipIf(TestClient is None or app is None, "FastAPI/TestClient not available")
class TestScoringApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)

    def test_canonical_schema_is_published(self) -> None:
        fields = self.client.get("/schema").json()["fields"]
        names = {field["name"] for field in fields}

        self.assertIn("time_on_site_seconds", names)
        self.assertIn("converted", names)

    def test_predict_accepts_canonical_fields(self) -> None:
        response = self.client.post(
            "/predict",
            json={"time_on_site_seconds": 1200, "page_views_per_visit": 5, "total_visits": 8},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(response.json()["band"], {"hot", "warm", "cool", "cold"})

    def test_predict_still_accepts_the_phase_1_payload(self) -> None:
        """The pre-2.0 request shape must keep working."""
        response = self.client.post(
            "/predict",
            json={
                "total_time_spent_on_website": 1200,
                "page_views_per_visit": 5,
                "total_visits": 8,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertGreater(response.json()["probability"], 0.5)

    def test_batch_scoring_matches_single_scoring(self) -> None:
        lead = {"time_on_site_seconds": 1500, "page_views_per_visit": 6, "total_visits": 9}

        single = self.client.post("/predict", json=lead).json()["probability"]
        batch = self.client.post("/predict/batch", json={"leads": [lead]}).json()["results"][0]

        self.assertAlmostEqual(single, batch["probability"], places=3)

    def test_explain_includes_a_next_best_action(self) -> None:
        payload = self.client.post(
            "/explain",
            json={
                "time_on_site_seconds": 1200,
                "page_views_per_visit": 5,
                "total_visits": 8,
                "occupation": "Working Professional",
            },
        ).json()

        self.assertIn("action", payload["next_best_action"])
        self.assertTrue(payload["contributions"])

    def test_negative_values_are_rejected(self) -> None:
        response = self.client.post("/predict", json={"time_on_site_seconds": -5})

        self.assertEqual(response.status_code, 422)

    def test_probes_respond(self) -> None:
        self.assertEqual(self.client.get("/healthz").status_code, 200)
        self.assertEqual(self.client.get("/readyz").status_code, 200)

    def test_request_id_header_is_returned(self) -> None:
        response = self.client.get("/healthz")

        self.assertIn("X-Request-ID", response.headers)
        self.assertIn("X-Response-Time-ms", response.headers)

    def test_roadmap_document_is_served(self) -> None:
        response = self.client.get("/ui/roadmap")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])


if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(TestClient is None or app is None, "FastAPI/TestClient not available")
class TestTransactionDurability(unittest.TestCase):
    """
    Writes must be committed by the time a service call returns.

    Relying on FastAPI's dependency teardown to commit meant the response was
    sent first: a caller could be told a sync succeeded and then have the very
    next request miss every row. TestClient hid this, so the invariant is
    asserted directly against a second, independent session.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._files = tempfile.TemporaryDirectory()
        cls.csv_path = write_fake_export(Path(cls._files.name) / "durable.csv", rows=120)
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)
        cls._files.cleanup()

    def test_sync_is_durable_before_it_returns(self) -> None:
        from sqlalchemy import func, select

        from app.db.models import Lead, Source
        from app.db.session import SessionLocal

        created = self.client.post(
            "/v1/sources",
            json={"name": "Durable", "kind": "csv", "config": {"path": str(self.csv_path)}},
        )
        self.assertEqual(created.status_code, 201, created.text)
        source_id = created.json()["id"]

        # A fresh session must already see the source the request just created.
        with SessionLocal() as probe:
            self.assertIsNotNone(probe.get(Source, source_id))

        self.client.post(f"/v1/sources/{source_id}/inspect")
        self.client.post(
            f"/v1/sources/{source_id}/mapping", json={"mapping": {}, "accept_proposal": True}
        )

        with SessionLocal() as probe:
            self.assertTrue(probe.get(Source, source_id).mapping_confirmed)

        payload = self.client.post(f"/v1/sources/{source_id}/sync", json={"limit": 500}).json()
        self.assertEqual(payload["status"], "success")

        with SessionLocal() as probe:
            stored = probe.scalar(
                select(func.count(Lead.id)).where(Lead.source_id == source_id)
            )
            scored = probe.scalar(
                select(func.count(Lead.id)).where(
                    Lead.source_id == source_id, Lead.latest_probability.is_not(None)
                )
            )

        self.assertEqual(stored, payload["created"], "synced leads were not committed")
        self.assertEqual(scored, payload["scored"], "scores were not committed")
