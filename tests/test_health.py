import unittest

from fastapi.testclient import TestClient

from traffic_law_v2.api import create_app
from traffic_law_v2.config import get_settings


class HealthTests(unittest.TestCase):
    def test_health_endpoint(self) -> None:
        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/v1/health")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["provider"], get_settings().model_provider)
        self.assertEqual(body["api_prefix"], "/api/v1")


if __name__ == "__main__":
    unittest.main()
