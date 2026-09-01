import unittest
from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from api.main import app

class TestAPIEndpoints(unittest.TestCase):
    """Integration test suite executing HTTP requests on our live FastAPI server."""

    def setUp(self):
        # Instantiate test client
        self.client = TestClient(app)

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")

    def test_prediction_endpoint(self):
        # Configure simulated post payload
        payload = {
            "subject_id": 1001,
            "waveforms": {
                "ECG": [0.1] * 10,
                "ABP": [110.0] * 10,
                "RESP": [18.0] * 10
            },
            "vitals_and_labs": {
                "creatinine": 1.4, "bun": 22.0, "urine_output": 35.0,
                "spo2": 0.96, "respiratory_rate": 18.0, "pao2_fio2": 320.0,
                "heart_rate": 88.0, "mean_arterial_pressure": 72.0, "systolic_bp": 115.0,
                "bilirubin_total": 0.9, "alt": 25.0, "ast": 30.0,
                "gcs": 15.0, "pupil_reaction": 1.0, "temperature": 37.0
            }
        }
        
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn("cascade_alert_probability", data)
        self.assertIn("organ_specific_hazards", data)
        self.assertIn("attention_edge_weights", data)
        self.assertEqual(len(data["organ_specific_hazards"]), 5)

    def test_marl_simulation_endpoints(self):
        # A. Reset sandbox
        response_reset = self.client.post("/api/marl/reset")
        self.assertEqual(response_reset.status_code, 200)
        self.assertEqual(response_reset.json()["status"], "reset_success")

        # B. Step sandbox
        step_payload = {
            "patho_action": [0.1, 0.05, 0.0],
            "therapy_action": [0.2, 0.1, 0.4, 0.0]
        }
        response_step = self.client.post("/api/marl/step", json=step_payload)
        self.assertEqual(response_step.status_code, 200)
        
        step_data = response_step.json()
        self.assertEqual(step_data["current_step"], 1)
        self.assertIn("rewards", step_data)
        self.assertFalse(step_data["terminated"])

if __name__ == "__main__":
    unittest.main()