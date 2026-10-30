import unittest
import json
import os
from app import app
from model import analyze, extract_url_features_dict, predict_url, predict_text

class TestPhishGuardAI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_01_feature_extraction(self):
        feats = extract_url_features_dict("https://accounts.google.com/signin")
        self.assertEqual(feats["https"], 1)
        self.assertEqual(feats["ip_address"], 0)
        self.assertGreater(feats["url_length"], 15)

    def test_02_url_prediction_phishing(self):
        phish_url = "http://192.168.1.50/secure-chase-banking/login.php"
        risk = predict_url(phish_url)
        self.assertIsNotNone(risk)
        self.assertGreater(risk, 0.6)

    def test_03_url_prediction_legit(self):
        legit_url = "https://google.com/about"
        risk = predict_url(legit_url)
        self.assertIsNotNone(risk)
        self.assertLess(risk, 0.4)

    def test_04_text_prediction_phishing(self):
        phish_text = "URGENT: Your bank account has been suspended. Verify your credentials immediately: http://phish.xyz"
        risk = predict_text(phish_text)
        self.assertIsNotNone(risk)
        self.assertGreater(risk, 0.7)

    def test_05_text_prediction_legit(self):
        legit_text = "Hi team, our sprint standup is scheduled for Monday at 10 AM. Please update your tickets."
        risk = predict_text(legit_text)
        self.assertIsNotNone(risk)
        self.assertLess(risk, 0.3)

    def test_06_combined_analyze_function(self):
        res = analyze(
            url="http://192.168.1.1/secure-login",
            message="Urgent action required! Verify your password immediately."
        )
        self.assertEqual(res["verdict"], "PHISHING")
        self.assertEqual(res["level"], "HIGH")
        self.assertGreater(res["risk"], 70.0)
        self.assertGreater(len(res["reasons"]), 0)

    def test_07_flask_home_route(self):
        response = self.app.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"PhishGuard", response.data)

    def test_08_flask_predict_api(self):
        payload = {
            "url": "https://coursera.org",
            "message": "Welcome to your class!"
        }
        response = self.app.post("/predict", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["verdict"], "LEGITIMATE")

    def test_09_flask_benchmark_api(self):
        response = self.app.get("/api/benchmark")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("url_models_benchmark", data)
        self.assertIn("text_models_benchmark", data)
        self.assertEqual(len(data["url_models_benchmark"]), 4)
        self.assertEqual(len(data["text_models_benchmark"]), 4)

    def test_10_results_images_exist(self):
        images = [
            "url_model_comparison.png",
            "text_model_comparison.png",
            "roc_curves.png",
            "confusion_matrices.png",
            "feature_importance.png"
        ]
        for img in images:
            path = os.path.join("results", img)
            self.assertTrue(os.path.exists(path), f"Missing image: {path}")

    def test_11_flask_url_info_api(self):
        response = self.app.get("/api/url-info?url=https://github.com/torvalds/linux")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["scheme"], "https")
        self.assertEqual(data["host"], "github.com")
        self.assertTrue(data["is_https"])
        self.assertFalse(data["is_ip"])

if __name__ == "__main__":
    unittest.main()
