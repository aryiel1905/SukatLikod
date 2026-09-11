from __future__ import annotations

import unittest

import numpy as np
from pydantic import ValidationError

from backend import api


class _DeterministicModel:
    classes_ = np.array(["needs_correction", "proper"])

    def predict_proba(self, rows: np.ndarray) -> np.ndarray:
        trunk_angle = float(rows[0][0])
        proper_probability = 0.9 if trunk_angle < 5 else 0.1
        return np.array([[1.0 - proper_probability, proper_probability]])


def _request(trunk_angle: float) -> api.PredictRequest:
    return api.PredictRequest(
        trunk_angle=trunk_angle,
        head_forward=0.02,
        shoulder_tilt=0.01,
        trunk_variance=0.2,
        neck_forward_contour=0,
        upper_back_curvature=0,
        torso_outline_angle=0,
        silhouette_stability=90,
    )


class PredictTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_model = api._model
        self.previous_features = api._features
        api._model = _DeterministicModel()
        api._features = api.DEFAULT_FEATURES

    def tearDown(self) -> None:
        api._model = self.previous_model
        api._features = self.previous_features

    def test_prediction_is_stateless_between_calls(self) -> None:
        first = api.predict(_request(2))
        middle = api.predict(_request(25))
        repeated = api.predict(_request(2))

        self.assertEqual(first.label, "proper")
        self.assertEqual(middle.label, "needs_correction")
        self.assertEqual(repeated.label, first.label)
        self.assertAlmostEqual(repeated.confidence, first.confidence)

    def test_feedback_remains_actionable(self) -> None:
        response = api.predict(_request(25))

        self.assertEqual(response.label, "needs_correction")
        self.assertIn("Straighten", response.feedback)
        self.assertGreaterEqual(response.confidence, 0)
        self.assertLessEqual(response.confidence, 1)

    def test_negative_features_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            _request(-1)


if __name__ == "__main__":
    unittest.main()
