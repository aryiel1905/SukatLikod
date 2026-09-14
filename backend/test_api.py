from __future__ import annotations

import unittest

import numpy as np
from pydantic import ValidationError

from backend import api


class _DeterministicModel:
    classes_ = np.array(api.CLASSES)
    feature_names_in_ = np.array(api.FEATURES)

    def predict_proba(self, frame):
        marker = float(frame.iloc[0, 0])
        if marker < 1:
            return np.array([[0.05, 0.90, 0.05]])
        if marker < 2:
            return np.array([[0.80, 0.15, 0.05]])
        return np.array([[0.05, 0.10, 0.85]])

    def predict(self, frame):
        probabilities = self.predict_proba(frame)[0]
        return np.array([self.classes_[int(np.argmax(probabilities))]])


def _request(marker: float = 0.0) -> api.PredictRequest:
    values = {name: 0.0 for name in api.FEATURES}
    values[api.FEATURES[0]] = marker
    return api.PredictRequest(**values)


class PredictTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_model = api._model
        api._model = _DeterministicModel()

    def tearDown(self) -> None:
        api._model = self.previous_model

    def test_native_three_class_predictions_drive_output(self) -> None:
        neutral = api.predict(_request(0))
        mild = api.predict(_request(1))
        severe = api.predict(_request(2))

        self.assertEqual(neutral.label, "neutral_posture")
        self.assertEqual(neutral.score, 95)
        self.assertEqual(mild.label, "mild_asymmetry")
        self.assertEqual(severe.label, "severe_misalignment")
        self.assertGreater(mild.score, severe.score)
        self.assertEqual(severe.feedback, api.FEEDBACK["severe_misalignment"])

    def test_requests_require_the_exact_complete_schema(self) -> None:
        values = {name: 0.0 for name in api.FEATURES}
        values.pop(api.FEATURES[-1])
        with self.assertRaises(ValidationError):
            api.PredictRequest(**values)

        values[api.FEATURES[-1]] = 0.0
        values["unexpected"] = 1.0
        with self.assertRaises(ValidationError):
            api.PredictRequest(**values)

    def test_model_schema_and_classes_are_validated(self) -> None:
        api.validate_model(_DeterministicModel())
        invalid = _DeterministicModel()
        invalid.feature_names_in_ = np.array(list(reversed(api.FEATURES)))
        with self.assertRaises(RuntimeError):
            api.validate_model(invalid)


if __name__ == "__main__":
    unittest.main()
