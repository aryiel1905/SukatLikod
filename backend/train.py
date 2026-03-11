from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "trunk_angle",
    "head_forward",
    "shoulder_tilt",
    "trunk_variance",
    "neck_forward_contour",
    "upper_back_curvature",
    "torso_outline_angle",
    "silhouette_stability",
]
LABEL = "label"


def validate_dataset(df: pd.DataFrame) -> None:
    required = FEATURES + [LABEL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    if df[LABEL].nunique() < 2:
        raise ValueError("Dataset must include at least two classes in 'label'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SukatLikod posture classifier.")
    parser.add_argument(
        "--data",
        default="backend/data/posture_dataset.csv",
        help="Path to labeled CSV dataset.",
    )
    parser.add_argument(
        "--model-out",
        default="backend/model/posture_model.joblib",
        help="Path to save trained model.",
    )
    parser.add_argument(
        "--meta-out",
        default="backend/model/model_meta.json",
        help="Path to save model metadata.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set ratio.",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Create it from backend/data/posture_dataset.sample.csv."
        )

    df = pd.read_csv(data_path)
    validate_dataset(df)

    X = df[FEATURES]
    y = df[LABEL].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=250,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))

    model_path = Path(args.model_out)
    meta_path = Path(args.meta_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)

    metadata = {
        "features": FEATURES,
        "classes": sorted(y.unique().tolist()),
        "model_type": "RandomForestClassifier",
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
