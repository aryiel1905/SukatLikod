from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


LABEL_MAP = {
    "TUP": "proper",
    "TLF": "needs_correction",
    "TLB": "needs_correction",
    "TLR": "needs_correction",
    "TLL": "needs_correction",
}

REQUIRED_COLUMNS = [
    "subject",
    "upperbody_label",
    "nose_z",
    "left_shoulder_x",
    "left_shoulder_y",
    "left_shoulder_z",
    "right_shoulder_x",
    "right_shoulder_y",
    "right_shoulder_z",
    "left_hip_x",
    "left_hip_y",
    "left_hip_z",
    "right_hip_x",
    "right_hip_y",
    "right_hip_z",
]


def trunk_angle_deg(mid_shoulder: np.ndarray, mid_hip: np.ndarray) -> np.ndarray:
    v = mid_shoulder - mid_hip
    norm = np.linalg.norm(v, axis=1)
    norm = np.where(norm < 1e-6, 1e-6, norm)
    cos = np.abs(v[:, 1]) / norm
    cos = np.clip(cos, 0.0, 1.0)
    return np.degrees(np.arccos(cos))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert reference keypoint CSV to SukatLikod training dataset schema.",
    )
    parser.add_argument(
        "--input",
        default="Reference/data.csv",
        help="Input reference CSV with keypoint columns.",
    )
    parser.add_argument(
        "--output",
        default="backend/data/posture_dataset.csv",
        help="Output posture dataset CSV.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=30,
        help="Rolling window size for trunk_variance.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    df = pd.read_csv(input_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    mid_shoulder = np.column_stack(
        [
            (df["left_shoulder_x"] + df["right_shoulder_x"]) / 2.0,
            (df["left_shoulder_y"] + df["right_shoulder_y"]) / 2.0,
            (df["left_shoulder_z"] + df["right_shoulder_z"]) / 2.0,
        ],
    )
    mid_hip = np.column_stack(
        [
            (df["left_hip_x"] + df["right_hip_x"]) / 2.0,
            (df["left_hip_y"] + df["right_hip_y"]) / 2.0,
            (df["left_hip_z"] + df["right_hip_z"]) / 2.0,
        ],
    )

    trunk_angle = trunk_angle_deg(mid_shoulder, mid_hip)
    head_forward = np.abs(df["nose_z"] - mid_shoulder[:, 2])
    shoulder_tilt = np.abs(df["left_shoulder_y"] - df["right_shoulder_y"])

    out = pd.DataFrame(
        {
            "subject": df["subject"],
            "upperbody_label": df["upperbody_label"].astype(str),
            "trunk_angle": trunk_angle,
            "head_forward": head_forward,
            "shoulder_tilt": shoulder_tilt,
        },
    )

    out["label"] = out["upperbody_label"].map(LABEL_MAP)
    out = out.dropna(subset=["label"]).copy()

    out["trunk_variance"] = (
        out.groupby(["subject", "upperbody_label"])["trunk_angle"]
        .transform(lambda s: s.rolling(window=args.window, min_periods=2).var(ddof=0))
        .fillna(0.0)
    )
    out["neck_forward_contour"] = 0.0
    out["upper_back_curvature"] = 0.0
    out["torso_outline_angle"] = 0.0
    out["silhouette_stability"] = 0.0

    final = out[
        [
            "trunk_angle",
            "head_forward",
            "shoulder_tilt",
            "trunk_variance",
            "neck_forward_contour",
            "upper_back_curvature",
            "torso_outline_angle",
            "silhouette_stability",
            "label",
        ]
    ].copy()

    final = final.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Rows: {len(final)}")
    print("Label counts:")
    print(final["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
