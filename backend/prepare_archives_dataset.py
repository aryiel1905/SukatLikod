from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# COCO-17 / MoveNet-style keypoint indexes
NOSE = 0
L_SHOULDER = 5
R_SHOULDER = 6
L_HIP = 11
R_HIP = 12

LABEL_MAP = {
    "[correct_posture]": "proper",
    "[INcorrect_posture]": "needs_correction",
}


def parse_meta(stem: str) -> tuple[str, int]:
    # Example: ap_2_10187 or augmented_ap_2_10187
    base = stem.replace("augmented_", "", 1)
    parts = base.split("_")
    if len(parts) >= 3 and parts[-1].isdigit():
        group = "_".join(parts[:-1])
        idx = int(parts[-1])
        return group, idx
    return base, 0


def trunk_angle_deg(mid_shoulder: np.ndarray, mid_hip: np.ndarray) -> float:
    v = mid_shoulder - mid_hip
    L = float(np.linalg.norm(v))
    if L < 1e-6:
        return 0.0
    cos = abs(float(v[1])) / L
    cos = float(np.clip(cos, 0.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def compute_features(kp: np.ndarray) -> tuple[float, float, float]:
    ls = kp[L_SHOULDER]
    rs = kp[R_SHOULDER]
    lh = kp[L_HIP]
    rh = kp[R_HIP]
    nose = kp[NOSE]

    mid_shoulder = (ls + rs) / 2.0
    mid_hip = (lh + rh) / 2.0

    trunk_angle = trunk_angle_deg(mid_shoulder, mid_hip)

    # Normalize head-forward by torso length for scale robustness.
    torso_len = float(
        np.hypot(mid_shoulder[0] - mid_hip[0], mid_shoulder[1] - mid_hip[1]),
    )
    torso_len = max(torso_len, 1e-6)
    head_forward = abs(float(nose[0] - mid_shoulder[0])) / torso_len

    shoulder_tilt = abs(float(ls[1] - rs[1]))
    return trunk_angle, head_forward, shoulder_tilt


def load_single_json(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"Unexpected keypoint JSON shape: {path}")

    item = data[0]
    keypoints = item.get("keypoints")
    if not isinstance(keypoints, list) or len(keypoints) < 13:
        raise ValueError(f"Missing keypoints in: {path}")

    arr = np.array(keypoints, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Invalid keypoint dimensions in: {path}")
    return arr[:, :3] if arr.shape[1] >= 3 else np.c_[arr[:, :2], np.zeros(len(arr))]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert archives_data keypoint JSON + labels to posture dataset CSV.",
    )
    parser.add_argument(
        "--root",
        default="Reference/archives_data/archives_data",
        help="Dataset root containing keypoints/, keypoints_augmented/, labels/.",
    )
    parser.add_argument(
        "--output",
        default="backend/data/posture_dataset.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--include-augmented",
        action="store_true",
        help="Include keypoints_augmented samples.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=30,
        help="Rolling window size for trunk_variance.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    labels_path = root / "labels" / "labels_for_train.csv"
    keypoints_dir = root / "keypoints"
    keypoints_aug_dir = root / "keypoints_augmented"

    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")
    if not keypoints_dir.exists():
        raise FileNotFoundError(f"Missing keypoints directory: {keypoints_dir}")

    labels_df = pd.read_csv(labels_path)
    if "filename" not in labels_df.columns or "label" not in labels_df.columns:
        raise ValueError("labels_for_train.csv must contain filename,label columns.")

    labels_df["base"] = labels_df["filename"].astype(str).str.replace(
        ".jpg",
        "",
        regex=False,
    )
    labels_df["mapped_label"] = labels_df["label"].astype(str).map(LABEL_MAP)
    labels_df = labels_df.dropna(subset=["mapped_label"]).copy()
    label_map = dict(zip(labels_df["base"], labels_df["mapped_label"]))

    rows: list[dict[str, float | str | int]] = []

    def append_from_dir(kp_dir: Path, augmented: bool) -> None:
        for jp in kp_dir.glob("*.json"):
            stem = jp.stem
            base = stem.replace("augmented_", "", 1) if augmented else stem
            if base not in label_map:
                continue

            try:
                kp = load_single_json(jp)
                trunk_angle, head_forward, shoulder_tilt = compute_features(kp)
                group, frame_idx = parse_meta(stem)
            except Exception:
                continue

            rows.append(
                {
                    "group": group,
                    "frame_idx": frame_idx,
                    "trunk_angle": trunk_angle,
                    "head_forward": head_forward,
                    "shoulder_tilt": shoulder_tilt,
                    "label": label_map[base],
                },
            )

    append_from_dir(keypoints_dir, augmented=False)
    if args.include_augmented and keypoints_aug_dir.exists():
        append_from_dir(keypoints_aug_dir, augmented=True)

    if not rows:
        raise RuntimeError("No matched samples found between labels and keypoint JSON files.")

    df = pd.DataFrame(rows)
    df = df.sort_values(["group", "frame_idx"]).reset_index(drop=True)

    df["trunk_variance"] = (
        df.groupby("group")["trunk_angle"]
        .transform(lambda s: s.rolling(window=args.window, min_periods=2).var(ddof=0))
        .fillna(0.0)
    )

    out = df[
        ["trunk_angle", "head_forward", "shoulder_tilt", "trunk_variance", "label"]
    ].copy()
    out = out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Rows: {len(out)}")
    print("Label counts:")
    print(out["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
