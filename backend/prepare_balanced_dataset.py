from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge posture CSV datasets and rebalance class ratio.",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input CSV files with posture schema.",
    )
    parser.add_argument(
        "--output",
        default="backend/data/posture_dataset.balanced.csv",
        help="Output merged balanced CSV.",
    )
    parser.add_argument(
        "--majority-ratio",
        type=float,
        default=1.2,
        help="Maximum majority/minority ratio after balancing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    args = parser.parse_args()

    frames = [pd.read_csv(Path(p)) for p in args.inputs]
    df = pd.concat(frames, ignore_index=True)

    required = ["trunk_angle", "head_forward", "shoulder_tilt", "trunk_variance", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    df = df[required].dropna().drop_duplicates().reset_index(drop=True)
    counts = df["label"].value_counts()
    if len(counts) < 2:
        raise ValueError("Need at least 2 classes to rebalance.")

    minority_label = counts.idxmin()
    majority_label = counts.idxmax()
    minority_count = int(counts.min())
    majority_target = int(round(minority_count * args.majority_ratio))

    minority_df = df[df["label"] == minority_label]
    majority_df = df[df["label"] == majority_label]

    if len(majority_df) > majority_target:
        majority_df = majority_df.sample(n=majority_target, random_state=args.seed)

    out = pd.concat([minority_df, majority_df], ignore_index=True)
    out = out.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Rows: {len(out)}")
    print("Label counts:")
    print(out["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
