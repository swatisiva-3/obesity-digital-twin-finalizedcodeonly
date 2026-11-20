"""Read-only preflight checks for selected pipeline stages."""

import argparse
from pathlib import Path

from pipeline_stages import STAGES, select_stages

ROOT = Path(__file__).resolve().parent

# Explicit file prerequisites for model stages.
REQUIRED_INPUTS = {
    "05": [
        "outputs/04_final_scored_v2.parquet",
        "models/train_val_test_indices_v2.json",
    ],
    "08": [
        "outputs/04_final_scored_v2.parquet",
        "models/train_val_test_indices_v2.json",
    ],
    "09": [
        "outputs/04_final_scored_v2.parquet",
        "models/train_val_test_indices_v2.json",
    ],
    "10": [
        "outputs/04_final_scored_v2.parquet",
        "models/train_val_test_indices_v2.json",
        "models/feature_metadata_v2.json",
        "models/training_metrics_v2.json",
        "models/feature_scaler_v2.joblib",
        "models/mlp_bmi_30d_v2.pth",
        "models/mlp_weight_30d_v2.pth",
        "outputs/09_xgboost_test_predictions_BMI_v2.csv",
        "outputs/09_xgboost_test_predictions_WEIGHT_v2.csv",
    ],
}

SUPPORTED = set(REQUIRED_INPUTS)



def validate_scored_dataset():
    """Validate the v2 scored dataset against permanent split indices."""
    import json
    import pandas as pd

    dataset_path = ROOT / "outputs/04_final_scored_v2.parquet"
    split_path = ROOT / "models/train_val_test_indices_v2.json"

    scored = pd.read_parquet(dataset_path, columns=["CASEID"])

    with split_path.open() as stream:
        split = json.load(stream)

    groups = [
        split["train_indices"],
        split["validation_indices"],
        split["test_indices"],
    ]

    indices = [index for group in groups for index in group]

    if len(scored) != 360720:
        raise ValueError("Unexpected scored dataset row count")

    if scored["CASEID"].isna().any():
        raise ValueError("Missing CASEIDs")

    if not scored["CASEID"].is_unique:
        raise ValueError("Duplicate CASEIDs")

    if [len(group) for group in groups] != [252504, 54108, 54108]:
        raise ValueError("Unexpected split sizes")

    if len(indices) != len(scored):
        raise ValueError("Split coverage mismatch")

    if len(set(indices)) != len(scored):
        raise ValueError("Duplicate split indices")

    if set(indices) != set(range(len(scored))):
        raise ValueError("Invalid split coverage")

    print("  PASS: V2 dataset and permanent split structure")

def check_stages(selected):
    print("\nPIPELINE PREFLIGHT")
    print("=" * 65)

    failures = []

    for stage_id in selected:
        print(f"\n[{stage_id}] {STAGES[stage_id]['script']}")

        script = ROOT / STAGES[stage_id]["script"]

        if not script.is_file():
            failures.append(f"Missing script: {script}")
            print("  FAIL: Script missing")
            continue

        if stage_id not in SUPPORTED:
            failures.append(
                f"Stage {stage_id} requires additional safety validation"
            )
            print("  BLOCKED: Stage not yet supported")
            continue

        for relative in REQUIRED_INPUTS[stage_id]:
            path = ROOT / relative

            if path.is_file():
                print("  PASS:", relative)
            else:
                failures.append(f"Missing input: {relative}")
                print("  FAIL:", relative)

    print("\n" + "=" * 65)

    if failures:
        print("PREFLIGHT FAILED / BLOCKED")
        for failure in failures:
            print(" -", failure)
        return False

    if any(stage in selected for stage in ("05", "08", "09", "10")):
        try:
            validate_scored_dataset()
        except Exception as error:
            print("PREFLIGHT FAILED:", error)
            return False

    print("PREFLIGHT PASSED: Required files and dataset structure validated.")
    print("No files modified.")
    return True


def main():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stage")
    group.add_argument("--from", dest="start")
    group.add_argument("--all", action="store_true")

    parser.add_argument("--to", dest="end")

    args = parser.parse_args()

    selected = select_stages(
        stage=args.stage,
        start=args.start,
        end=args.end,
        all_stages=args.all,
    )

    if not check_stages(selected):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
