"""Isolated stage 09 execution; never publishes to active outputs."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from pipeline_stages import STAGES

ROOT = Path(__file__).resolve().parent

INPUTS = {
    "outputs/04_final_scored_v2.parquet",
    "models/train_val_test_indices_v2.json",
}

EXPECTED = STAGES["09"]["outputs"]


def run_stage09(transaction, execute=False):
    transaction = Path(transaction).resolve()
    staging = transaction / "staging"
    work = staging / "workspace"

    if ROOT == transaction or ROOT in transaction.parents:
        pass
    else:
        raise RuntimeError(
            "Transaction must be inside the project directory"
        )

    for relative in INPUTS:
        source = ROOT / relative
        if not source.is_file():
            raise RuntimeError("Missing input: " + relative)

    print("STAGE 09 ISOLATED EXECUTION")
    print("=" * 55)
    print("Workspace:", work)
    print("Inputs:")
    for name in sorted(INPUTS):
        print(" ", name)
    print("Expected outputs:")
    for name in EXPECTED:
        print(" ", name)

    if not execute:
        print("\nDRY RUN ONLY: no files modified.")
        return

    if work.exists() or (
        staging.exists() and any(staging.iterdir())
    ):
        raise RuntimeError(
            "Staging is not empty; refusing to reuse it."
        )

    for name in ("outputs", "models"):
        (work / name).mkdir(parents=True)

    for relative in INPUTS:
        destination = work / relative
        shutil.copy2(ROOT / relative, destination)

    # A separate process loads config, then overrides its directories.
    # The live config.py is never edited.
    launcher = r'''
import config
import runpy
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
workspace = Path(sys.argv[2]).resolve()

config.OUTPUT_DIR = workspace / "outputs"
config.MODEL_DIR = workspace / "models"

runpy.run_path(
    str(root / "09_xgboost_models_v2.py"),
    run_name="__main__",
)
'''

    log_path = transaction / "logs" / "stage09.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w") as log:
        result = subprocess.run(
            [sys.executable, "-c", launcher, str(ROOT), str(work)],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if result.returncode != 0:
        raise RuntimeError(
            "Stage 09 failed; inspect " + str(log_path)
        )

    for relative in EXPECTED:
        generated = work / relative
        if not generated.is_file() or generated.stat().st_size == 0:
            raise RuntimeError(
                "Missing or empty generated output: " + relative
            )

    metrics = work / "models/xgboost_metrics_v2.json"
    with metrics.open() as stream:
        json.load(stream)

    print("\nSTAGE 09 EXECUTION COMPLETED")
    print("All three outputs exist and are nonempty.")
    print("Metrics JSON parsed successfully.")
    print("Active research outputs were not published.")
    print("Log:", log_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transaction", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    run_stage09(args.transaction, execute=args.execute)


if __name__ == "__main__":
    main()
