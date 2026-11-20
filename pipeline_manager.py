#!/usr/bin/env python3

import argparse
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Generated results that can be archived before a new run.
# This is an initial list, not yet a complete pipeline reset.
RESET_FILES = [
    "outputs/04_final_scored_v2.parquet",
    "outputs/optuna_history_v2.png",
    "outputs/09_xgboost_test_predictions_BMI_v2.csv",
    "outputs/09_xgboost_test_predictions_WEIGHT_v2.csv",
    "outputs/09_xgboost_actual_vs_predicted_BMI_v2.png",
    "outputs/09_xgboost_actual_vs_predicted_WEIGHT_v2.png",
    "outputs/10_paired_model_comparison_v2.json",
    "outputs/10_paired_test_predictions_BMI_30D_v2.csv",
    "outputs/10_paired_test_predictions_WEIGHT_30D_KG_v2.csv",
    "outputs/training_leakage_audit_v2.json",
    "outputs/training_leakage_audit_v2.txt",
    "models/training_metrics_v2.json",
    "models/model_comparison_metrics_v2.json",
    "models/xgboost_metrics_v2.json",
    "models/feature_metadata_v2.json",
    "models/train_val_indices_v2.json",
    "models/feature_scaler_v2.joblib",
    "models/mlp_bmi_30d_v2.pth",
    "models/mlp_weight_30d_v2.pth",
]

# These files must never be removed by this utility.
PROTECTED_FILES = [
    "outputs/01_raw_extract.parquet",
    "outputs/02_master_flagged.parquet",
    "outputs/03_scored.parquet",
    "models/train_val_test_indices.json",
    "models/train_val_test_indices_v2.json",
    "models/learned_weights_v2.json",
    "models/clinical_baseline_weights.json",
]

def reset(apply):
    print("\nV2 PIPELINE RESET")
    print("=" * 65)

    existing = []

    for relative in RESET_FILES:
        path = ROOT / relative
        if path.is_file():
            existing.append((relative, path))

    if not existing:
        print("No listed generated files found.")
        return

    print("\nFiles selected for archiving:\n")

    for relative, path in existing:
        print(f"  {relative} ({path.stat().st_size:,} bytes)")

    print(f"\nTotal files: {len(existing)}")

    if not apply:
        print("\nDRY RUN ONLY: No files changed.")
        print("To archive these files, run:")
        print("  python pipeline_manager.py reset --apply")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    archive = ROOT / "archives" / f"pipeline_reset_{timestamp}"

    # Copy and verify all files before removing any originals.
    for relative, source in existing:
        destination = archive / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

        if (
            source.stat().st_size != destination.stat().st_size
            or source.read_bytes() != destination.read_bytes()
        ):
            raise RuntimeError(
                f"Backup verification failed: {relative}"
            )

    print(f"\nVerified archive: {archive.relative_to(ROOT)}")

    for relative, source in existing:
        source.unlink()
        print(f"Archived: {relative}")

    print("\nRESET COMPLETE")
    print("Protected inputs and splits were not modified.")



def score_fixed(weights_path, apply):
    """Regenerate v2 phenotype scores without running Optuna."""
    import json
    import numpy as np
    import pandas as pd
    import scoring_engine

    source = ROOT / "outputs/03_scored.parquet"
    destination = ROOT / "outputs/04_final_scored_v2.parquet"
    weights_file = Path(weights_path).expanduser()

    if not weights_file.is_absolute():
        weights_file = ROOT / weights_file

    if not source.is_file():
        raise FileNotFoundError(source)

    if not weights_file.is_file():
        raise FileNotFoundError(weights_file)

    if weights_file.resolve() == destination.resolve():
        raise ValueError("Weights path cannot be the output dataset.")

    with weights_file.open() as f:
        weights = json.load(f)

    variable_weights = weights["variable_weights"]
    domain_weights = weights["domain_weights"]

    print("\\nFIXED-WEIGHT SCORING")
    print("=" * 65)
    print(f"Input:   {source.relative_to(ROOT)}")
    print(f"Weights: {weights_file}")
    print(f"Output:  {destination.relative_to(ROOT)}")
    print("Optuna:  SKIPPED")

    for domain, values in variable_weights.items():
        if not isinstance(values, dict) or not values:
            raise ValueError(f"Invalid weights for {domain}")

        numbers = np.asarray(list(values.values()), dtype=float)

        if not np.isfinite(numbers).all() or (numbers < 0).any():
            raise ValueError(f"Invalid numeric weights for {domain}")

        if not np.isclose(numbers.sum(), 100.0, atol=1e-4):
            raise ValueError(f"Weights do not sum to 100: {domain}")

    domain_numbers = np.asarray(list(domain_weights.values()), dtype=float)

    if not np.isfinite(domain_numbers).all():
        raise ValueError("Domain weights contain nonfinite values")

    if (domain_numbers < 0).any():
        raise ValueError("Domain weights contain negative values")

    if not np.isclose(domain_numbers.sum(), 100.0, atol=1e-4):
        raise ValueError("Domain weights do not sum to 100")

    if not apply:
        print("\\nDRY RUN ONLY: No files changed.")
        print("To generate scores:")
        print(
            "  python pipeline_manager.py score-fixed "
            "--weights models/learned_weights_v2.json --apply"
        )
        return

    if destination.exists():
        raise FileExistsError(
            f"{destination} already exists. "
            "Archive it using reset before regenerating."
        )

    df = pd.read_parquet(source)

    domain_scores, total_score = (
        scoring_engine.full_scoring_pipeline(
            df, variable_weights, domain_weights
        )
    )

    final_df = df.copy()

    for column in domain_scores.columns:
        final_df[column] = domain_scores[column]

    final_df["TOTAL_PHENOTYPE_SCORE"] = total_score

    temporary = destination.with_name(
        destination.stem + ".temporary.parquet"
    )

    if temporary.exists():
        raise FileExistsError(temporary)

    try:
        final_df.to_parquet(temporary, index=True)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(f"\\nSUCCESS: Saved {len(final_df):,} scored patients.")
    print(f"Output: {destination.relative_to(ROOT)}")


def run_pipeline(mode, weights_path=None, apply=False):
    """Execute the v2 pipeline in dependency order."""
    import subprocess
    import sys

    scored = ROOT / "outputs/04_final_scored_v2.parquet"
    upstream = ROOT / "outputs/03_scored.parquet"
    split = ROOT / "models/train_val_test_indices_v2.json"

    scripts = [
        "05_train_model_v2.py",
        "08_baseline_models_v2.py",
        "09_xgboost_models_v2.py",
        "10_paired_model_comparison_v2.py",
        "audit_training_leakage_v2.py",
    ]

    print("\nV2 PIPELINE RUNNER")
    print("=" * 65)
    print("Mode:", mode.upper())

    required = [upstream, split]

    if mode == "optuna":
        scripts.insert(0, "04_optuna_weight_learning_v2.py")
    else:
        if not weights_path:
            raise ValueError("Fixed mode requires --weights")

        weights = Path(weights_path).expanduser()
        if not weights.is_absolute():
            weights = ROOT / weights

        required.append(weights)

    for item in required:
        if not item.is_file():
            raise FileNotFoundError(f"Required input missing: {item}")

    for script in scripts:
        if not (ROOT / script).is_file():
            raise FileNotFoundError(f"Required script missing: {script}")

    if mode == "fixed":
        print("Scoring: use fixed weights (Optuna skipped)")
        print("Weights:", weights)
    else:
        print("Scoring: optimize weights using Optuna")

    for number, script in enumerate(scripts, 1):
        print(f"  {number}. {script}")

    if not apply:
        print("\nDRY RUN ONLY: Nothing executed.")
        return

    # Refuse to overwrite an existing scored dataset.
    if scored.exists():
        raise FileExistsError(
            "Existing scored dataset detected: "
            "outputs/04_final_scored_v2.parquet. "
            "Archive existing results before running."
        )

    # Refuse to start while other known v2 artifacts exist.
    remaining = [
        relative for relative in RESET_FILES
        if (ROOT / relative).exists()
    ]

    if remaining:
        raise RuntimeError(
            "Existing v2 artifacts detected. "
            "Archive them before running:\n  "
            + "\n  ".join(remaining)
        )

    if mode == "fixed":
        score_fixed(str(weights), apply=True)

        if not scored.is_file():
            raise RuntimeError("Fixed scoring did not produce its output.")

    for script in scripts:
        print("\n" + "=" * 65, flush=True)
        print("RUNNING:", script, flush=True)
        print("=" * 65, flush=True)

        subprocess.run(
            [sys.executable, script],
            cwd=str(ROOT),
            check=True
        )

        if script == "04_optuna_weight_learning_v2.py":
            if not scored.is_file():
                raise RuntimeError(
                    "Optuna did not produce the v2 scored dataset."
                )

    print("\nPIPELINE EXECUTION COMPLETED")



EXPECTED_STAGE_OUTPUTS = {
    "audit_training_leakage_v2.py": [
        "outputs/training_leakage_audit_v2.json",
        "outputs/training_leakage_audit_v2.txt",
    ],
    "04_optuna_weight_learning_v2.py": [
        "outputs/04_final_scored_v2.parquet",
        "models/learned_weights_v2.json",
        "outputs/optuna_history_v2.png",
    ],
    "05_train_model_v2.py": [
        "models/mlp_bmi_30d_v2.pth",
        "models/mlp_weight_30d_v2.pth",
        "models/feature_scaler_v2.joblib",
        "models/feature_metadata_v2.json",
        "models/training_metrics_v2.json",
        "models/train_val_indices_v2.json",
    ],
    "08_baseline_models_v2.py": [
        "models/model_comparison_metrics_v2.json",
    ],
    "09_xgboost_models_v2.py": [
        "models/xgboost_metrics_v2.json",
        "outputs/09_xgboost_test_predictions_BMI_v2.csv",
        "outputs/09_xgboost_test_predictions_WEIGHT_v2.csv",
    ],
    "10_paired_model_comparison_v2.py": [
        "outputs/10_paired_model_comparison_v2.json",
        "outputs/10_paired_test_predictions_BMI_30D_v2.csv",
        "outputs/10_paired_test_predictions_WEIGHT_30D_KG_v2.csv",
    ],
}


def validate_preflight(mode, weights_path=None):
    """Validate essential inputs before an experiment changes files."""
    import json
    import numpy as np
    import pandas as pd

    df = pd.read_parquet(ROOT / "outputs/03_scored.parquet")

    with (ROOT / "models/train_val_test_indices_v2.json").open() as f:
        split = json.load(f)

    groups = [
        split["train_indices"],
        split["validation_indices"],
        split["test_indices"],
    ]

    sets = [set(group) for group in groups]

    if len(df) != 360720:
        raise ValueError("Unexpected dataset row count")

    if [len(group) for group in groups] != [252504, 54108, 54108]:
        raise ValueError("Unexpected split sizes")

    if any(len(group) != len(indices)
           for group, indices in zip(groups, sets)):
        raise ValueError("Duplicate indices inside a split")

    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("Overlapping splits")

    if set.union(*sets) != set(df.index):
        raise ValueError("Split does not cover dataset exactly")

    if mode == "fixed":
        if not weights_path:
            raise ValueError("Fixed mode requires weights")

        with Path(weights_path).open() as f:
            weights = json.load(f)

        variable_weights = weights["variable_weights"]
        domain_weights = weights["domain_weights"]

        if set(variable_weights) != set(domain_weights):
            raise ValueError("Variable/domain weight groups differ")

        for domain, values in variable_weights.items():
            numbers = np.asarray(list(values.values()), dtype=float)

            if (not len(numbers) or
                    not np.isfinite(numbers).all() or
                    (numbers < 0).any() or
                    not np.isclose(numbers.sum(), 100.0, atol=1e-4)):
                raise ValueError(f"Invalid variable weights: {domain}")

        numbers = np.asarray(list(domain_weights.values()), dtype=float)

        if (not np.isfinite(numbers).all() or
                (numbers < 0).any() or
                not np.isclose(numbers.sum(), 100.0, atol=1e-4)):
            raise ValueError("Invalid domain weights")

    print("PREFLIGHT PASS: Dataset, split, and required weights validated.")

def experiment(mode, weights_path=None, apply=False):
    """Safely archive existing results and execute an experiment."""
    import hashlib
    import json
    import subprocess
    import sys
    import traceback

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    archive = ROOT / "archives" / f"experiment_{timestamp}"

    protected = set(PROTECTED_FILES)
    targets = list(RESET_FILES)

    if mode == "optuna":
        targets.append("models/learned_weights_v2.json")

    overlap = set(targets) & (protected - {"models/learned_weights_v2.json"})

    if overlap:
        raise RuntimeError(f"Protected files in archive targets: {overlap}")

    if mode == "fixed":
        if not weights_path:
            raise ValueError("Fixed mode requires --weights")

        weights = Path(weights_path).expanduser()

        if not weights.is_absolute():
            weights = ROOT / weights

        if not weights.is_file():
            raise FileNotFoundError(weights)

        weights = weights.resolve()

        for relative in targets:
            if (ROOT / relative).resolve() == weights:
                raise RuntimeError(
                    "Selected fixed weights would be removed during reset."
                )

    required = [
        ROOT / "outputs/03_scored.parquet",
        ROOT / "models/train_val_test_indices_v2.json",
    ]

    for item in required:
        if not item.is_file():
            raise FileNotFoundError(item)

    scripts = [
        "05_train_model_v2.py",
        "08_baseline_models_v2.py",
        "09_xgboost_models_v2.py",
        "10_paired_model_comparison_v2.py",
        "audit_training_leakage_v2.py",
    ]

    if mode == "optuna":
        scripts.insert(0, "04_optuna_weight_learning_v2.py")

    for script in scripts:
        if not (ROOT / script).is_file():
            raise FileNotFoundError(script)

    existing = [
        relative for relative in targets
        if (ROOT / relative).is_file()
    ]

    print("\nV2 EXPERIMENT")
    print("=" * 65)
    print("Mode:", mode)
    print("Archive target:", archive.relative_to(ROOT))
    print("Files to archive:", len(existing))
    print("Scripts:", ", ".join(scripts))

    if mode == "fixed":
        print("Weights:", weights)

    # Validate inputs even during a dry run.
    validate_preflight(
        mode,
        str(weights) if mode == "fixed" else None
    )

    if not apply:
        print("\nDRY RUN: Preflight passed. No files changed.")
        return

    def sha256(path):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    archive.mkdir(parents=True, exist_ok=False)

    manifest = {
        "mode": mode,
        "started_at": datetime.now().isoformat(),
        "status": "running",
        "archive": str(archive.relative_to(ROOT)),
        "archived_files": {},
        "stages": [],
    }

    if mode == "fixed":
        manifest["weights_path"] = str(weights)
        manifest["weights_sha256"] = sha256(weights)
        shutil.copy2(weights, archive / "fixed_weights.json")

    manifest_path = archive / "manifest.json"

    def save_manifest():
        with manifest_path.open("w") as stream:
            json.dump(manifest, stream, indent=2)

    try:
        # Verify every backup before deleting any source.
        for relative in existing:
            original = ROOT / relative
            destination = archive / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, destination)

            before = sha256(original)
            after = sha256(destination)

            if before != after:
                raise RuntimeError(
                    f"Archive verification failed: {relative}"
                )

            manifest["archived_files"][relative] = before

        save_manifest()

        for relative in existing:
            (ROOT / relative).unlink()

        print("\nVerified backups created.")
        print("Previous generated results archived.")

        if mode == "fixed":
            score_fixed(
                str(archive / "fixed_weights.json"),
                apply=True
            )

            if not (ROOT / "outputs/04_final_scored_v2.parquet").is_file():
                raise RuntimeError("Fixed scoring output missing.")

        for script in scripts:
            print("\nRUNNING:", script, flush=True)

            stage = {
                "script": script,
                "started_at": datetime.now().isoformat(),
                "status": "running",
            }

            manifest["stages"].append(stage)
            save_manifest()

            subprocess.run(
                [sys.executable, script],
                cwd=str(ROOT),
                check=True
            )

            missing = [
                relative
                for relative in EXPECTED_STAGE_OUTPUTS.get(script, [])
                if not (ROOT / relative).is_file()
            ]

            if missing:
                raise RuntimeError(
                    f"Stage {script} completed but outputs are missing: "
                    + ", ".join(missing)
                )

            stage["status"] = "completed"
            stage["finished_at"] = datetime.now().isoformat()
            stage["verified_outputs"] = (
                EXPECTED_STAGE_OUTPUTS.get(script, [])
            )
            save_manifest()

        manifest["status"] = "completed"
        manifest["finished_at"] = datetime.now().isoformat()
        save_manifest()

        print("\nEXPERIMENT COMPLETED")
        print("Manifest:", manifest_path)

    except (Exception, KeyboardInterrupt) as error:
        interrupted = isinstance(error, KeyboardInterrupt)
        status = "interrupted" if interrupted else "failed"

        manifest["status"] = status
        manifest["error"] = (
            "KeyboardInterrupt"
            if interrupted
            else str(error)
        )
        manifest["traceback"] = traceback.format_exc()
        manifest["finished_at"] = datetime.now().isoformat()

        for stage in manifest.get("stages", []):
            if stage.get("status") == "running":
                stage["status"] = status
                stage["finished_at"] = datetime.now().isoformat()
                stage["error"] = manifest["error"]

        save_manifest()
        raise



def preview_selected_stages(stage=None, start=None, end=None, all_stages=False):
    """Preview execution and downstream archive requirements."""
    from pipeline_stages import STAGES, select_stages, downstream_stages

    selected = select_stages(
        stage=stage,
        start=start,
        end=end,
        all_stages=all_stages,
    )

    downstream = downstream_stages(selected)
    affected = selected + downstream

    print("\\nPIPELINE EXECUTION PLAN")
    print("=" * 65)

    print("\\nSTAGES TO EXECUTE:")
    for stage_id in selected:
        print(f"  [{stage_id}] {STAGES[stage_id]['script']}")

    print("\\nDOWNSTREAM STAGES INVALIDATED:")
    if downstream:
        for stage_id in downstream:
            print(f"  [{stage_id}] {STAGES[stage_id]['script']}")
    else:
        print("  None")

    print("\\nEXISTING FILES REQUIRING ARCHIVAL:")

    existing = []
    seen = set()

    for stage_id in affected:
        for relative in STAGES[stage_id]["outputs"]:
            if relative in seen:
                continue

            seen.add(relative)
            path = ROOT / relative

            if path.is_file():
                existing.append((stage_id, relative, path.stat().st_size))

    if existing:
        for stage_id, relative, size in existing:
            print(f"  [{stage_id}] {relative} ({size:,} bytes)")
    else:
        print("  None")

    print(f"\\nTotal files requiring archival: {len(existing)}")

    print("\\nDRY RUN ONLY: No files changed.")
    print("Archiving and execution are not enabled for this command yet.")



def main():
    parser = argparse.ArgumentParser(
        description="Manage obesity digital-twin v2 pipeline outputs."
    )

    subparsers = parser.add_subparsers(dest="command")

    reset_parser = subparsers.add_parser(
        "reset",
        help="Preview or archive generated v2 outputs."
    )

    reset_parser.add_argument(
        "--apply",
        action="store_true",
        help="Archive verified copies and remove active originals."
    )

    fixed_parser = subparsers.add_parser(
        "score-fixed",
        help="Regenerate v2 phenotype scores with fixed weights."
    )

    fixed_parser.add_argument(
        "--weights",
        required=True,
        help="Path to fixed variable/domain weights JSON."
    )

    fixed_parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the scored dataset."
    )

    for command in ("run-fixed", "run-optuna"):
        run_parser = subparsers.add_parser(
            command,
            help="Preview or execute the v2 model pipeline."
        )
        run_parser.add_argument(
            "--weights",
            help="Fixed-weight JSON file (run-fixed only)."
        )
        run_parser.add_argument(
            "--apply",
            action="store_true",
            help="Execute the pipeline."
        )

    experiment_parser = subparsers.add_parser(
        "experiment",
        help="Archive results and run a complete v2 experiment."
    )

    experiment_parser.add_argument(
        "--mode",
        choices=["fixed", "optuna"],
        required=True
    )

    experiment_parser.add_argument(
        "--weights",
        help="Fixed weights JSON (required for fixed mode)."
    )

    experiment_parser.add_argument(
        "--apply",
        action="store_true",
        help="Archive previous outputs and execute."
    )


    selection_parser = subparsers.add_parser(
        "run",
        help="Preview individual, segment, or full pipeline stages."
    )

    selection = selection_parser.add_mutually_exclusive_group(
        required=True
    )

    selection.add_argument("--stage")
    selection.add_argument("--from", dest="start")
    selection.add_argument("--all", action="store_true")

    selection_parser.add_argument("--to", dest="end")

    args = parser.parse_args()

    if args.command == "run":
        from pipeline_stages import select_stages
        from pipeline_execution_plan import validate_plan
        from pipeline_preflight import check_stages

        selected = select_stages(
            stage=args.stage,
            start=args.start,
            end=args.end,
            all_stages=args.all,
        )

        preview_selected_stages(
            stage=args.stage,
            start=args.start,
            end=args.end,
            all_stages=args.all,
        )

        plan_ok = validate_plan(selected)

        # Run file/dataset preflight only after the plan is approved.
        if not plan_ok:
            raise SystemExit(
                "Execution plan blocked. No files changed."
            )

        if not check_stages(selected):
            raise SystemExit(
                "Preflight failed. No files changed."
            )

        print("\nALL CURRENT SAFETY CHECKS PASSED")
        print("Execution is not enabled yet.")
        print("No files changed.")
    elif args.command == "reset":
        reset(args.apply)
    elif args.command == "score-fixed":
        score_fixed(args.weights, args.apply)
    elif args.command == "run-fixed":
        run_pipeline("fixed", args.weights, args.apply)
    elif args.command == "run-optuna":
        run_pipeline("optuna", apply=args.apply)
    elif args.command == "experiment":
        experiment(args.mode, args.weights, args.apply)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
