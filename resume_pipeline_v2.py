import argparse
import hashlib
import json
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import pipeline_manager as pm

ROOT = pm.ROOT

STAGES = [
    "05_train_model_v2.py",
    "08_baseline_models_v2.py",
    "09_xgboost_models_v2.py",
    "10_paired_model_comparison_v2.py",
]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_manifest(path, manifest):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    archive = (ROOT / args.archive).resolve()

    if ROOT not in archive.parents:
        raise RuntimeError("Archive must be inside project directory")

    manifest_path = archive / "manifest.json"

    with manifest_path.open() as f:
        manifest = json.load(f)

    if manifest["mode"] != "fixed":
        raise RuntimeError("This resume script supports fixed mode only")

    if manifest["status"] not in ("interrupted", "failed"):
        raise RuntimeError(
            "Experiment must be interrupted or failed to resume"
        )

    completed = {
        stage["script"]
        for stage in manifest.get("stages", [])
        if stage.get("status") == "completed"
    }

    remaining = [stage for stage in STAGES if stage not in completed]

    # Resume only at the beginning of the unfinished suffix.
    if remaining != STAGES[len(completed):]:
        raise RuntimeError("Nonsequential stage history; refusing resume")

    scored = ROOT / "outputs/04_final_scored_v2.parquet"

    if not scored.is_file():
        raise RuntimeError("Fixed-weight scored dataset is missing")

    weights = archive / "fixed_weights.json"

    if not weights.is_file():
        raise RuntimeError("Archived fixed weights are missing")

    if digest(weights) != manifest["weights_sha256"]:
        raise RuntimeError("Archived weights do not match manifest")

    pm.validate_preflight("fixed", str(weights))

    print("\nRESUME EXPERIMENT")
    print("=" * 60)
    print("Archive:", archive)
    print("Completed stages:", sorted(completed))
    print("Remaining stages:", remaining)
    print("Scoring: SKIPPED")

    if not args.apply:
        print("\nDRY RUN: No files changed.")
        return

    manifest["status"] = "running"
    manifest["resumed_at"] = datetime.now().isoformat()
    manifest.pop("error", None)
    manifest.pop("traceback", None)
    save_manifest(manifest_path, manifest)

    try:
        for script in remaining:
            print("\nRUNNING:", script, flush=True)

            stage = {
                "script": script,
                "started_at": datetime.now().isoformat(),
                "status": "running",
            }

            manifest.setdefault("stages", []).append(stage)
            save_manifest(manifest_path, manifest)

            subprocess.run(
                [sys.executable, script],
                cwd=str(ROOT),
                check=True,
            )

            expected = pm.EXPECTED_STAGE_OUTPUTS.get(script, [])
            missing = [
                name for name in expected
                if not (ROOT / name).is_file()
            ]

            if missing:
                raise RuntimeError(
                    "Missing stage outputs: " + ", ".join(missing)
                )

            stage["status"] = "completed"
            stage["finished_at"] = datetime.now().isoformat()
            stage["verified_outputs"] = expected
            save_manifest(manifest_path, manifest)

        manifest["status"] = "completed"
        manifest["finished_at"] = datetime.now().isoformat()
        save_manifest(manifest_path, manifest)

        print("\nRESUMED EXPERIMENT COMPLETED")

    except (Exception, KeyboardInterrupt) as error:
        interrupted = isinstance(error, KeyboardInterrupt)
        status = "interrupted" if interrupted else "failed"

        manifest["status"] = status
        manifest["error"] = (
            "KeyboardInterrupt" if interrupted else str(error)
        )
        manifest["traceback"] = traceback.format_exc()
        manifest["finished_at"] = datetime.now().isoformat()

        for stage in manifest.get("stages", []):
            if stage.get("status") == "running":
                stage["status"] = status
                stage["finished_at"] = datetime.now().isoformat()
                stage["error"] = manifest["error"]

        save_manifest(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
