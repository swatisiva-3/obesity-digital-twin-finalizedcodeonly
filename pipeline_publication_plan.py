"""Read-only publication planning for staged pipeline outputs."""

import hashlib
import json
from pathlib import Path

from pipeline_stages import STAGES, downstream_stages
from pipeline_archive_plan import build_archive_plan

ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def build_publication_plan(transaction, stage):
    transaction = Path(transaction).resolve()
    stage = str(stage)

    if stage not in STAGES:
        raise ValueError("Unknown pipeline stage: " + stage)

    transaction_root = (
        ROOT / "archives" / "transactions"
    ).resolve()

    if transaction.parent != transaction_root:
        raise RuntimeError(
            "Transaction must be a direct child of the transaction directory."
        )

    manifest_path = transaction / "manifest.json"

    if not manifest_path.is_file():
        raise RuntimeError("Transaction manifest missing.")

    manifest = json.loads(manifest_path.read_text())

    if manifest.get("stage") != stage:
        raise RuntimeError("Transaction stage mismatch.")

    if manifest.get("status") != "staged":
        raise RuntimeError(
            "Transaction must have status 'staged'."
        )

    if manifest.get("outputs_published") is not False:
        raise RuntimeError(
            "Transaction has already been published or has unknown state."
        )

    workspace = transaction / "staging" / "workspace"

    replacements = []

    for relative in STAGES[stage]["outputs"]:
        staged = workspace / relative

        if not staged.is_file() or staged.stat().st_size == 0:
            raise RuntimeError(
                "Missing or empty staged output: " + relative
            )

        active = ROOT / relative

        replacements.append({
            "path": relative,
            "staged_sha256": sha256(staged),
            "staged_size": staged.stat().st_size,
            "active_sha256": (
                sha256(active) if active.is_file() else None
            ),
        })

    changed_replacements = [
        item for item in replacements
        if item["staged_sha256"] != item["active_sha256"]
    ]

    if not changed_replacements:
        return {
            "transaction_id": manifest["transaction_id"],
            "stage": stage,
            "status": "no_changes",
            "replacements": replacements,
            "changed_replacements": [],
            "invalidated_stages": [],
            "invalidated_artifacts": [],
            "affected_artifact_count": 0,
            "archive_bytes_required": 0,
            "publication_enabled": False,
            "active_files_modified": False,
        }

    downstream = downstream_stages([stage])

    archive_plan = build_archive_plan([stage])

    invalidated = [
        item for item in archive_plan["files"]
        if item["stage"] in downstream
    ]

    return {
        "transaction_id": manifest["transaction_id"],
        "stage": stage,
        "status": "publication_planned",
        "replacements": replacements,
        "changed_replacements": changed_replacements,
        "invalidated_stages": downstream,
        "invalidated_artifacts": invalidated,
        "affected_artifact_count": len(archive_plan["files"]),
        "archive_bytes_required": archive_plan["total_bytes"],
        "publication_enabled": False,
        "active_files_modified": False,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--transaction", required=True)
    parser.add_argument("--stage", required=True)

    args = parser.parse_args()

    plan = build_publication_plan(
        args.transaction,
        args.stage,
    )

    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
