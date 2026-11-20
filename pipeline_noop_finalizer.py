"""Safely finalize transactions whose outputs have not changed."""

import argparse
import json
from pathlib import Path

from pipeline_transaction import (
    atomic_json,
    transaction_lock,
)
from pipeline_publication_plan import build_publication_plan
from pipeline_transaction_journal import inspect_journal


def finalize_noop(transaction, stage, apply=False):
    transaction = Path(transaction).resolve()

    with transaction_lock():
        plan = build_publication_plan(transaction, stage)

        if plan["status"] != "no_changes":
            raise RuntimeError(
                "Transaction has changed outputs. "
                "NO_OP finalization refused."
            )

        if inspect_journal(transaction) is not None:
            raise RuntimeError(
                "Coordination journal exists. "
                "Inspect recovery state before finalizing."
            )

        manifest_path = transaction / "manifest.json"
        manifest = json.loads(manifest_path.read_text())

        if manifest.get("transaction_id") != transaction.name:
            raise RuntimeError("Transaction identity mismatch.")

        result = {
            "transaction_id": transaction.name,
            "stage": stage,
            "decision": "NO_OP",
            "verified_outputs": len(plan["replacements"]),
            "active_files_modified": False,
            "finalization_requested": apply,
        }

        if not apply:
            result["status"] = "DRY_RUN"
            return result

        manifest["status"] = "completed"
        manifest["completion_reason"] = "no_output_changes"
        manifest["outputs_published"] = False
        manifest["active_files_modified"] = False

        atomic_json(manifest_path, manifest)

        result["status"] = "COMPLETED"
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transaction", required=True)
    parser.add_argument("--stage", default="09")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    result = finalize_noop(
        args.transaction,
        args.stage,
        apply=args.apply,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
