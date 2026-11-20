"""Explicit recovery for interrupted pipeline publication."""

import argparse
import json
from pathlib import Path

from pipeline_publish import recover_publication
from pipeline_recovery_assessment import assess_recovery
from pipeline_transaction import transaction_lock, atomic_json
from pipeline_transaction_journal import (
    inspect_journal,
    transition_journal,
)


def recover_transaction(transaction, active, apply=False, lock_root=None):
    transaction = Path(transaction).resolve()
    active = Path(active).resolve()

    with transaction_lock(root=lock_root):
        journal = inspect_journal(transaction)

        if journal is None:
            raise RuntimeError("Coordination journal missing.")

        if journal.get("transaction_id") != transaction.name:
            raise RuntimeError("Transaction identity mismatch.")

        if journal.get("status") not in {
            "PUBLISHING",
            "PUBLISHED_STALE",
            "RECOVERY_REQUIRED",
        }:
            raise RuntimeError(
                "Transaction is not in a supported recovery state."
            )

        backup = transaction / "backup" / "publication"
        publication_manifest = backup / "publication_manifest.json"

        if not publication_manifest.is_file():
            raise RuntimeError(
                "Publication manifest missing. Manual recovery required."
            )

        publication = json.loads(publication_manifest.read_text())

        expected = {
            item["path"]
            for item in journal.get("changed_outputs", [])
        }
        actual = set(publication.get("originals", {}))

        if not expected or expected != actual:
            raise RuntimeError(
                "Publication inventory does not match coordination journal."
            )

        assessment = assess_recovery(transaction, active)

        if not assessment.get("recovery_required"):
            raise RuntimeError("Recovery is not required.")

        if not assessment.get("backup_verified"):
            raise RuntimeError("Publication backups are not verified.")

        if not apply:
            return {
                "status": "DRY_RUN",
                "transaction_id": transaction.name,
                "recovery_required": True,
                "files_to_restore": sorted(actual),
                "active_files_modified": False,
            }

        if journal["status"] != "RECOVERY_REQUIRED":
            transition_journal(
                transaction,
                "RECOVERY_REQUIRED",
            )

        result = recover_publication(
            active,
            backup,
            allow_completed=True,
        )

        if result["status"] != "rolled_back":
            raise RuntimeError("Publication rollback incomplete.")

        transition_journal(
            transaction,
            "ROLLED_BACK",
            publication_completed=False,
        )

        return {
            "status": "ROLLED_BACK",
            "transaction_id": transaction.name,
            "restored_files": sorted(actual),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transaction", required=True)
    parser.add_argument("--active", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    result = recover_transaction(
        args.transaction,
        args.active,
        apply=args.apply,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
