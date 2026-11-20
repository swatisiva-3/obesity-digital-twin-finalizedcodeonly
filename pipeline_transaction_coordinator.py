"""Read-only coordinator for staged pipeline transactions."""

import argparse
import json
from pathlib import Path

from pipeline_transaction import (
    transaction_lock,
    scan_incomplete_transactions,
)
from pipeline_transaction_controller import inspect_transaction
from pipeline_transaction_journal import inspect_journal
from pipeline_recovery_assessment import assess_recovery

ROOT = Path(__file__).resolve().parent


def coordinate(transaction, stage):
    with transaction_lock():
        incomplete = scan_incomplete_transactions()
        target = Path(transaction).resolve()

        blocking = []

        for item in incomplete:
            directory = item.get("directory")

            if directory is None:
                raise RuntimeError(
                    "Incomplete transaction has no directory field: "
                    + json.dumps(item)
                )

            if Path(directory).resolve() != target:
                blocking.append(item)

        if blocking:
            raise RuntimeError(
                "Other incomplete transactions must be resolved "
                "before proceeding:\n"
                + json.dumps(blocking, indent=2)
            )

        result = inspect_transaction(transaction, stage)
        publication = result["publication_plan"]

        if result["decision"] == "NO_OP":
            decision = "NO_OP"
            next_action = (
                "No publication required. Transaction finalization "
                "must separately verify provenance and transaction state."
            )
        elif result["decision"] == "REQUIRES_ARCHIVE_AND_APPROVAL":
            decision = "REQUIRES_APPROVAL"
            next_action = (
                "Archive affected artifacts, verify backups, "
                "handle downstream invalidation, and obtain "
                "explicit approval before publication."
            )
        else:
            raise RuntimeError("Unexpected controller decision")

        return {
            "transaction_id": publication["transaction_id"],
            "stage": stage,
            "decision": decision,
            "changed_outputs": publication["changed_replacements"],
            "invalidated_stages": publication["invalidated_stages"],
            "invalidated_artifacts": publication["invalidated_artifacts"],
            "archive_bytes_required": publication["archive_bytes_required"],
            "incomplete_transactions": incomplete,
            "next_action": next_action,
            "live_publication_enabled": False,
            "coordination_journal": inspect_journal(transaction),
            "recovery_assessment": assess_recovery(
                transaction, ROOT
            ),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transaction", required=True)
    parser.add_argument("--stage", default="09")
    args = parser.parse_args()

    result = coordinate(args.transaction, args.stage)

    print("\nTRANSACTION COORDINATOR")
    print("=" * 65)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
