"""Read-only transaction controller for staged pipeline results.

This module cannot publish, delete, archive, or overwrite research files.
"""

import argparse
import json
from pathlib import Path

from pipeline_execution_plan import validate_plan
from pipeline_preflight import check_stages
from pipeline_publication_plan import build_publication_plan
from pipeline_stages import STAGES
from pipeline_transaction import transaction_lock

ROOT = Path(__file__).resolve().parent


def inspect_transaction(transaction, stage):
    if stage not in STAGES:
        raise ValueError("Unknown stage: " + stage)

    if stage != "09":
        raise RuntimeError(
            "Transaction inspection currently supports Stage 09 only."
        )

    if not validate_plan([stage]):
        raise RuntimeError("Execution safety validation failed.")

    if not check_stages([stage]):
        raise RuntimeError("Preflight validation failed.")

    plan = build_publication_plan(transaction, stage)

    if plan["status"] == "no_changes":
        decision = "NO_OP"
        reason = "All staged output checksums match active outputs."
    elif plan["status"] == "publication_planned":
        decision = "REQUIRES_ARCHIVE_AND_APPROVAL"
        reason = (
            "Changed outputs require verified archival, "
            "downstream invalidation handling, and explicit approval."
        )
    else:
        raise RuntimeError(
            "Unexpected publication status: " + plan["status"]
        )

    return {
        "transaction_id": plan["transaction_id"],
        "stage": stage,
        "decision": decision,
        "reason": reason,
        "publication_plan": plan,
        "execution_enabled": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transaction", required=True)
    parser.add_argument("--stage", default="09")

    args = parser.parse_args()

    with transaction_lock():
        result = inspect_transaction(
            args.transaction,
            args.stage,
        )

    print("\nTRANSACTION CONTROLLER DECISION")
    print("=" * 65)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
