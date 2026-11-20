"""Persistent journal for coordinated pipeline publication."""

import json
from pathlib import Path

from pipeline_transaction import atomic_json


def prepare_journal(transaction, publication_plan):
    transaction = Path(transaction).resolve()
    journal_path = transaction / "coordination_journal.json"

    if journal_path.exists():
        raise RuntimeError(
            "Transaction already has a coordination journal. "
            "Inspect it before proceeding."
        )

    if publication_plan["status"] != "publication_planned":
        raise RuntimeError(
            "Only changed-output transactions require publication."
        )

    journal = {
        "transaction_id": publication_plan["transaction_id"],
        "stage": publication_plan["stage"],
        "status": "PREPARED",
        "changed_outputs": publication_plan["changed_replacements"],
        "stale_stages": publication_plan["invalidated_stages"],
        "stale_artifacts": publication_plan["invalidated_artifacts"],
        "archive_verified": False,
        "publication_completed": False,
        "validity_record_written": False,
    }

    atomic_json(journal_path, journal)
    return journal


def inspect_journal(transaction):
    path = Path(transaction) / "coordination_journal.json"

    if not path.is_file():
        return None

    return json.loads(path.read_text())


ALLOWED_TRANSITIONS = {
    "PREPARED": {"ARCHIVED", "ROLLED_BACK"},
    "ARCHIVED": {"PUBLISHING", "ROLLED_BACK"},
    "PUBLISHING": {"PUBLISHED_STALE", "RECOVERY_REQUIRED"},
    "PUBLISHED_STALE": {"COMPLETED", "RECOVERY_REQUIRED"},
    "RECOVERY_REQUIRED": {"ROLLED_BACK"},
    "COMPLETED": set(),
    "ROLLED_BACK": set(),
}


def transition_journal(transaction, new_status, **updates):
    """Persist a validated journal state transition.

    The caller must hold the pipeline transaction lock.
    """
    transaction = Path(transaction).resolve()
    journal_path = transaction / "coordination_journal.json"

    if not journal_path.is_file():
        raise RuntimeError("Coordination journal missing.")

    journal = json.loads(journal_path.read_text())
    current = journal.get("status")

    if new_status not in ALLOWED_TRANSITIONS.get(current, set()):
        raise RuntimeError(
            f"Invalid journal transition: {current} -> {new_status}"
        )

    protected = {"transaction_id", "stage", "status"}
    if protected.intersection(updates):
        raise RuntimeError("Cannot overwrite journal identity or status.")

    journal.update(updates)
    journal["status"] = new_status

    atomic_json(journal_path, journal)
    return journal
