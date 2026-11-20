"""Read-only assessment of coordinated publication recovery."""

import json
from pathlib import Path

from pipeline_publish import inspect_publication
from pipeline_transaction_journal import inspect_journal


def assess_recovery(transaction, active):
    transaction = Path(transaction).resolve()
    active = Path(active).resolve()

    journal = inspect_journal(transaction)

    if journal is None:
        return {
            "recovery_required": False,
            "reason": "No coordination journal exists.",
            "automatic_recovery_enabled": False,
        }

    status = journal.get("status")

    if status in {"COMPLETED", "ROLLED_BACK"}:
        return {
            "recovery_required": False,
            "reason": "Transaction is already finalized.",
            "automatic_recovery_enabled": False,
        }

    backup = transaction / "backup" / "publication"
    publication_manifest = backup / "publication_manifest.json"

    if not publication_manifest.is_file():
        return {
            "recovery_required": status in {
                "PUBLISHING",
                "PUBLISHED_STALE",
                "RECOVERY_REQUIRED",
            },
            "reason": (
                "No publication manifest exists. "
                "Manual inspection required before recovery."
            ),
            "automatic_recovery_enabled": False,
        }

    publication = json.loads(publication_manifest.read_text())

    publication_status = publication.get("status")

    if publication_status == "rolled_back":
        return {
            "recovery_required": False,
            "coordination_status": status,
            "publication_status": publication_status,
            "reason": (
                "Publication rollback completed. "
                "Coordination journal still requires reconciliation."
            ),
            "coordination_reconciliation_required": True,
            "automatic_recovery_enabled": False,
        }

    if publication_status in {
        "prepared", "publishing", "rolling_back", "completed"
    }:
        report = inspect_publication(
            active,
            backup,
            allow_completed=True,
        )

        return {
            "recovery_required": True,
            "coordination_status": status,
            "publication_status": publication_status,
            "backup_verified": report["recoverable"],
            "files": report["files"],
            "automatic_recovery_enabled": False,
        }

    return {
        "recovery_required": False,
        "coordination_status": status,
        "publication_status": publication_status,
        "reason": "Publication is not in a recoverable state.",
        "automatic_recovery_enabled": False,
    }
