"""Transaction management for safe pipeline execution."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parent
TRANSACTION_ROOT = ROOT / "archives" / "transactions"


def atomic_json(path, data):
    """Write a manifest using atomic replacement."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")

    with temporary.open("w") as stream:
        json.dump(data, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())

    os.replace(temporary, path)


def create_transaction(stage, root=None):
    """Create an isolated transaction directory."""

    transaction_root = (
        Path(root) if root is not None else TRANSACTION_ROOT
    )

    transaction_root.mkdir(parents=True, exist_ok=True)

    transaction_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + uuid4().hex[:12]
    )

    directory = transaction_root / transaction_id
    directory.mkdir(exist_ok=False)

    for name in ("backup", "staging", "logs"):
        (directory / name).mkdir()

    manifest = {
        "transaction_id": transaction_id,
        "stage": stage,
        "status": "created",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "archive_verified": False,
        "outputs_published": False,
    }

    atomic_json(directory / "manifest.json", manifest)

    return directory, manifest


def update_status(directory, status):
    """Update a transaction's status."""

    path = Path(directory) / "manifest.json"

    with path.open() as stream:
        manifest = json.load(stream)

    manifest["status"] = status
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()

    atomic_json(path, manifest)

    return manifest


from contextlib import contextmanager
import fcntl


@contextmanager
def transaction_lock(root=None):
    """Allow only one cooperating pipeline transaction at a time."""

    transaction_root = (
        Path(root) if root is not None else TRANSACTION_ROOT
    )
    transaction_root.mkdir(parents=True, exist_ok=True)

    lock_path = transaction_root / ".pipeline.lock"

    with lock_path.open("a+") as lock_file:
        try:
            fcntl.flock(
                lock_file.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError:
            raise RuntimeError(
                "Another pipeline transaction is already running."
            )

        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def scan_incomplete_transactions(root=None):
    """Find transactions that may require recovery.

    This function never modifies files.
    """
    transaction_root = (
        Path(root) if root is not None else TRANSACTION_ROOT
    )

    if not transaction_root.exists():
        return []

    completed_statuses = {
        "completed",
        "rolled_back",
    }

    incomplete = []

    for directory in sorted(transaction_root.iterdir()):
        if not directory.is_dir():
            continue

        manifest_path = directory / "manifest.json"

        if not manifest_path.is_file():
            incomplete.append({
                "directory": str(directory),
                "status": "missing_manifest",
            })
            continue

        try:
            with manifest_path.open() as stream:
                manifest = json.load(stream)

            status = manifest.get("status", "unknown")

        except (OSError, ValueError):
            status = "invalid_manifest"

        if status not in completed_statuses:
            incomplete.append({
                "directory": str(directory),
                "status": status,
            })

    return incomplete
