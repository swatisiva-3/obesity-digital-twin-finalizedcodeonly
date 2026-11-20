"""Recoverable publication of staged pipeline outputs."""

import hashlib
import json
import os
import shutil
from pathlib import Path


def checksum(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")

    with temporary.open("w") as stream:
        json.dump(data, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())

    os.replace(temporary, path)


def publish_outputs(active, staging, backup, filenames,
                    fail_after=None):
    """Publish staged files, rolling back on ordinary exceptions."""

    active = Path(active)
    staging = Path(staging)
    backup = Path(backup)

    filenames = list(filenames)

    if len(set(filenames)) != len(filenames):
        raise ValueError("Duplicate output filenames")

    for name in filenames:
        relative = Path(name)

        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe output path: " + name)

        candidate = staging / relative

        if not candidate.is_file() or candidate.stat().st_size == 0:
            raise RuntimeError("Missing or empty staged output: " + name)

    backup.mkdir(parents=True, exist_ok=True)

    originals = {}

    for name in filenames:
        source = active / name
        destination = backup / name

        if source.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

            original_hash = checksum(source)

            if checksum(destination) != original_hash:
                raise RuntimeError("Backup verification failed: " + name)

            originals[name] = original_hash
        else:
            originals[name] = None

    manifest_path = backup / "publication_manifest.json"

    manifest = {
        "status": "prepared",
        "originals": originals,
        "published": [],
    }

    atomic_json(manifest_path, manifest)

    try:
        manifest["status"] = "publishing"
        atomic_json(manifest_path, manifest)

        for index, name in enumerate(filenames):
            source = staging / name
            destination = active / name

            destination.parent.mkdir(parents=True, exist_ok=True)

            # Copy into the destination directory before replacement.
            pending = destination.with_name(destination.name + ".pending")
            shutil.copy2(source, pending)

            # Write-ahead record: identify the next replacement.
            manifest["pending_file"] = name
            atomic_json(manifest_path, manifest)

            os.replace(pending, destination)

            manifest["published"].append(name)
            manifest["pending_file"] = None
            atomic_json(manifest_path, manifest)

            if fail_after is not None and index + 1 == fail_after:
                raise RuntimeError("Simulated publication failure")

        manifest["status"] = "completed"
        atomic_json(manifest_path, manifest)

    except Exception:
        manifest["status"] = "rolling_back"
        atomic_json(manifest_path, manifest)

        for name in filenames:
            destination = active / name
            original_hash = originals[name]

            if original_hash is None:
                destination.unlink(missing_ok=True)
            else:
                archived = backup / name

                if checksum(archived) != original_hash:
                    raise RuntimeError(
                        "Cannot restore: backup checksum mismatch"
                    )

                restore_pending = destination.with_name(
                    destination.name + ".restore_pending"
                )

                shutil.copy2(archived, restore_pending)
                os.replace(restore_pending, destination)

        manifest["status"] = "rolled_back"
        atomic_json(manifest_path, manifest)
        raise

    return manifest


def inspect_publication(
    active, backup, allow_completed=False
):
    """Read-only assessment of an interrupted publication."""

    active = Path(active)
    backup = Path(backup)
    manifest_path = backup / "publication_manifest.json"

    if not manifest_path.is_file():
        raise RuntimeError("Publication manifest missing")

    with manifest_path.open() as stream:
        manifest = json.load(stream)

    status = manifest.get("status")

    allowed_statuses = {
        "prepared", "publishing", "rolling_back"
    }

    if allow_completed:
        allowed_statuses.add("completed")

    if status not in allowed_statuses:
        raise RuntimeError(
            "Transaction is not in a recoverable state: "
            + str(status)
        )

    originals = manifest.get("originals")

    if not isinstance(originals, dict) or not originals:
        raise RuntimeError("Invalid original-file inventory")

    report = []

    for name, original_hash in originals.items():
        relative = Path(name)

        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
        ):
            raise RuntimeError("Unsafe journal path: " + name)

        current = active / relative
        archived = backup / relative

        if original_hash is None:
            if archived.exists():
                raise RuntimeError(
                    "Unexpected backup for originally absent file: "
                    + name
                )

            report.append({
                "file": name,
                "original_existed": False,
                "current_exists": current.exists(),
                "backup_verified": True,
            })
            continue

        if not isinstance(original_hash, str):
            raise RuntimeError("Invalid checksum: " + name)

        if not archived.is_file():
            raise RuntimeError("Missing backup: " + name)

        if checksum(archived) != original_hash:
            raise RuntimeError("Backup checksum mismatch: " + name)

        report.append({
            "file": name,
            "original_existed": True,
            "current_exists": current.is_file(),
            "backup_verified": True,
            "current_matches_original": (
                current.is_file()
                and checksum(current) == original_hash
            ),
        })

    return {
        "status": status,
        "recoverable": True,
        "files": report,
    }


def recover_publication(
    active, backup, allow_completed=False
):
    """Restore original outputs from a verified publication backup.

    Caller must hold the exclusive transaction lock and ensure
    no other process is modifying the active outputs.
    """

    active = Path(active)
    backup = Path(backup)

    # Validate every backup before changing anything.
    report = inspect_publication(
        active, backup, allow_completed=allow_completed
    )

    manifest_path = backup / "publication_manifest.json"

    with manifest_path.open() as stream:
        manifest = json.load(stream)

    originals = manifest["originals"]

    manifest["status"] = "rolling_back"
    atomic_json(manifest_path, manifest)

    for item in report["files"]:
        name = item["file"]
        destination = active / name
        original_hash = originals[name]

        if original_hash is None:
            destination.unlink(missing_ok=True)
            continue

        archived = backup / name

        destination.parent.mkdir(parents=True, exist_ok=True)

        temporary = destination.with_name(
            destination.name + ".restore_pending"
        )

        shutil.copy2(archived, temporary)
        os.replace(temporary, destination)

        if checksum(destination) != original_hash:
            raise RuntimeError(
                "Restored checksum mismatch: " + name
            )

    # Verify the complete restored state.
    for name, original_hash in originals.items():
        destination = active / name

        if original_hash is None:
            if destination.exists():
                raise RuntimeError(
                    "New output still exists after recovery: " + name
                )
        elif (
            not destination.is_file()
            or checksum(destination) != original_hash
        ):
            raise RuntimeError(
                "Final recovery verification failed: " + name
            )

    manifest["status"] = "rolled_back"
    manifest["recovered"] = True
    atomic_json(manifest_path, manifest)

    return manifest
