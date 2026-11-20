"""Create and verify backups without modifying active files."""

import json
import shutil
from pathlib import Path

from pipeline_archive_plan import sha256_file


def validate_archive_path(root, relative):
    """Reject unsafe paths and symbolic links."""

    root = Path(root).absolute()
    relative = Path(relative)

    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
    ):
        raise RuntimeError("Unsafe archive path: " + str(relative))

    current = root

    for part in relative.parts:
        current = current / part

        if current.is_symlink():
            raise RuntimeError(
                "Symbolic links are not allowed: " + str(current)
            )

    resolved = (root / relative).resolve()

    if not resolved.is_relative_to(root.resolve()):
        raise RuntimeError(
            "Archive path escapes project root: " + str(relative)
        )

    return root / relative



def create_verified_archive(plan, root, destination):
    """Create a verified archive without leaving partial final archives."""
    import os
    import tempfile

    root = Path(root).resolve()
    requested_destination = Path(destination).absolute()
    destination = (
        requested_destination.parent.resolve()
        / requested_destination.name
    )

    if not plan.get("dry_run", False):
        raise RuntimeError("Expected a validated dry-run archive plan.")

    if destination.exists() or destination.is_symlink():
        raise RuntimeError("Archive destination already exists.")

    if destination.parent.resolve() == root or (
        destination.parent.resolve().is_relative_to(root)
        and destination.parent.resolve() != root / "archives"
        and not (root / "archives").resolve()
            .is_relative_to(destination.parent.resolve())
    ):
        raise RuntimeError(
            "Archive destination must not be inside active output directories."
        )

    files = plan["files"]
    if not files:
        raise RuntimeError("Archive plan contains no files.")

    names = [str(item["path"]) for item in files]
    if len(names) != len(set(names)):
        raise RuntimeError("Duplicate archive paths.")

    for item in files:
        source = validate_archive_path(root, item["path"])
        if not source.is_file():
            raise RuntimeError("Source missing: " + str(source))
        if sha256_file(source) != item["sha256"]:
            raise RuntimeError("Source changed since planning: " + str(source))

    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=".archive_build_",
        dir=destination.parent,
    ) as temporary:
        temporary = Path(temporary)
        archived = []

        for item in files:
            relative = Path(item["path"])
            source = validate_archive_path(root, relative)
            target = temporary / relative

            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

            copied_hash = sha256_file(target)
            if copied_hash != item["sha256"]:
                raise RuntimeError(
                    "Backup verification failed: " + str(target)
                )

            archived.append({
                "path": item["path"],
                "sha256": copied_hash,
                "size_bytes": target.stat().st_size,
            })

        manifest = {
            "status": "verified",
            "file_count": len(archived),
            "files": archived,
        }

        (temporary / "archive_manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )

        if destination.exists() or destination.is_symlink():
            raise RuntimeError("Archive destination appeared during copying.")

        os.rename(temporary, destination)

    return manifest
