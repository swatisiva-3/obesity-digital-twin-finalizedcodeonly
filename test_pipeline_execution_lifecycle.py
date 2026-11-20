"""Test archive and execution behavior using temporary files only."""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def simulate_run(root, should_fail=False):
    active = root / "outputs" / "result.txt"
    archive = root / "archives" / "result.txt"
    manifest_path = root / "manifest.json"

    active.parent.mkdir(parents=True, exist_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)

    active.write_text("original research result\n")

    # Archive before changing anything.
    original_hash = sha256(active)
    shutil.copy2(active, archive)

    assert sha256(archive) == original_hash

    manifest = {
        "archive_verified": True,
        "status": "running",
        "original_sha256": original_hash,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    try:
        if should_fail:
            raise RuntimeError("Simulated stage failure")

        # Write a replacement into a temporary file first.
        candidate = root / "outputs" / "result.pending"
        candidate.write_text("new research result\n")

        # Publish only after the replacement exists.
        candidate.replace(active)

        manifest["status"] = "completed"

    except Exception:
        manifest["status"] = "failed"

        # Original remains untouched in this failure scenario.
        assert sha256(active) == original_hash

    manifest_path.write_text(json.dumps(manifest, indent=2))

    assert sha256(archive) == original_hash

    return manifest["status"], active.read_text().strip()


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        success = simulate_run(root / "success")
        failure = simulate_run(root / "failure", should_fail=True)

        assert success == ("completed", "new research result")
        assert failure == ("failed", "original research result")

        print("SUCCESS TEST: PASS")
        print("FAILURE TEST: PASS")
        print("ARCHIVE CHECKSUM TEST: PASS")
        print("MANIFEST STATUS TEST: PASS")
        print()
        print("All tests used temporary files.")
        print("No research artifacts modified.")


if __name__ == "__main__":
    main()
