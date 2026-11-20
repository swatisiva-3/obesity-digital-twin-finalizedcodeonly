"""Verified, non-destructive archival of pipeline outputs."""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from pipeline_archive_plan import (
    ROOT,
    build_archive_plan,
    sha256_file,
)
from pipeline_stages import select_stages


def create_archive(plan):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    archive = ROOT / "archives" / f"stage_run_{timestamp}"

    if not plan["files"]:
        print("No existing files to archive.")
        return None

    archive.mkdir(parents=True, exist_ok=False)

    manifest = {
        **plan,
        "archive_created_at": datetime.now().isoformat(),
        "status": "copying",
        "originals_preserved": True,
    }

    manifest_path = archive / "manifest.json"

    try:
        for item in plan["files"]:
            source = ROOT / item["path"]
            destination = archive / item["path"]

            # Detect modifications between planning and copying.
            if (
                source.stat().st_size != item["size_bytes"]
                or sha256_file(source) != item["sha256"]
            ):
                raise RuntimeError(
                    f"Source changed since planning: {item['path']}"
                )

            destination.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            shutil.copy2(source, destination)

            if sha256_file(destination) != item["sha256"]:
                raise RuntimeError(
                    f"Archive verification failed: {item['path']}"
                )

            print("VERIFIED:", item["path"])

        manifest["status"] = "verified"
        manifest_path.write_text(
            json.dumps(manifest, indent=2)
        )

        print("\nARCHIVE VERIFIED")
        print("Location:", archive.relative_to(ROOT))
        print("Files:", len(plan["files"]))
        print("Original files remain untouched.")

        return archive

    except Exception:
        manifest["status"] = "failed"
        manifest_path.write_text(
            json.dumps(manifest, indent=2)
        )
        raise


def main():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stage")
    group.add_argument("--from", dest="start")
    group.add_argument("--all", action="store_true")

    parser.add_argument("--to", dest="end")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Create verified archive copies without removing originals.",
    )

    args = parser.parse_args()

    selected = select_stages(
        stage=args.stage,
        start=args.start,
        end=args.end,
        all_stages=args.all,
    )

    plan = build_archive_plan(selected)

    print("\nARCHIVE PLAN")
    print("=" * 65)
    print("Selected:", plan["selected_stages"])
    print("Invalidated:", plan["invalidated_stages"])
    print("Files:", len(plan["files"]))
    print("Total bytes:", plan["total_bytes"])

    if not args.apply:
        print("\nDRY RUN: No files copied or modified.")
        return

    create_archive(plan)


if __name__ == "__main__":
    main()
