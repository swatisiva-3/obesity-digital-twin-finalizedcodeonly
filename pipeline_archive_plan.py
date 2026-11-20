"""Read-only archive planning for selected pipeline stages."""

import hashlib
import json
from pathlib import Path

from pipeline_stages import STAGES, downstream_stages, select_stages

ROOT = Path(__file__).resolve().parent


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def build_archive_plan(selected):
    downstream = downstream_stages(selected)
    affected = list(dict.fromkeys(selected + downstream))

    files = []
    seen = set()

    for stage_id in affected:
        for relative in STAGES[stage_id]["outputs"]:
            if relative in seen:
                continue

            seen.add(relative)
            path = ROOT / relative

            if not path.is_file():
                continue

            files.append({
                "stage": stage_id,
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })

    return {
        "selected_stages": selected,
        "invalidated_stages": downstream,
        "affected_stages": affected,
        "files": files,
        "total_bytes": sum(item["size_bytes"] for item in files),
        "dry_run": True,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument("--stage")
    group.add_argument("--from", dest="start")
    group.add_argument("--all", action="store_true")

    parser.add_argument("--to", dest="end")

    args = parser.parse_args()

    selected = select_stages(
        stage=args.stage,
        start=args.start,
        end=args.end,
        all_stages=args.all,
    )

    plan = build_archive_plan(selected)

    print(json.dumps(plan, indent=2))
