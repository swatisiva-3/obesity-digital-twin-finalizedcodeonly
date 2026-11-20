"""Safe pipeline execution controller.

Currently supports planning only.
No execution or file removal is enabled.
"""

import argparse

from pipeline_stages import select_stages
from pipeline_execution_plan import validate_plan
from pipeline_preflight import check_stages
from pipeline_archive_plan import build_archive_plan
from pipeline_publication_plan import build_publication_plan


def prepare_execution(selected):
    print("\n" + "=" * 65)
    print("PIPELINE EXECUTION CONTROLLER")
    print("=" * 65)

    # 1. Validate protected files and dependencies.
    if not validate_plan(selected):
        raise RuntimeError("Unsafe execution plan")

    # 2. Validate required inputs and dataset structure.
    if not check_stages(selected):
        raise RuntimeError("Preflight validation failed")

    # 3. Generate a fresh checksum inventory.
    plan = build_archive_plan(selected)

    print("\nARCHIVE REQUIREMENTS")
    print("Selected stages:", plan["selected_stages"])
    print("Downstream invalidated:", plan["invalidated_stages"])
    print("Files to archive:", len(plan["files"]))
    print("Total archive bytes:", plan["total_bytes"])

    print("\nEXECUTION DISABLED")
    print("All checks completed without modifying research files.")

    return plan


def main():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stage")
    group.add_argument("--from", dest="start")
    group.add_argument("--all", action="store_true")

    parser.add_argument("--to", dest="end")
    parser.add_argument(
        "--transaction",
        help="Inspect an existing staged transaction without publishing.",
    )

    args = parser.parse_args()

    selected = select_stages(
        stage=args.stage,
        start=args.start,
        end=args.end,
        all_stages=args.all,
    )

    prepare_execution(selected)

    if args.transaction:
        if selected != ["09"]:
            raise RuntimeError(
                "Existing staged transactions currently support Stage 09 only."
            )

        publication = build_publication_plan(
            args.transaction,
            "09",
        )

        print("\nPUBLICATION DECISION")
        print("=" * 65)
        print("Transaction:", publication["transaction_id"])
        print("Status:", publication["status"])

        if publication["status"] == "no_changes":
            print("All staged outputs match active outputs.")
            print("No replacement or downstream invalidation required.")
        else:
            print("Changed outputs detected.")
            print("Publication remains disabled.")
            print(
                "Downstream invalidated:",
                publication["invalidated_stages"],
            )


if __name__ == "__main__":
    main()
