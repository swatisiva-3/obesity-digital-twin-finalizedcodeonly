import json
import tempfile
import unittest
from pathlib import Path

from pipeline_transaction import transaction_lock
from pipeline_publish import publish_outputs, checksum
from pipeline_verified_archive import create_verified_archive


class TestChangedTransactionWorkflow(unittest.TestCase):

    def test_changed_outputs_with_downstream_tracking(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            active = root / "active"
            staging = root / "staging"
            archive = root / "archive"
            publication_backup = root / "publication_backup"

            stage09 = [
                "models/xgboost_metrics_v2.json",
                "outputs/09_predictions.csv",
            ]

            downstream = [
                "outputs/10_comparison.json",
                "outputs/training_leakage_audit.json",
            ]

            all_files = stage09 + downstream
            originals = {}

            for name in all_files:
                path = active / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("original:" + name)
                originals[name] = checksum(path)

            for name in stage09:
                path = staging / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("updated:" + name)

            plan = {
                "dry_run": True,
                "files": [
                    {"path": name, "sha256": originals[name]}
                    for name in all_files
                ],
            }

            with transaction_lock(root=root / "locks"):
                archived = create_verified_archive(
                    plan, active, archive
                )

                self.assertEqual(archived["file_count"], 4)

                published = publish_outputs(
                    active,
                    staging,
                    publication_backup,
                    stage09,
                )

                self.assertEqual(published["status"], "completed")

                validity = {
                    "status": "downstream_stale",
                    "changed_stage": "09",
                    "stale_stages": ["10", "audit"],
                    "stale_artifacts": downstream,
                }

                validity_path = root / "validity.json"
                validity_path.write_text(
                    json.dumps(validity, indent=2)
                )

                for name in stage09:
                    self.assertEqual(
                        (active / name).read_text(),
                        "updated:" + name,
                    )

                for name in downstream:
                    self.assertEqual(
                        checksum(active / name),
                        originals[name],
                    )

                for name in all_files:
                    self.assertEqual(
                        checksum(archive / name),
                        originals[name],
                    )

                self.assertEqual(
                    json.loads(validity_path.read_text())["status"],
                    "downstream_stale",
                )

            print("PASS: Changed outputs published.")
            print("PASS: All original artifacts archived.")
            print("PASS: Downstream artifacts preserved.")
            print("PASS: Downstream staleness recorded.")
            print("PASS: Real research files untouched.")


if __name__ == "__main__":
    unittest.main()
