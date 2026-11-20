import tempfile
import unittest
from pathlib import Path

from pipeline_publish import (
    checksum,
    publish_outputs,
)


class TestPipelinePublication(unittest.TestCase):

    def test_successful_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            active = root / "active"
            staging = root / "staging"
            backup = root / "backup"

            active.mkdir()
            staging.mkdir()

            (active / "result.txt").write_text("original")
            (staging / "result.txt").write_text("replacement")

            result = publish_outputs(
                active, staging, backup, ["result.txt"]
            )

            self.assertEqual(result["status"], "completed")
            self.assertEqual(
                (active / "result.txt").read_text(),
                "replacement"
            )
            self.assertEqual(
                (backup / "result.txt").read_text(),
                "original"
            )

    def test_failure_restores_all_originals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            active = root / "active"
            staging = root / "staging"
            backup = root / "backup"

            active.mkdir()
            staging.mkdir()

            names = ["bmi.csv", "weight.csv", "metrics.json"]

            originals = {}
            for name in names:
                (active / name).write_text("original:" + name)
                (staging / name).write_text("new:" + name)
                originals[name] = checksum(active / name)

            with self.assertRaises(RuntimeError):
                publish_outputs(
                    active,
                    staging,
                    backup,
                    names,
                    fail_after=1,
                )

            for name in names:
                self.assertEqual(
                    checksum(active / name),
                    originals[name],
                )

                self.assertEqual(
                    checksum(backup / name),
                    originals[name],
                )


    def test_recovery_after_interrupted_publication(self):
        import json

        from pipeline_publish import (
            inspect_publication,
            recover_publication,
        )
        from pipeline_transaction import transaction_lock

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            active = root / "active"
            backup = root / "backup"

            active.mkdir()
            backup.mkdir()

            originals = {}

            for name in ["bmi.csv", "weight.csv"]:
                (active / name).write_text("original:" + name)
                (backup / name).write_text("original:" + name)
                originals[name] = checksum(active / name)

            manifest = {
                "status": "publishing",
                "originals": originals,
                "published": ["bmi.csv"],
            }

            (backup / "publication_manifest.json").write_text(
                json.dumps(manifest)
            )

            # Simulate an interrupted publication.
            (active / "bmi.csv").write_text("new BMI")

            with transaction_lock(root / "transactions"):
                report = inspect_publication(active, backup)

                self.assertTrue(report["recoverable"])

                recover_publication(active, backup)

            for name in originals:
                self.assertEqual(
                    checksum(active / name),
                    originals[name],
                )


if __name__ == "__main__":
    unittest.main()
