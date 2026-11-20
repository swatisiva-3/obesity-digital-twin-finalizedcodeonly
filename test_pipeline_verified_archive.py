import tempfile
import unittest
from pathlib import Path

from pipeline_archive_plan import sha256_file
from pipeline_verified_archive import create_verified_archive


class TestVerifiedArchive(unittest.TestCase):

    def test_archive_and_checksum_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            active = root / "active"
            destination = root / "backup"

            paths = [
                "models/xgboost_metrics_v2.json",
                "outputs/09_xgboost_test_predictions_BMI_v2.csv",
                "outputs/10_paired_model_comparison_v2.json",
            ]

            files = []

            for relative in paths:
                source = active / relative
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text("research artifact: " + relative)

                files.append({
                    "path": relative,
                    "sha256": sha256_file(source),
                    "size_bytes": source.stat().st_size,
                })

            plan = {
                "dry_run": True,
                "files": files,
            }

            result = create_verified_archive(
                plan,
                active,
                destination,
            )

            self.assertEqual(result["status"], "verified")
            self.assertEqual(result["file_count"], 3)

            for item in files:
                self.assertEqual(
                    sha256_file(active / item["path"]),
                    sha256_file(destination / item["path"]),
                )

            with self.assertRaises(RuntimeError):
                create_verified_archive(
                    plan,
                    active,
                    destination,
                )

            print("\nPASS: Archive created.")
            print("PASS: All checksums verified.")
            print("PASS: Existing archive cannot be overwritten.")
            print("PASS: Original files preserved.")


if __name__ == "__main__":
    unittest.main()
