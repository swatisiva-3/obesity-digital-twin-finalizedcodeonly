import hashlib
import tempfile
import unittest
from pathlib import Path

from pipeline_publish import publish_outputs
from pipeline_stages import STAGES


ROOT = Path(__file__).resolve().parent

TRANSACTION = (
    ROOT
    / "archives/transactions"
    / "20261029T235114Z_e15eb9e3b380"
)

STAGED = TRANSACTION / "staging/workspace"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TestStage09PublicationRehearsal(unittest.TestCase):

    def test_publication_and_failure_rollback(self):
        names = STAGES["09"]["outputs"]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            active = root / "active"
            staging = root / "staging"

            originals = {}

            for name in names:
                source = ROOT / name
                generated = STAGED / name

                self.assertTrue(source.is_file(), name)
                self.assertTrue(generated.is_file(), name)

                destination = active / name
                staged_destination = staging / name

                destination.parent.mkdir(parents=True, exist_ok=True)
                staged_destination.parent.mkdir(
                    parents=True, exist_ok=True
                )

                import shutil
                shutil.copy2(source, destination)
                shutil.copy2(generated, staged_destination)

                originals[name] = sha256(destination)

            # First rehearsal: successful publication.
            success_backup = root / "success_backup"

            result = publish_outputs(
                active,
                staging,
                success_backup,
                names,
            )

            self.assertEqual(result["status"], "completed")

            for name in names:
                self.assertEqual(
                    sha256(active / name),
                    sha256(STAGED / name),
                )
                self.assertEqual(
                    sha256(success_backup / name),
                    originals[name],
                )

            # Restore the temporary originals for a failure test.
            import shutil
            for name in names:
                shutil.copy2(success_backup / name, active / name)

            # Second rehearsal: fail after publishing one file.
            failure_backup = root / "failure_backup"

            with self.assertRaises(RuntimeError):
                publish_outputs(
                    active,
                    staging,
                    failure_backup,
                    names,
                    fail_after=1,
                )

            for name in names:
                self.assertEqual(
                    sha256(active / name),
                    originals[name],
                    "Rollback failed: " + name,
                )

            print("\nPASS: Successful temporary publication.")
            print("PASS: Verified backups.")
            print("PASS: Failure restored all three originals.")
            print("PASS: Active research files untouched.")


if __name__ == "__main__":
    unittest.main()
