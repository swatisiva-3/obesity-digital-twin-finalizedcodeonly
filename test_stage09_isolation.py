import tempfile
import unittest
from pathlib import Path


class TestStage09Isolation(unittest.TestCase):

    def test_workspace_paths_are_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            active = root / "project"
            workspace = root / "transactions" / "staging" / "workspace"

            active_outputs = active / "outputs"
            active_models = active / "models"
            staged_outputs = workspace / "outputs"
            staged_models = workspace / "models"

            for directory in (
                active_outputs,
                active_models,
                staged_outputs,
                staged_models,
            ):
                directory.mkdir(parents=True)

            active_file = active_outputs / "result.csv"
            active_file.write_text("original research result")

            staged_file = staged_outputs / "result.csv"
            staged_file.write_text("new model result")

            self.assertNotEqual(
                active_outputs.resolve(),
                staged_outputs.resolve()
            )
            self.assertNotEqual(
                active_models.resolve(),
                staged_models.resolve()
            )
            self.assertEqual(
                active_file.read_text(),
                "original research result"
            )
            self.assertEqual(
                staged_file.read_text(),
                "new model result"
            )


if __name__ == "__main__":
    unittest.main()
