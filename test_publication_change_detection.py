import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pipeline_publication_plan as planner
import pipeline_archive_plan as archive


class TestPublicationChangeDetection(unittest.TestCase):

    def test_changed_stage09_invalidates_downstream(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transaction = (
                root / "archives" / "transactions" / "test_transaction"
            )
            workspace = transaction / "staging" / "workspace"
            workspace.mkdir(parents=True)

            (transaction / "manifest.json").write_text(
                json.dumps({
                    "transaction_id": "test_transaction",
                    "stage": "09",
                    "status": "staged",
                    "outputs_published": False,
                })
            )

            from pipeline_stages import STAGES

            for relative in STAGES["09"]["outputs"]:
                active = root / relative
                staged = workspace / relative

                active.parent.mkdir(parents=True, exist_ok=True)
                staged.parent.mkdir(parents=True, exist_ok=True)

                active.write_text("original result")
                staged.write_text("new result")

            for stage in ("10", "audit"):
                for relative in STAGES[stage]["outputs"]:
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("existing downstream result")

            with patch.object(planner, "ROOT", root), \
                 patch.object(archive, "ROOT", root):

                result = planner.build_publication_plan(
                    transaction, "09"
                )

            self.assertEqual(
                result["status"], "publication_planned"
            )
            self.assertEqual(
                len(result["replacements"]), 3
            )
            self.assertEqual(
                result["invalidated_stages"],
                ["10", "audit"]
            )
            self.assertEqual(
                len(result["invalidated_artifacts"]), 5
            )
            self.assertFalse(result["publication_enabled"])

            print("\nPASS: Changed outputs detected.")
            print("PASS: Stage 10 and audit invalidated.")
            print("PASS: Publication remains disabled.")


if __name__ == "__main__":
    unittest.main()
