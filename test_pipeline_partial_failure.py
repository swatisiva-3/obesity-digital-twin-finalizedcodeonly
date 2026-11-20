"""Test isolated staging and partial-failure recovery."""

import tempfile
from pathlib import Path


def simulate_stage(root, fail_after_first=False):
    active = root / "active"
    staging = root / "staging"

    active.mkdir(parents=True)
    staging.mkdir(parents=True)

    filenames = [
        "bmi_predictions.csv",
        "weight_predictions.csv",
        "metrics.json",
    ]

    for name in filenames:
        (active / name).write_text("original:" + name)

    try:
        (staging / filenames[0]).write_text("new BMI")

        if fail_after_first:
            raise RuntimeError("Simulated failure after BMI output")

        (staging / filenames[1]).write_text("new weight")
        (staging / filenames[2]).write_text("new metrics")

        # Verify every output before publishing.
        for name in filenames:
            assert (staging / name).is_file()
            assert (staging / name).stat().st_size > 0

        # This publication loop is not atomic as a group.
        # It will need a recovery mechanism in the real runner.
        for name in filenames:
            (staging / name).replace(active / name)

        return "completed", active, filenames

    except RuntimeError:
        return "failed", active, filenames


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        status, active, names = simulate_stage(
            root / "failure",
            fail_after_first=True,
        )

        assert status == "failed"

        for name in names:
            assert (active / name).read_text() == "original:" + name

        print("PARTIAL FAILURE TEST: PASS")
        print("All three original outputs preserved.")

        status, active, names = simulate_stage(root / "success")

        assert status == "completed"
        assert (active / names[0]).read_text() == "new BMI"
        assert (active / names[1]).read_text() == "new weight"
        assert (active / names[2]).read_text() == "new metrics"

        print("SUCCESSFUL STAGING TEST: PASS")
        print("All three outputs published.")

        print("\nNo research artifacts modified.")


if __name__ == "__main__":
    main()
