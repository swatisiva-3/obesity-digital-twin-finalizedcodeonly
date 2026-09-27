"""
run_pipeline.py
================
Runs the whole pipeline, in order, stopping at the first step that fails.

    python run_pipeline.py                      # everything (01 -> 08)
    python run_pipeline.py --from 05            # resume from a step
    python run_pipeline.py --steps 01 02 03     # only these steps
    python run_pipeline.py --quick              # fast end-to-end check (few Optuna trials, few epochs,
                                                # small PhenoGraph sample) -- NOT for final results
    python run_pipeline.py --patient 2024_1234567   # also draw that patient's SHAP waterfall (Step 6)

Order (each step's outputs are the next step's inputs -- see README):
    01_data_loader -> 02_preprocessing -> 03_scoring -> 04_optuna_weight_learning
    -> 04_split_comparison -> 05_train_model -> 06_shap_explain
    -> 07_phenograph_clustering -> 08_baseline_models
(00_setup_environment.py is run once by hand, before this.)
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [
    ("01", "01_data_loader.py"),
    ("02", "02_preprocessing.py"),
    ("03", "03_scoring.py"),
    ("04", "04_optuna_weight_learning.py"),
    ("04b", "04_split_comparison.py"),
    ("05", "05_train_model.py"),
    ("06", "06_shap_explain.py"),
    ("07", "07_phenograph_clustering.py"),
    ("08", "08_baseline_models.py"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="+", help="step ids to run, e.g. 01 02 04b")
    ap.add_argument("--from", dest="start", help="run this step and everything after it")
    ap.add_argument("--quick", action="store_true", help="fast smoke test of every step")
    ap.add_argument("--patient", help="PATIENT_KEY for a single-patient SHAP explanation")
    args = ap.parse_args()

    ids = [s for s, _ in STEPS]
    selected = STEPS
    if args.steps:
        bad = set(args.steps) - set(ids)
        if bad:
            sys.exit(f"Unknown step(s) {bad}. Valid: {ids}")
        selected = [s for s in STEPS if s[0] in args.steps]
    elif args.start:
        if args.start not in ids:
            sys.exit(f"Unknown step {args.start}. Valid: {ids}")
        selected = STEPS[ids.index(args.start):]

    env = dict(os.environ)
    if args.quick:
        env["MBSAQIP_QUICK"] = "1"   # read by config.py
    extra = {
        "04": ["--weight-trials", "10", "--mlp-trials", "4"] if args.quick else [],
        "05": ["--max-epochs", "8"] if args.quick else [],
        "06": ["--patient", args.patient] if args.patient else [],
        "01": ["--no-excel"] if args.quick else [],
    }

    timings = []
    for sid, script in selected:
        cmd = [sys.executable, os.path.join(HERE, script)] + extra.get(sid, [])
        print(f"\n{'#' * 78}\n# STEP {sid}: {script}\n{'#' * 78}", flush=True)
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=HERE, env=env)
        timings.append((sid, script, time.time() - t0, rc))
        if rc != 0:
            print(f"\nSTEP {sid} ({script}) FAILED with exit code {rc} -- stopping. Fix it, then resume with:\n"
                  f"    python run_pipeline.py --from {sid}")
            break

    print(f"\n{'=' * 78}\nSUMMARY")
    for sid, script, secs, rc in timings:
        print(f"  {sid:4s} {script:34s} {secs/60:6.1f} min  {'OK' if rc == 0 else 'FAILED'}")
    print("Outputs: outputs/excel (spreadsheets), outputs/figures (plots), models/ (trained model, split, weights)")
    sys.exit(0 if all(rc == 0 for *_, rc in timings) else 1)


if __name__ == "__main__":
    main()
