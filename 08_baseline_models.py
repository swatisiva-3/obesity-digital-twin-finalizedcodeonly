"""
08_baseline_models.py
======================
STEP 8 -- baseline models, to show whether the MLP (Step 5) actually earns
its complexity. Same features, same permanent 70/15/15 split, same
train-only preprocessing as the MLP.

INPUT:  outputs/04_final_scored.parquet   (Step 4; falls back to 03_scored.parquet)
        models/train_val_test_indices.json (permanent split -- the SAME patients the MLP used)
        models/mlp_metrics.json            (Step 5 -- for the side-by-side comparison; optional)
OUTPUT: models/baseline_metrics.json
        outputs/excel/08_baseline_vs_mlp.xlsx
        outputs/figures/08_baseline_vs_mlp.png

Baselines, per target (30-day BMI and 30-day weight):
  1. Mean predictor       -- predicts the TRAINING-set mean for everyone.
  2. Carry-forward        -- predicts 30-day value = pre-op value (no model at all).
  3. Ridge regression     -- linear model on the same features; alpha chosen
                             from config.BASELINE_RIDGE_ALPHAS using the
                             VALIDATION set only. The TEST set is touched
                             exactly once, after alpha is fixed.
Missing values: training-set medians. Scaling: training-set mean/SD.
"""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import model_utils as mu
import utils


def calculate_metrics(y_true, y_pred):
    return mu.regression_metrics(y_true, y_pred)


def print_metrics(name, split, m):
    utils.log(f"  {name:32s} {split:10s} RMSE {m['RMSE']:8.4f}  MAE {m['MAE']:8.4f}  R^2 {m['R2']:8.4f}")


def main():
    from scipy.linalg import LinAlgWarning
    warnings.filterwarnings("ignore", category=LinAlgWarning)
    utils.log("=" * 70)
    utils.log("STEP 8: BASELINE MODELS vs MLP")
    utils.log("=" * 70)
    df, source = mu.load_scored_with_fallback()
    df["OPYEAR"] = df["OPYEAR"].astype(str)
    split = mu.get_or_create_split(df)
    parts = mu.split_frames(df, split)
    del df
    utils.log(f"Input: {source}; train {len(parts['train']):,} / validation {len(parts['validation']):,} / "
              f"test {len(parts['test']):,}")

    Xtr, _, _ = mu.build_feature_matrix(parts["train"])
    prep = mu.Preprocessor().fit(Xtr)
    X = {k: prep.transform_X(mu.build_feature_matrix(v)[0]).values for k, v in parts.items()}

    rows, results = [], {}
    for t in mu.TARGET_NAMES:
        spec = mu.TARGETS[t]
        y = {k: v[spec["outcome"]].values for k, v in parts.items()}
        results[t] = {}

        mean_val = float(np.mean(y["train"]))
        for sp in ("validation", "test"):
            m = calculate_metrics(y[sp], np.full(len(y[sp]), mean_val))
            print_metrics(f"{t} mean predictor", sp, m)
            rows.append({"target": f"{t}_30D", "model": "Mean predictor (training mean)", "split": sp, **m})
        results[t]["mean_predictor"] = {"train_mean": mean_val}

        for sp in ("validation", "test"):
            m = calculate_metrics(y[sp], parts[sp][spec["baseline"]].values)
            print_metrics(f"{t} carry-forward", sp, m)
            rows.append({"target": f"{t}_30D", "model": "Carry-forward (30-day = pre-op)", "split": sp, **m})

        alpha_rows = []
        for a in config.BASELINE_RIDGE_ALPHAS:
            r = Ridge(alpha=a).fit(X["train"], y["train"])
            m = calculate_metrics(y["validation"], r.predict(X["validation"]))
            alpha_rows.append({"target": f"{t}_30D", "alpha": a, **{f"validation_{k}": v for k, v in m.items()}})
        best_alpha = min(alpha_rows, key=lambda r: r["validation_RMSE"])["alpha"]
        ridge = Ridge(alpha=best_alpha).fit(X["train"], y["train"])
        for sp in ("validation", "test"):
            m = calculate_metrics(y[sp], ridge.predict(X[sp]))
            print_metrics(f"{t} Ridge (alpha={best_alpha})", sp, m)
            rows.append({"target": f"{t}_30D", "model": f"Ridge regression (alpha={best_alpha}, chosen on validation)",
                         "split": sp, **m})
        results[t]["ridge"] = {"alpha_grid": alpha_rows, "best_alpha": best_alpha}

    if os.path.exists(config.MLP_METRICS_PATH):
        mm = utils.load_json(config.MLP_METRICS_PATH)
        for r in mm["performance"]:
            if r["split"] in ("validation", "test") and r["target"] in ("BMI_30D", "WEIGHT_30D"):
                rows.append({"target": r["target"], "model": "MLP (Step 5)", "split": r["split"],
                             **{k: r[k] for k in ("RMSE", "MAE", "R2", "n")}})
    else:
        utils.log("  (Step 5 hasn't run -- MLP column will be missing from the comparison)")

    comp = pd.DataFrame(rows)
    utils.save_json({"comparison": rows, "details": results, "input": source}, os.path.join(config.MODEL_DIR, "baseline_metrics.json"))
    wide = comp.pivot_table(index=["target", "model"], columns="split", values=["RMSE", "MAE", "R2"]).round(4)
    wide.columns = [f"{m}_{s}" for m, s in wide.columns]
    alpha = pd.DataFrame([r for t in results for r in results[t]["ridge"]["alpha_grid"]])
    utils.save_excel_sheets(utils.excel_path("08_baseline_vs_mlp.xlsx"), {
        "comparison": wide.reset_index(), "comparison_long": comp, "ridge_alpha_selection": alpha,
        "notes": pd.DataFrame({"note": [
            "All models use the permanent split in models/train_val_test_indices.json.",
            "Imputation (medians) and scaling (mean/SD) are fit on TRAINING rows only.",
            "Ridge alpha is chosen on the VALIDATION set; the TEST set is evaluated once, afterwards.",
            "Carry-forward is the natural floor for 30-day BMI/weight -- a useful model must beat it.",
        ]})})

    test = comp[comp["split"] == "test"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, t in zip(axes, ["BMI_30D", "WEIGHT_30D"]):
        s = test[test["target"] == t].sort_values("RMSE", ascending=False)
        ax.barh(s["model"], s["RMSE"], color=["firebrick" if m.startswith("MLP") else "grey" for m in s["model"]])
        for i, (rm, r2) in enumerate(zip(s["RMSE"], s["R2"])):
            ax.text(rm, i, f"  RMSE {rm:.3f} | R² {r2:.3f}", va="center", fontsize=8)
        ax.set_xlabel(f"test RMSE ({'kg/m^2' if t.startswith('BMI') else 'kg'})")
        ax.set_title(f"30-day {'BMI' if t.startswith('BMI') else 'weight'}: baselines vs MLP (test set)")
        ax.set_xlim(0, s["RMSE"].max() * 1.45)
    fig.tight_layout()
    utils.save_fig(fig, "08_baseline_vs_mlp")
    utils.log("Step 8 complete.")


if __name__ == "__main__":
    main()
