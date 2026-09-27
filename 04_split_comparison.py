"""
04_split_comparison.py
=======================
STEP 4b -- the permanent train/validation/test split, checked and compared.

INPUT:  outputs/04_final_scored.parquet   (Step 4; falls back to 03_scored.parquet)
        models/train_val_test_indices.json (created here if Step 4 hasn't made it yet)
OUTPUT: models/train_val_test_indices.json -> used by Steps 4, 5, 6, 8 (never re-drawn unless deleted)
        outputs/excel/04_split_comparison.xlsx
        outputs/figures/04_split_outcome_distributions.png
        outputs/figures/04_split_strategy_comparison.png

Three questions, three sheets:
  1. Is the split the right size and year-balanced?  (70/15/15 inside EVERY year)
  2. Do train, validation and test look like the same population?
     Mean/SD of outcomes, baseline BMI/weight, age, follow-up day and the
     phenotype scores per split, with standardized mean differences (SMD;
     |SMD| < 0.1 is the usual "well balanced" rule of thumb).
  3. How much does the choice of split change the answer? A fast linear
     probe (Ridge, same features and train-only preprocessing as the MLP) is
     evaluated under:
       - the permanent RANDOM split (train -> validation, train -> test)
       - a TEMPORAL split: train on the earlier years, test on
         config.TEMPORAL_HOLDOUT_YEAR (a year the model has never seen)
       - LEAVE-ONE-YEAR-OUT for each year
     If the temporal / leave-one-year-out errors are close to the random-
     split errors, a model trained on pooled years transfers across years.
"""
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

BALANCE_VARS = ["OUTCOME_BMI_30D", "BMI_CHANGE_30D", "BMI_PCT_CHANGE_30D", "OUTCOME_WEIGHT_30D_KG",
                "WEIGHT_CHANGE_30D_KG", "BASELINE_BMI", "BASELINE_WEIGHT_KG", "AGE__STD", "FOLLOWUP_DAY",
                "Adiposity_SCORE", "Metabolic_SCORE", "Behavioral_SCORE", "Socio-Environmental_SCORE",
                "TOTAL_PHENOTYPE_SCORE"]


def smd(a, b):
    a, b = a.dropna(), b.dropna()
    pooled = np.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def ridge_probe(train, evals):
    """Fit Ridge on `train` (train-only preprocessing), return metrics on each eval frame."""
    from scipy.linalg import LinAlgWarning
    warnings.filterwarnings("ignore", category=LinAlgWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    Xtr, _, _ = mu.build_feature_matrix(train)
    prep = mu.Preprocessor().fit(Xtr, mu.change_targets(train))
    model = Ridge(alpha=1.0).fit(prep.transform_X(Xtr).values, prep.transform_Y(mu.change_targets(train)).values)
    rows = []
    for name, ev in evals.items():
        Xev, _, _ = mu.build_feature_matrix(ev)
        change = prep.inverse_Y(model.predict(prep.transform_X(Xev).values))
        row = {"evaluated_on": name, "n_train": len(train), "n_eval": len(ev)}
        for j, t in enumerate(mu.TARGET_NAMES):
            spec = mu.TARGETS[t]
            m_abs = mu.regression_metrics(ev[spec["outcome"]], ev[spec["baseline"]].values + change[:, j])
            m_chg = mu.regression_metrics(ev[spec["change"]], change[:, j])
            row.update({f"{t}_30D_RMSE": m_abs["RMSE"], f"{t}_30D_MAE": m_abs["MAE"], f"{t}_30D_R2": m_abs["R2"],
                        f"{t}_CHANGE_R2": m_chg["R2"]})
        rows.append(row)
    return rows


def main():
    utils.log("=" * 70)
    utils.log("STEP 4b: SPLIT CHECK + SPLIT-STRATEGY COMPARISON")
    utils.log("=" * 70)
    import os
    path = config.FINAL_SCORED_PATH if os.path.exists(config.FINAL_SCORED_PATH) else config.SCORED_PATH
    df = utils.read_model_columns(path)
    split = mu.get_or_create_split(df)
    parts = mu.split_frames(df, split)
    eligible = df[df["HAS_VALID_OUTCOME"]]

    # 1. sizes by year
    size_rows = []
    for yr in sorted(eligible["OPYEAR"].unique()):
        n_year = int((eligible["OPYEAR"] == yr).sum())
        row = {"OPYEAR": yr, "eligible_patients": n_year}
        for k, p in parts.items():
            n = int((p["OPYEAR"] == yr).sum())
            row[f"{k}_n"], row[f"{k}_pct_of_year"] = n, round(100 * n / n_year, 2)
        size_rows.append(row)
    total = {"OPYEAR": "ALL", "eligible_patients": len(eligible)}
    for k, p in parts.items():
        total[f"{k}_n"], total[f"{k}_pct_of_year"] = len(p), round(100 * len(p) / len(eligible), 2)
    sizes = pd.DataFrame(size_rows + [total])
    utils.log("Split sizes:\n" + sizes.to_string(index=False))

    # 2. balance
    bal = []
    for v in BALANCE_VARS:
        if v not in df.columns:
            continue
        row = {"variable": v}
        for k, p in parts.items():
            row[f"{k}_mean"], row[f"{k}_sd"] = round(p[v].mean(), 3), round(p[v].std(), 3)
        row["SMD_train_vs_validation"] = round(smd(parts["train"][v], parts["validation"][v]), 4)
        row["SMD_train_vs_test"] = round(smd(parts["train"][v], parts["test"][v]), 4)
        bal.append(row)
    balance = pd.DataFrame(bal)
    worst = balance[["SMD_train_vs_validation", "SMD_train_vs_test"]].abs().max().max()
    utils.log(f"Largest |SMD| across balance variables: {worst:.4f} ({'well balanced' if worst < 0.1 else 'CHECK'})")

    # 3. split strategies
    utils.log("Split-strategy comparison (Ridge probe) ...")
    rows = []
    for r in ridge_probe(parts["train"], {"validation (random split)": parts["validation"],
                                          "test (random split)": parts["test"]}):
        rows.append({"strategy": "Permanent random 70/15/15 (stratified by year)", **r})
    hold = config.TEMPORAL_HOLDOUT_YEAR
    older = eligible[eligible["OPYEAR"] != hold]
    for r in ridge_probe(older, {f"all {hold} patients": eligible[eligible["OPYEAR"] == hold]}):
        rows.append({"strategy": f"Temporal: train {'+'.join(sorted(older['OPYEAR'].unique()))} -> test {hold}", **r})
    for yr in sorted(eligible["OPYEAR"].unique()):
        if yr == hold:
            continue  # identical to the temporal row above
        for r in ridge_probe(eligible[eligible["OPYEAR"] != yr], {f"all {yr} patients": eligible[eligible["OPYEAR"] == yr]}):
            rows.append({"strategy": f"Leave-one-year-out ({yr} held out)", **r})
    strategies = pd.DataFrame(rows)
    utils.log("\n" + strategies[["strategy", "evaluated_on", "BMI_30D_RMSE", "BMI_30D_R2", "BMI_CHANGE_R2",
                                 "WEIGHT_30D_RMSE"]].round(4).to_string(index=False))

    utils.save_excel_sheets(utils.excel_path("04_split_comparison.xlsx"), {
        "split_sizes_by_year": sizes, "balance_SMD": balance, "strategy_comparison": strategies,
        "notes": pd.DataFrame({"note": [
            f"Split file: {config.SPLIT_INDICES_PATH} (keys = PATIENT_KEY). Seed {config.RANDOM_SEED}.",
            "Only patients with a valid 30-day BMI AND weight outcome (after Step 2 cleaning) are split.",
            "SMD = standardized mean difference; |SMD| < 0.1 is conventionally 'balanced'.",
            "Strategy comparison uses a Ridge regression as a fast, deterministic probe -- the MLP (Step 5) "
            "is trained only on the permanent random split.",
            "30D metrics = predicted 30-day BMI/weight vs actual. CHANGE_R2 = R^2 on the change from pre-op, "
            "which is the harder, more informative number."]})})

    # figures
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for k, p in parts.items():
        axes[0].hist(p["BMI_PCT_CHANGE_30D"].dropna(), bins=80, range=(-20, 5), histtype="step", density=True, lw=1.6, label=f"{k} (n={len(p):,})")
        axes[1].hist(p["WEIGHT_CHANGE_30D_KG"].dropna(), bins=80, range=(-30, 5), histtype="step", density=True, lw=1.6, label=k)
    axes[0].set_xlabel("30-day BMI change (% of pre-op)")
    axes[1].set_xlabel("30-day weight change (kg)")
    for ax in axes:
        ax.set_ylabel("density")
        ax.legend()
    fig.suptitle("Outcome distributions are the same in train / validation / test")
    utils.save_fig(fig, "04_split_outcome_distributions")

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [f"{s}\n[{e}]" for s, e in zip(strategies["strategy"], strategies["evaluated_on"])]
    ax.barh(labels, strategies["BMI_30D_RMSE"], color="teal")
    ax.invert_yaxis()
    ax.set_xlabel("RMSE, predicted vs actual 30-day BMI (kg/m^2) -- Ridge probe")
    ax.set_title("Does the model transfer across years? Random vs temporal vs leave-one-year-out")
    utils.save_fig(fig, "04_split_strategy_comparison")
    utils.log("Step 4b complete.")


if __name__ == "__main__":
    main()
