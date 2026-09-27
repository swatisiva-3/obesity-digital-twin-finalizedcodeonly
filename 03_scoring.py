"""
03_scoring.py
==============
STEP 3 -- clinical-weight phenotype scoring (no outcome data needed).

INPUT:  outputs/02_preprocessed.parquet             (Step 2)
OUTPUT: outputs/03_scored.parquet                   -> Step 4 (split + Optuna)
        models/clinical_baseline_weights.json        -> Step 4 seeds Optuna from these
        outputs/excel/03_phenotype_scores.xlsx       (weights used, domain/total score summary
                                                      by year, per-patient scores for every patient
                                                      -- one sheet per year)
        outputs/figures/03_domain_scores_by_year.png

  1. Start from config.VARIABLE_RATIONALE's clinical weights (variables that
     don't exist in the data, e.g. MOBILITY_DEVICE/PRIORITY, have their weight
     redistributed within their domain).
  2. Flag-count rule: any variable flagged on more than
     config.FLAG_THRESHOLD_COUNT patients has its weight nudged by
     config.FLAG_WEIGHT_STEP in the direction set by its flag_direction_logic.
  3. Per patient: 0-100 score for each of the 4 domains + one 0-100 total
     phenotype score. Missing values are handled per patient (a missing lab
     drops out of THAT patient's weighted average instead of counting as 0).
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import scoring_engine
import utils

DOMAINS = list(config.VARIABLE_RATIONALE.keys())


def main():
    utils.log("=" * 70)
    utils.log("STEP 3: CLINICAL-WEIGHT PHENOTYPE SCORING")
    utils.log("=" * 70)
    df = utils.load_modeling_frame(config.PREPROCESSED_PATH)
    utils.log(f"Loaded {len(df):,} patients from Step 2")

    clinical = scoring_engine.default_variable_weights()
    effective = scoring_engine.apply_flag_count_weight_nudge(df, clinical)
    domain_weights = scoring_engine.default_domain_weights()

    domain_scores, total = scoring_engine.full_scoring_pipeline(df, effective, domain_weights)
    df = pd.concat([df.drop(columns=[c for c in domain_scores.columns if c in df.columns], errors="ignore"),
                    domain_scores], axis=1)
    df["TOTAL_PHENOTYPE_SCORE"] = total
    df.to_parquet(config.SCORED_PATH, index=False)
    utils.log(f"Saved {config.SCORED_PATH}")

    utils.save_json({"variable_weights": effective, "domain_weights": domain_weights,
                     "clinical_weights_before_flag_nudge": clinical},
                    os.path.join(config.MODEL_DIR, "clinical_baseline_weights.json"))

    # ---- Excel ----
    wrows = []
    for d, vw in effective.items():
        for v, w in vw.items():
            wrows.append({"domain": d, "variable": v, "clinical_weight_pct": round(clinical[d][v], 2),
                          "after_flag_nudge_pct": round(w, 2), "domain_weight_pct": domain_weights[d]})
    score_cols = [f"{d}_SCORE" for d in DOMAINS] + ["TOTAL_PHENOTYPE_SCORE"]
    summ = df.groupby("OPYEAR")[score_cols].agg(["mean", "std", "median"]).round(2)
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    overall = df[score_cols].agg(["mean", "std", "median"]).T.round(2).reset_index().rename(columns={"index": "score"})
    per_patient = df[["PATIENT_KEY", "CASEID", "OPYEAR", "HAS_VALID_OUTCOME", "OUTCOME_BMI_30D", "OUTCOME_WEIGHT_30D_KG"]
                     + [c for c in df.columns if c.endswith("__SEVERITY")] + score_cols]
    if not config.WRITE_FULL_PATIENT_EXCEL:
        per_patient = per_patient.groupby("OPYEAR", group_keys=False).head(config.EXCEL_PREVIEW_ROWS)
    utils.save_df_by_year_excel(per_patient, utils.excel_path("03_phenotype_scores.xlsx"), extra_sheets={
        "weights_used": pd.DataFrame(wrows), "scores_overall": overall, "scores_by_year": summ.reset_index()})

    for d in DOMAINS:
        utils.log(f"  {d:20s} mean={df[f'{d}_SCORE'].mean():6.2f}  std={df[f'{d}_SCORE'].std():6.2f}")
    utils.log(f"  {'TOTAL':20s} mean={total.mean():6.2f}  std={total.std():6.2f}")

    # ---- figure: domain + total score distribution by year ----
    years = sorted(df["OPYEAR"].unique())
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), sharey=False)
    for ax, col in zip(axes, score_cols):
        data = [df.loc[df["OPYEAR"] == y, col].dropna().values for y in years]
        ax.boxplot(data, tick_labels=years, showfliers=False)
        ax.set_title(col.replace("_SCORE", "").replace("_", " "))
        ax.set_ylabel("score (0-100)")
    fig.suptitle("Phenotype domain and total scores by year (clinical weights)")
    fig.tight_layout()
    utils.save_fig(fig, "03_domain_scores_by_year")
    utils.log("Step 3 complete.")


if __name__ == "__main__":
    main()
