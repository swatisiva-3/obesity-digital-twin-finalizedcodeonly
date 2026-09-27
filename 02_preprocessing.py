"""
02_preprocessing.py
====================
STEP 2 -- unit standardization, outcome cleaning, abnormal-value flagging,
and 0-100 severity scoring for every rationale variable, on the pooled
3-year dataset.

INPUT:  outputs/01_predictors_all_variables.parquet   (Step 1)
        outputs/01_outcomes_bmi_weight_30day.parquet  (Step 1)  -- joined on PATIENT_KEY
OUTPUT: outputs/02_preprocessed.parquet               -> Step 3
        outputs/excel/02_flag_summary.xlsx             (flag rates per variable, overall + by year;
                                                        outcome-cleaning log)
        outputs/excel/02_preprocessed_PREVIEW.xlsx      (first rows per year with every __STD /
                                                        __FLAG / __FLAG_REASON / __SEVERITY column)
        outputs/figures/02_flag_rates_by_year.png

What happens here, in order:
  1. UNITS -- every height -> cm, every weight -> kg (both "in"/"cm" and
     "lbs"/"kg" appear in every year).
  2. OUTCOMES -- 30-day BMI / weight standardized to kg, plus derived
     BMI_CHANGE_30D, BMI_PCT_CHANGE_30D, WEIGHT_CHANGE_30D_KG and FOLLOWUP_DAY.
     Physiologically implausible values (see OUTCOME_SANITY below) are set
     to missing and logged, not silently kept.
  3. FLAGGING + SEVERITY -- each variable is compared against its clinical
     ideal/tolerance from config.VARIABLE_RATIONALE. Outside tolerance ->
     flagged with a reason ("High than normal by X% apart from ideal").
     Every variable also gets a 0-100 __SEVERITY score used by Step 3.
     Weight variables use each patient's own Devine ideal body weight.

NOTE ON SCALING: this step does NOT z-score anything for the model. Feature
scaling for the MLP / Ridge / clustering is fit on the TRAINING split only
(model_utils.py), so no information from validation/test patients leaks
into training.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import utils

# Anything outside these ranges is treated as a data-entry error -> missing.
OUTCOME_SANITY = {
    "bmi_min": 15.0, "bmi_max": 120.0,
    "weight_kg_min": 30.0, "weight_kg_max": 350.0,
    "max_abs_pct_bmi_change": 30.0,   # >30% BMI change within 30 days is not physiologic
}


def flag_numeric(values, ideal, tolerance, direction):
    diff = values - ideal
    if direction == "high_is_bad":
        is_flag, bad_side = diff > tolerance, diff.clip(lower=0)
    elif direction == "low_is_bad":
        is_flag, bad_side = diff < -tolerance, (-diff).clip(lower=0)
    else:
        is_flag, bad_side = diff.abs() > tolerance, diff.abs()

    with np.errstate(divide="ignore", invalid="ignore"):
        pct = pd.Series(np.where(ideal != 0, diff.abs() / ideal.abs() * 100, np.nan), index=values.index)
    side = np.where(diff > 0, "High than normal", "Low than normal")
    amount = np.where(pct.notna(), pct.round(1).astype(str) + "% apart from ideal",
                      diff.abs().round(2).astype(str) + " units apart from ideal (ideal = 0)")
    reason = np.where(is_flag, pd.Series(side, index=values.index) + " by " + pd.Series(amount, index=values.index),
                      "Within normal range")
    assessable = values.notna() & ideal.notna()
    reason = pd.Series(reason, index=values.index).where(assessable, "Not assessable (missing data)")
    return is_flag.where(assessable, False).astype(bool), reason, pct, bad_side


def process_numeric(df, var_name, spec):
    col = spec["column"]
    if spec.get("unit_kind") == "height" and spec.get("unit_column"):
        std = utils.height_to_cm_series(df[col], df[spec["unit_column"]])
    elif spec.get("unit_kind") == "weight" and spec.get("unit_column"):
        std = utils.weight_to_kg_series(df[col], df[spec["unit_column"]])
    else:
        std = pd.to_numeric(df[col], errors="coerce")
    df[f"{var_name}__STD"] = std

    severity = pd.Series(np.nan, index=df.index)
    flag_eligible = var_name not in config.FLAG_INELIGIBLE_VARIABLES and spec.get("tolerance") is not None
    if flag_eligible:
        ideal = df["IDEAL_BODY_WEIGHT_KG"] if spec.get("ideal") == "devine_ibw" else pd.Series(float(spec["ideal"]), index=df.index)
        is_flag, reason, pct, bad_side = flag_numeric(std, ideal, spec["tolerance"], spec["direction"])
        df[f"{var_name}__FLAG"], df[f"{var_name}__FLAG_REASON"], df[f"{var_name}__PCT_DEVIATION"] = is_flag, reason, pct
        if spec.get("contributes_to_severity_score") and spec.get("severity_span"):
            severity = (bad_side / spec["severity_span"] * 100).clip(0, 100).where(std.notna() & ideal.notna())
    else:
        df[f"{var_name}__FLAG"] = False
        df[f"{var_name}__FLAG_REASON"] = "Not flagged (no clinical abnormal threshold for this variable)"
        df[f"{var_name}__PCT_DEVIATION"] = np.nan
        if spec.get("contributes_to_severity_score") and spec.get("ideal") is not None and spec.get("severity_span"):
            diff = std - float(spec["ideal"])
            bad_side = diff.clip(lower=0) if spec["direction"] == "high_is_bad" else diff.abs()
            severity = (bad_side / spec["severity_span"] * 100).clip(0, 100)
    df[f"{var_name}__SEVERITY"] = severity
    return df


def process_categorical(df, var_name, spec):
    raw = df[spec["column"]].astype("string")
    scores = spec.get("category_scores")
    severity = raw.map(scores).astype(float) if scores else pd.Series(np.nan, index=df.index)
    flag_eligible = var_name not in config.FLAG_INELIGIBLE_VARIABLES and scores is not None
    if flag_eligible:
        assessable = raw.notna() & severity.notna()
        is_flag = (severity.fillna(0) > 0) & assessable
        reason = pd.Series(np.where(is_flag, "Abnormal / risk-positive category: " + raw.fillna(""),
                                    "Within normal category"), index=df.index)
        reason = reason.where(assessable, "Not assessable (missing or unmapped category)")
        unmapped = raw.notna() & ~raw.isin(list(scores.keys()))
        if unmapped.any():
            utils.log(f"  NOTE {var_name}: {int(unmapped.sum()):,} rows have a category not in config "
                      f"category_scores: {sorted(raw[unmapped].unique().tolist())[:5]}")
    else:
        is_flag = pd.Series(False, index=df.index)
        reason = pd.Series("Not flagged (context variable, no clinical abnormal category)", index=df.index)
        if not spec.get("contributes_to_severity_score", True):
            severity = pd.Series(np.nan, index=df.index)
    df[f"{var_name}__STD"] = raw
    df[f"{var_name}__FLAG"] = is_flag.astype(bool)
    df[f"{var_name}__FLAG_REASON"] = reason
    df[f"{var_name}__PCT_DEVIATION"] = np.nan
    df[f"{var_name}__SEVERITY"] = severity
    return df


def build_outcomes(df):
    """Standardize the 30-day outcomes and derive change variables. Returns a cleaning-log DataFrame."""
    log_rows = []
    df["BASELINE_BMI"] = pd.to_numeric(df["BMI_PREOP"], errors="coerce")
    df["BASELINE_WEIGHT_KG"] = utils.weight_to_kg_series(df["WGT_PREOP"], df["WGTUNIT_PREOP"])
    df["OUTCOME_BMI_30D"] = pd.to_numeric(df[config.OUTCOME_VARS["bmi_30d"]["column"]], errors="coerce")
    wspec = config.OUTCOME_VARS["weight_30d"]
    df["OUTCOME_WEIGHT_30D_KG"] = utils.weight_to_kg_series(df[wspec["column"]], df[wspec["unit_column"]])
    df["FOLLOWUP_DAY"] = pd.to_numeric(df["DTBMI_30D"], errors="coerce")

    s = OUTCOME_SANITY
    checks = {
        "OUTCOME_BMI_30D outside plausible range": ~df["OUTCOME_BMI_30D"].between(s["bmi_min"], s["bmi_max"]) & df["OUTCOME_BMI_30D"].notna(),
        "OUTCOME_WEIGHT_30D_KG outside plausible range": ~df["OUTCOME_WEIGHT_30D_KG"].between(s["weight_kg_min"], s["weight_kg_max"]) & df["OUTCOME_WEIGHT_30D_KG"].notna(),
        "BASELINE_BMI outside plausible range": ~df["BASELINE_BMI"].between(s["bmi_min"], s["bmi_max"]) & df["BASELINE_BMI"].notna(),
    }
    for label, mask in checks.items():
        n = int(mask.sum())
        log_rows.append({"check": label, "rows_set_to_missing": n})
        target = label.split(" ")[0]
        df.loc[mask, target] = np.nan

    df["BMI_CHANGE_30D"] = df["OUTCOME_BMI_30D"] - df["BASELINE_BMI"]
    df["BMI_PCT_CHANGE_30D"] = df["BMI_CHANGE_30D"] / df["BASELINE_BMI"] * 100
    df["WEIGHT_CHANGE_30D_KG"] = df["OUTCOME_WEIGHT_30D_KG"] - df["BASELINE_WEIGHT_KG"]
    implausible = df["BMI_PCT_CHANGE_30D"].abs() > s["max_abs_pct_bmi_change"]
    log_rows.append({"check": f"|30-day BMI change| > {s['max_abs_pct_bmi_change']}% of pre-op (outcome set to missing)",
                     "rows_set_to_missing": int(implausible.sum())})
    for c in ["OUTCOME_BMI_30D", "OUTCOME_WEIGHT_30D_KG", "BMI_CHANGE_30D", "BMI_PCT_CHANGE_30D", "WEIGHT_CHANGE_30D_KG"]:
        df.loc[implausible, c] = np.nan

    df["HAS_VALID_OUTCOME"] = (df["OUTCOME_BMI_30D"].notna() & df["OUTCOME_WEIGHT_30D_KG"].notna()
                               & df["BASELINE_BMI"].notna() & df["BASELINE_WEIGHT_KG"].notna())
    for yr, sub in df.groupby("OPYEAR"):
        log_rows.append({"check": f"{yr}: patients with a valid 30-day outcome after cleaning",
                         "rows_set_to_missing": None, "count": int(sub["HAS_VALID_OUTCOME"].sum()),
                         "of_total": len(sub)})
    return pd.DataFrame(log_rows)


def main():
    utils.log("=" * 70)
    utils.log("STEP 2: PREPROCESSING (units, outcomes, flags, severity)")
    utils.log("=" * 70)
    X = utils.load_modeling_frame(config.PREDICTORS_PATH)
    Y = utils.load_modeling_frame(config.OUTCOMES_PATH)
    df = X.merge(Y.drop(columns=["CASEID", "OPYEAR"]), on="PATIENT_KEY", how="left", validate="one_to_one")
    utils.log(f"Joined predictors + outcomes: {len(df):,} patients")

    hgt = config.VARIABLE_RATIONALE["Adiposity"]["variables"]["HGT"]
    sex_col = config.VARIABLE_RATIONALE["Socio-Environmental"]["variables"]["SEX"]["column"]
    height_cm = utils.height_to_cm_series(df[hgt["column"]], df[hgt["unit_column"]])
    height_cm = height_cm.where(height_cm.between(120, 230))   # implausible heights -> missing
    df["IDEAL_BODY_WEIGHT_KG"] = utils.devine_ibw_series(height_cm, df[sex_col]).where(height_cm.notna())

    for domain_name, domain in config.VARIABLE_RATIONALE.items():
        for var_name, spec in domain["variables"].items():
            if spec["column"] is None or spec["column"] not in df.columns:
                continue
            if spec["var_type"] == "numeric":
                df = process_numeric(df, var_name, spec)
            else:
                df = process_categorical(df, var_name, spec)
    df = df.copy()  # defragment

    cleaning_log = build_outcomes(df)
    utils.log("Outcome cleaning:\n" + cleaning_log.to_string(index=False))

    df.to_parquet(config.PREPROCESSED_PATH, index=False)
    utils.log(f"Saved {config.PREPROCESSED_PATH} ({len(df):,} rows, {df.shape[1]} columns)")

    # ---- Excel: flag summary overall and by year ----
    flag_vars = [c[:-6] for c in df.columns if c.endswith("__FLAG")]
    rows = []
    for v in flag_vars:
        row = {"variable": v, "n_flagged_all_years": int(df[f"{v}__FLAG"].sum()),
               "pct_flagged_all_years": round(100 * df[f"{v}__FLAG"].mean(), 2)}
        for yr, sub in df.groupby("OPYEAR"):
            row[f"pct_flagged_{yr}"] = round(100 * sub[f"{v}__FLAG"].mean(), 2)
        rows.append(row)
    flag_summary = pd.DataFrame(rows).sort_values("n_flagged_all_years", ascending=False)
    sev_cols = [c for c in df.columns if c.endswith("__SEVERITY")]
    sev_by_year = df.groupby("OPYEAR")[sev_cols].mean().T.round(2)
    sev_by_year.index = [i.replace("__SEVERITY", "") for i in sev_by_year.index]
    outcome_by_year = df[df["HAS_VALID_OUTCOME"]].groupby("OPYEAR")[
        ["BASELINE_BMI", "OUTCOME_BMI_30D", "BMI_CHANGE_30D", "BMI_PCT_CHANGE_30D",
         "BASELINE_WEIGHT_KG", "OUTCOME_WEIGHT_30D_KG", "WEIGHT_CHANGE_30D_KG", "FOLLOWUP_DAY"]].mean().round(3)
    utils.save_excel_sheets(utils.excel_path("02_flag_summary.xlsx"), {
        "flag_rates": flag_summary, "mean_severity_by_year": sev_by_year.reset_index().rename(columns={"index": "variable"}),
        "outcomes_by_year": outcome_by_year.reset_index(), "outcome_cleaning_log": cleaning_log})

    preview_cols = ["PATIENT_KEY", "OPYEAR", "IDEAL_BODY_WEIGHT_KG"] + \
        [c for c in df.columns if c.endswith(("__STD", "__FLAG", "__FLAG_REASON", "__SEVERITY"))] + \
        ["BASELINE_BMI", "OUTCOME_BMI_30D", "BMI_PCT_CHANGE_30D", "BASELINE_WEIGHT_KG", "OUTCOME_WEIGHT_30D_KG", "FOLLOWUP_DAY"]
    per_year = config.EXCEL_PREVIEW_ROWS // max(df["OPYEAR"].nunique(), 1)
    preview = df.groupby("OPYEAR", group_keys=False).head(per_year)[preview_cols]
    utils.save_excel_sheets(utils.excel_path("02_preprocessed_PREVIEW.xlsx"), {"preview": preview})

    # ---- figure: flag rates by year ----
    top = flag_summary[flag_summary["n_flagged_all_years"] > 0].head(15)
    years = sorted(df["OPYEAR"].unique())
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.8 / len(years)
    y = np.arange(len(top))
    for i, yr in enumerate(years):
        ax.barh(y + i * width, top[f"pct_flagged_{yr}"], height=width, label=yr)
    ax.set_yticks(y + width * (len(years) - 1) / 2)
    ax.set_yticklabels(top["variable"])
    ax.invert_yaxis()
    ax.set_xlabel("% of patients flagged abnormal")
    ax.set_title("Abnormal-value flag rates by variable and year")
    ax.legend(title="Year")
    utils.save_fig(fig, "02_flag_rates_by_year")
    utils.log("Step 2 complete.")


if __name__ == "__main__":
    main()
