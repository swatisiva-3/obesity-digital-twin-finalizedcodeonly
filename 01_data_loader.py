"""
01_data_loader.py
==================
Loads 2017 / 2020 / 2024, standardizes every column name and category-string
spelling to one canonical schema (column_mapping.py), joins in each year's
30-day BMI/weight outcome (2017 needs a join against bmifinal; 2020/2024
already have it in main), and MERGES all three years into one pooled
analytic dataset -- this is the Python port of your C++ multi-year merge
program, extended with what direct inspection of the real files found.

INPUT:  data/MBSAQIP DATA FINAL/MBSAQIP_PUF_<year>_TXT/*.txt  (config.YEAR_FILES)
OUTPUT: outputs/01_predictors_all_variables.parquet   -> all predictor variables, 3 years merged
        outputs/01_outcomes_bmi_weight_30day.parquet  -> 30-day BMI + weight outcomes (join key PATIENT_KEY)
            ^ these TWO files are what Steps 2-8 train / validate / test on
        outputs/excel/01_merged_all_variables.xlsx       (Excel copy of the predictors, one sheet per year)
        outputs/excel/01_outcomes_bmi_weight_30day.xlsx  (Excel copy of the outcomes, one sheet per year)
        outputs/excel/01_load_summary.xlsx               (per-year counts, completeness, every rename /
                                                          value harmonization applied, CASEID-in-all-4-files)

Run: python 01_data_loader.py            (add --no-excel to skip the ~3 min full Excel export)
"""
import argparse
import os

import pandas as pd

import config
import column_mapping as cm
import utils


def _read_raw(path):
    if path is None or not os.path.exists(path):
        return None
    return pd.read_csv(path, sep=config.FILE_DELIMITER, dtype=str, encoding="utf-8",
                        engine="c", on_bad_lines="warn")


def load_year_main(year):
    """Load + standardize one year's main file (rename, value-normalize)."""
    path = config.year_file_path(year, "main")
    if path is None or not os.path.exists(path):
        utils.log(f"  [{year}] main file not found at {path} -- skipping this year entirely")
        return None
    df = _read_raw(path)
    utils.log(f"  [{year}] loaded main: {len(df):,} rows, {len(df.columns)} raw columns")
    df = df.rename(columns=cm.canonical_rename_map(year))
    df["OPYEAR"] = df["OPYEAR"].astype(str)
    cm.apply_value_normalization(df, year)
    return df


def attach_2017_outcome(df):
    """
    2017-only: join bmifinal.txt onto main by CASEID for the 30-day outcome.

    bmifinal is a LONGITUDINAL follow-up table -- confirmed by inspection, NOT
    documented in any PUF user guide: one row per follow-up encounter for
    patients seen more than once before day 30 (up to 14 rows for one patient
    in this real file). Within a patient, HGT/WGT_CLOSEST/WGT_HIGH_BAR/BMI/
    BMI_HIGH_BAR are constant (they're an echo of the PRE-OP values already in
    main) -- the columns that actually vary row-to-row are WGT_DISCH/
    BMI_DISCH (the measurement taken AT that encounter) and DTBMI (days
    post-op that encounter happened, 0-30). So the correct 30-day outcome is
    NOT bmifinal's "BMI"/"WGT_CLOSEST" columns (those are just pre-op again) --
    it's BMI_DISCH/WGT_DISCH from the row with the LARGEST DTBMI per patient
    (i.e. the encounter closest to day 30), which is what this function
    selects before joining. Getting this wrong (an earlier pass at this did)
    silently duplicates rows AND labels every 2017 patient's post-op outcome
    with their pre-op value.
    """
    path = config.year_file_path("2017", "bmi")
    if path is None or not os.path.exists(path):
        utils.log("  [2017] bmifinal.txt not found -- 2017 will have NO 30-day outcome "
                   "(Optuna/MLP/SHAP/classification steps will skip 2017 rows)")
        return df
    bmi = _read_raw(path)
    utils.log(f"  [2017] loaded bmifinal (outcome source): {len(bmi):,} rows, "
              f"{bmi['CASEID'].nunique():,} unique CASEIDs (longitudinal -- multiple follow-up rows per patient)")
    bmi["DTBMI_f"] = pd.to_numeric(bmi["DTBMI"], errors="coerce")
    bmi["WGT_DISCH_f"] = pd.to_numeric(bmi["WGT_DISCH"], errors="coerce")
    bmi["BMI_DISCH_f"] = pd.to_numeric(bmi["BMI_DISCH"], errors="coerce")
    closest = (bmi.dropna(subset=["DTBMI_f"])
                  .sort_values("DTBMI_f")
                  .groupby("CASEID", as_index=False)
                  .tail(1))
    closest = closest.rename(columns={
        "WGT_DISCH_f": "WGT_CLOSEST30D",
        "WGTUNIT_DISCH": "WGTUNIT_CLOSEST30D",
        "BMI_DISCH_f": "BMI_CLOSEST30D",
        "DTBMI_f": "DTBMI_30D",
    })[["CASEID", "WGT_CLOSEST30D", "WGTUNIT_CLOSEST30D", "BMI_CLOSEST30D", "DTBMI_30D"]]
    utils.log(f"  [2017] reduced to {len(closest):,} patients (one row each: the encounter closest to day 30, "
              f"median day {closest['DTBMI_30D'].median():.0f})")
    merged = df.merge(closest, on="CASEID", how="left")
    n_matched = merged["BMI_CLOSEST30D"].notna().sum()
    utils.log(f"  [2017] outcome joined by CASEID: {n_matched:,}/{len(merged):,} rows have a 30-day BMI value "
              f"(mean BMI drop from pre-op to this follow-up: "
              f"{(pd.to_numeric(merged['BMI'], errors='coerce') - merged['BMI_CLOSEST30D']).mean():.2f} kg/m^2)")
    return merged


def load_ancillary(year, key):
    path = config.year_file_path(year, key)
    if path is None or not os.path.exists(path):
        return None
    df = _read_raw(path)
    df = df.rename(columns=cm.canonical_rename_map(year))
    return df


def build_all4files_linkage_sheet(year, main_caseids):
    """Bonus sheet: which CASEIDs from main also appear in intv/reop/read for this year."""
    rows = []
    for key in ("intv", "reop", "read"):
        df = load_ancillary(year, key)
        present = set(df["CASEID"]) if df is not None else set()
        rows.append({"file": key, "n_caseids": len(present)})
    intv = load_ancillary(year, "intv")
    reop = load_ancillary(year, "reop")
    read = load_ancillary(year, "read")
    if intv is None and reop is None and read is None:
        return None
    ids_intv = set(intv["CASEID"]) if intv is not None else set()
    ids_reop = set(reop["CASEID"]) if reop is not None else set()
    ids_read = set(read["CASEID"]) if read is not None else set()
    all4 = main_caseids & ids_intv & ids_reop & ids_read if (ids_intv and ids_reop and ids_read) else set()
    summary = pd.DataFrame([
        {"year": year, "in_main": len(main_caseids), "in_intv": len(ids_intv),
         "in_reop": len(ids_reop), "in_read": len(ids_read), "in_all_4": len(all4)}
    ])
    return summary


# Columns kept in the merged dataset: identity + every rationale variable's
# real column (skipping the ones that are config-level None) + outcome vars
# + the REOP30/READ30/INTV30 30-day-event flags (asked for explicitly, and
# needed for LINK_MODE="strict_intersection" sanity checks).
def rationale_columns():
    cols = set()
    for domain in config.VARIABLE_RATIONALE.values():
        for spec in domain["variables"].values():
            if spec["column"]:
                cols.add(spec["column"])
            if spec.get("unit_column"):
                cols.add(spec["unit_column"])
    return sorted(cols)


def build_keep_columns():
    keep = {"CASEID", "OPYEAR"}
    keep.update(rationale_columns())
    for spec in config.OUTCOME_VARS.values():
        keep.add(spec["column"])
        if spec.get("unit_column"):
            keep.add(spec["unit_column"])
    keep.update({"REOP30", "READ30", "INTV30", "DTBMI_30D"})
    return keep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=config.SAMPLE_SIZE)
    parser.add_argument("--no-excel", action="store_true", help="skip the (slow) full-data Excel exports")
    args = parser.parse_args()

    utils.log("=" * 70)
    utils.log("STEP 1: LOAD + STANDARDIZE + MERGE 2017 / 2020 / 2024")
    utils.log("=" * 70)

    keep_cols = build_keep_columns()
    year_frames = []
    linkage_sheets = []

    for year in config.YEARS:
        utils.log(f"-- {year} --")
        df = load_year_main(year)
        if df is None:
            continue
        if year == "2017":
            df = attach_2017_outcome(df)

        missing_from_this_year = [c for c in keep_cols if c not in df.columns]
        if missing_from_this_year:
            utils.log(f"  [{year}] columns not present as raw columns in this year's file: "
                      f"{missing_from_this_year}"
                      + (" (HISPANIC is re-derived from the 2024 race field next)" if year == "2024" else ""))
        for c in keep_cols:
            if c not in df.columns:
                df[c] = pd.NA
        df = df[sorted(keep_cols)].copy()
        df = cm.harmonize_context_variables(df, year)

        # numeric coercion for the variables that need it
        numeric_cols = [spec["column"] for domain in config.VARIABLE_RATIONALE.values()
                        for spec in domain["variables"].values()
                        if spec["column"] and spec["var_type"] == "numeric" and spec["column"] != "OPYEAR"]
        # NBHTN_MEDS is top-coded as the TEXT "3 or more" in every year (not just 2017's
        # "3+" -- confirmed present in the raw 2020/2024 files too), which pd.to_numeric
        # would otherwise silently turn into NaN for ~35% of rows in every year. Treat it
        # as 3 for scoring (severity_span=3 already saturates at that value anyway).
        if "NBHTN_MEDS" in df.columns:
            df["NBHTN_MEDS"] = df["NBHTN_MEDS"].replace({"3 or more": "3", "3+": "3"})
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        for c in [config.OUTCOME_VARS["bmi_30d"]["column"], config.OUTCOME_VARS["weight_30d"]["column"],
                  "DTBMI_30D"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        # REOP30/READ30/INTV30 are "Yes"/"No" categoricals in every year -- keep as strings.

        if args.sample_size and len(df) > args.sample_size:
            df = df.sample(n=args.sample_size, random_state=config.RANDOM_SEED)
            utils.log(f"  [{year}] sub-sampled to {len(df):,} rows (config.SAMPLE_SIZE)")

        year_frames.append(df)

        linkage = build_all4files_linkage_sheet(year, set(df["CASEID"]))
        if linkage is not None:
            linkage_sheets.append(linkage)

    if not year_frames:
        raise SystemExit("No year data loaded -- check config.YEAR_FOLDERS / config.YEAR_FILES paths.")

    merged = pd.concat(year_frames, ignore_index=True)
    utils.log("-" * 70)
    utils.log(f"MERGED: {len(merged):,} total patients across {merged['OPYEAR'].nunique()} years")
    utils.log(merged.groupby("OPYEAR").size().to_string())

    # global patient key: CASEIDs are only guaranteed unique WITHIN a year
    merged.insert(0, "PATIENT_KEY", merged["OPYEAR"].astype(str) + "_" + merged["CASEID"].astype(str))
    n_dupe = merged["PATIENT_KEY"].duplicated().sum()
    if n_dupe:
        utils.log(f"WARNING: {n_dupe} duplicated PATIENT_KEYs -- keeping first occurrence")
        merged = merged.drop_duplicates("PATIENT_KEY")

    bmi_col = config.OUTCOME_VARS["bmi_30d"]["column"]
    wgt_col = config.OUTCOME_VARS["weight_30d"]["column"]
    wgt_unit_col = config.OUTCOME_VARS["weight_30d"]["unit_column"]
    outcome_cols = [bmi_col, wgt_col, wgt_unit_col, "DTBMI_30D"]

    # ---- per-year load summary (Excel) -----------------------------------
    summary_rows = []
    for year, sub in merged.groupby("OPYEAR"):
        n = len(sub)
        n_ok = int((sub[bmi_col].notna() & sub[wgt_col].notna()).sum())
        utils.log(f"  {year}: {n_ok:,}/{n:,} ({100*n_ok/n:.1f}%) have both 30-day BMI and weight")
        summary_rows.append({
            "OPYEAR": year, "patients": n, "with_30d_BMI_and_weight": n_ok,
            "pct_with_outcome": round(100 * n_ok / n, 2),
            "mean_preop_BMI": round(pd.to_numeric(sub["BMI"], errors="coerce").mean(), 2),
            "mean_30d_BMI": round(sub[bmi_col].mean(), 2),
            "median_followup_day": sub["DTBMI_30D"].median(),
            "outcome_source": ("2017_MBSAQIP_bmifinal.txt (closest-to-day-30 encounter)" if year == "2017"
                               else f"{year}_MBSAQIP_mainfinal.txt"),
        })
    load_summary = pd.DataFrame(summary_rows)
    completeness = (merged.drop(columns=["PATIENT_KEY"]).groupby("OPYEAR")
                    .apply(lambda g: g.notna().mean().round(4)).T.reset_index()
                    .rename(columns={"index": "column"}))
    import column_mapping as _cm
    harmonization = pd.DataFrame(
        [{"year": y, "column": c, "raw_value": k, "harmonized_value": v}
         for (y, c), m in _cm.VALUE_NORMALIZATION.items() for k, v in m.items()]
        + [{"year": "all", "column": "SURGSPECIALTY_BAR", "raw_value": k, "harmonized_value": v}
           for k, v in _cm.SURGSPECIALTY_HARMONIZE.items()]
        + [{"year": "all", "column": "SEX", "raw_value": k, "harmonized_value": v}
           for k, v in _cm.SEX_HARMONIZE.items()]
        + [{"year": "2024", "column": "HISPANIC", "raw_value": "(no raw column)",
            "harmonized_value": "Yes if 'Hispanic or Latino' selected as a race option, else 'Not collected (2024)'"},
           {"year": "all", "column": "RACE_PUF", "raw_value": "any 'A~B' multi-select / low-frequency combos",
            "harmonized_value": "Multiracial"}])
    aliases = pd.DataFrame([{"year": y, "raw_column": k, "canonical_column": v}
                            for y, m in _cm.HEADER_ALIASES.items() for k, v in m.items()])
    sheets = {"load_summary": load_summary, "column_completeness": completeness,
              "value_harmonization": harmonization, "column_aliases": aliases}
    if linkage_sheets:
        sheets["all4files_linkage"] = pd.concat(linkage_sheets, ignore_index=True)
    utils.save_excel_sheets(utils.excel_path("01_load_summary.xlsx"), sheets)

    # ---- the TWO files every downstream step trains/validates/tests on ---
    predictors = merged.drop(columns=outcome_cols)
    outcomes = merged[["PATIENT_KEY", "CASEID", "OPYEAR", "BMI", "WGT_CLOSEST", "WGTUNIT_CLOSEST"] + outcome_cols].rename(
        columns={"BMI": "BMI_PREOP", "WGT_CLOSEST": "WGT_PREOP", "WGTUNIT_CLOSEST": "WGTUNIT_PREOP"})

    utils.ensure_dir(config.OUTPUT_DIR)
    predictors.to_parquet(config.PREDICTORS_PATH, index=False)
    outcomes.to_parquet(config.OUTCOMES_PATH, index=False)
    utils.log(f"Saved {config.PREDICTORS_PATH} and {config.OUTCOMES_PATH}")

    if not args.no_excel and not config.WRITE_FULL_PATIENT_EXCEL:
        utils.log("config.WRITE_FULL_PATIENT_EXCEL = False -> writing only the first "
                  f"{config.EXCEL_PREVIEW_ROWS} rows per year to the Excel copies")
        predictors_x = predictors.groupby("OPYEAR", group_keys=False).head(config.EXCEL_PREVIEW_ROWS)
        outcomes_x = outcomes.groupby("OPYEAR", group_keys=False).head(config.EXCEL_PREVIEW_ROWS)
    else:
        predictors_x, outcomes_x = predictors, outcomes
    if not args.no_excel:
        utils.save_df_by_year_excel(predictors_x, utils.excel_path("01_merged_all_variables.xlsx"),
                                    extra_sheets={"README": pd.DataFrame({"note": [
                                        "All 3 years merged into one analytic dataset (one sheet per year only "
                                        "because Excel caps a sheet at 1,048,576 rows).",
                                        "PATIENT_KEY = OPYEAR_CASEID is the join key to 01_outcomes_bmi_weight_30day.xlsx.",
                                        "Column names are the canonical short PUF codes (see 01_load_summary.xlsx).",
                                    ]})})
        utils.save_df_by_year_excel(outcomes_x, utils.excel_path("01_outcomes_bmi_weight_30day.xlsx"),
                                    extra_sheets={"summary": load_summary})
    utils.log("Step 1 complete.")


if __name__ == "__main__":
    main()
