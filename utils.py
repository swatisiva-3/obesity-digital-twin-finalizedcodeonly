"""
utils.py -- small shared helpers used across every numbered script.
"""
import json
import os
import sys
import time

import pandas as pd

import config


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def weight_to_kg(value, unit):
    if value is None or (isinstance(value, float) and (value != value)):
        return None
    if unit is None:
        return value
    u = str(unit).strip().lower()
    if u in ("kg", "kgs", "kilogram", "kilograms"):
        return value
    if u in ("lbs", "lb", "pound", "pounds"):
        return value * config.LBS_TO_KG
    return value


def height_to_cm(value, unit):
    if value is None or (isinstance(value, float) and (value != value)):
        return None
    if unit is None:
        return value
    u = str(unit).strip().lower()
    if u in ("cm", "cms", "centimeter", "centimeters"):
        return value
    if u in ("in", "inch", "inches"):
        return value * config.IN_TO_CM
    return value


def devine_ibw_kg(height_cm, sex):
    """Devine ideal body weight formula, kg. height_cm required, sex 'Male'/'Female'."""
    if height_cm is None or height_cm != height_cm:
        return None
    height_in = height_cm / config.IN_TO_CM
    inches_over_5ft = max(height_in - 60, 0)
    sex_str = str(sex).strip().lower()
    if sex_str.startswith("m"):
        return 50.0 + 2.3 * inches_over_5ft
    else:
        return 45.5 + 2.3 * inches_over_5ft


def save_json(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_df_excel(df, path, sheet_name="Sheet1", max_rows_for_full_excel=250_000):
    """
    Saves a DataFrame to .xlsx (xlsxwriter -- much faster than openpyxl on
    large sheets). Excel has a hard 1,048,576-row limit, and a full-cohort
    3-year merge (500k+ rows) is unwieldy to even open in Excel -- past
    max_rows_for_full_excel this writes a row-capped PREVIEW .xlsx (first N
    rows, clearly named) plus the FULL data as .csv.gz (opens fine in Excel
    too, just not as a native .xlsx), and logs which is which.
    """
    ensure_dir(os.path.dirname(path))
    t0 = time.time()
    if len(df) <= max_rows_for_full_excel:
        df.to_excel(path, sheet_name=sheet_name, index=False, engine="xlsxwriter")
        log(f"  wrote {path} ({len(df):,} rows, {time.time()-t0:.0f}s)")
    else:
        preview_path = path.replace(".xlsx", "_PREVIEW.xlsx")
        df.head(max_rows_for_full_excel).to_excel(preview_path, sheet_name=sheet_name, index=False, engine="xlsxwriter")
        full_path = path.replace(".xlsx", ".csv.gz")
        df.to_csv(full_path, index=False, compression="gzip")
        log(f"  {len(df):,} rows exceeds the {max_rows_for_full_excel:,}-row practical Excel cap -- wrote "
            f"preview (first {max_rows_for_full_excel:,} rows) {preview_path} and full data {full_path} "
            f"({time.time()-t0:.0f}s)")


EXCEL_SHEET_ROW_LIMIT = 1_048_575  # Excel's hard per-sheet limit (minus header row)


def save_excel_sheets(path, sheets):
    """
    Write several DataFrames into ONE .xlsx, one sheet each ({sheet_name: df}).
    Used for the "one Excel with everything" outputs: the pooled 3-year data
    is written as one sheet per year (each year is well under Excel's
    1,048,576-row per-sheet limit, even though the pooled total is not), so
    every row of every year still lives in a single workbook.
    """
    ensure_dir(os.path.dirname(path))
    t0 = time.time()
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        for name, df in sheets.items():
            if len(df) > EXCEL_SHEET_ROW_LIMIT:
                log(f"  sheet {name!r} has {len(df):,} rows > Excel limit -- truncating that sheet "
                    f"(full data is in the matching .csv.gz / .parquet)")
                df = df.head(EXCEL_SHEET_ROW_LIMIT)
            df.to_excel(writer, sheet_name=str(name)[:31], index=False)
    log(f"  wrote {path} ({', '.join(f'{k}: {len(v):,} rows' for k, v in sheets.items())}; {time.time()-t0:.0f}s)")


def save_df_by_year_excel(df, path, year_col="OPYEAR", extra_sheets=None):
    """One workbook, one sheet per year (+ optional extra summary sheets first)."""
    sheets = dict(extra_sheets or {})
    for yr, sub in df.groupby(year_col, sort=True):
        sheets[f"year_{yr}"] = sub
    save_excel_sheets(path, sheets)


def save_fig(fig, name, dpi=300, formats=("png",)):
    """Save a matplotlib figure into outputs/figures/ in each requested format."""
    import matplotlib.pyplot as plt
    out = ensure_dir(os.path.join(config.OUTPUT_DIR, "figures"))
    paths = []
    for fmt in formats:
        p = os.path.join(out, f"{name}.{fmt}")
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    log(f"  figure -> {', '.join(paths)}")
    return paths


def excel_path(name):
    return os.path.join(ensure_dir(config.MERGED_RAW_XLSX_DIR), name)


# ---- vectorized unit helpers (row-wise .apply is far too slow at 500k+ rows) ----
def weight_to_kg_series(values, units):
    v = pd.to_numeric(values, errors="coerce")
    u = units.astype("string").str.strip().str.lower()
    return v.where(~u.isin(["lbs", "lb", "pound", "pounds"]), v * config.LBS_TO_KG)


def height_to_cm_series(values, units):
    v = pd.to_numeric(values, errors="coerce")
    u = units.astype("string").str.strip().str.lower()
    return v.where(~u.isin(["in", "inch", "inches"]), v * config.IN_TO_CM)


def devine_ibw_series(height_cm, sex):
    """Devine ideal body weight (kg): 50 kg (men) / 45.5 kg (women) + 2.3 kg per inch over 5 ft."""
    import numpy as np
    inches_over_5ft = (height_cm / config.IN_TO_CM - 60).clip(lower=0)
    is_male = sex.astype("string").str.lower().str.startswith("m").fillna(False)
    return pd.Series(np.where(is_male, 50.0, 45.5), index=height_cm.index) + 2.3 * inches_over_5ft


def load_modeling_frame(path):
    """Read a parquet written by an earlier step, failing with a clear message if that step hasn't run."""
    if not os.path.exists(path):
        raise SystemExit(f"Missing input {path} -- run the earlier step that produces it first "
                         f"(see README 'Step inputs and outputs').")
    return pd.read_parquet(path)


MODEL_ID_COLUMNS = ["PATIENT_KEY", "CASEID", "OPYEAR", "HAS_VALID_OUTCOME",
                    "BASELINE_BMI", "OUTCOME_BMI_30D", "BMI_CHANGE_30D", "BMI_PCT_CHANGE_30D",
                    "BASELINE_WEIGHT_KG", "OUTCOME_WEIGHT_30D_KG", "WEIGHT_CHANGE_30D_KG", "FOLLOWUP_DAY",
                    "REOP30", "READ30", "INTV30"]


def read_model_columns(path, extra_suffixes=()):
    """
    Read only the columns the modeling steps need (ids, outcomes, __STD,
    __SEVERITY, scores) -- skipping the ~50 text __FLAG_REASON columns keeps
    memory to a fraction of the full file on a laptop.
    """
    import pyarrow.parquet as pq
    if not os.path.exists(path):
        raise SystemExit(f"Missing input {path} -- run the earlier step that produces it first.")
    names = pq.read_schema(path).names
    suffixes = ("__STD", "__SEVERITY", "_SCORE", "_SCORE_CLINICAL", "TOTAL_PHENOTYPE_SCORE",
                "TOTAL_PHENOTYPE_SCORE_CLINICAL") + tuple(extra_suffixes)
    keep = [c for c in names if c in MODEL_ID_COLUMNS or c.endswith(suffixes)]
    df = pd.read_parquet(path, columns=keep)
    df["OPYEAR"] = df["OPYEAR"].astype(str)
    for c in df.columns:
        if c.endswith("__STD") and not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype("category")
    return df
