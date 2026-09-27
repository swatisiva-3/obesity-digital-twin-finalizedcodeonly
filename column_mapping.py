"""
column_mapping.py
==================
THE PORT OF YOUR C++ MERGE PROGRAM'S SCHEMA LOGIC, adapted to the REAL
2017 / 2020 / 2024 files (inspected directly -- header names, delimiter,
and value formats were all confirmed against the actual .txt files before
a single line of this was written, not assumed from the PUF user guides).

Your original C++ program merged 2017/2020/2023 MAIN files into one
185-column master schema using a small alias table (race_PUF->RACE_PUF,
hispanic->HISPANIC, HTN_MEDS->NBHTN_MEDS) plus one value normalization
("3+" -> "3 or more" for 2017's hypertension-med count). That logic is
reproduced here as CANONICAL_COLUMN_ALIASES / VALUE_NORMALIZATION, extended
with what real-data inspection turned up that the C++ code didn't cover:

  1. The 2024 file (185 columns -- this IS the "newest/reference year"
     master schema your C++ program was built around, just delivered a
     year later than "2023") ships with full HUMAN-READABLE column labels
     ("Case Identification Number", "Sex", "Year of Operation", ...)
     instead of the short PUF codes 2017/2020 use. Every raw MBSAQIP
     variable this pipeline needs is mapped below.
  2. 2024 has NO Hispanic/ethnicity column at all (confirmed absent, not
     renamed -- see YEAR_COLUMN_ABSENT).
  3. Three variables have DIFFERENT category-string spellings in 2017 than
     the 2020/2024 "new registry" wording, which would otherwise silently
     fail to match config.VARIABLE_RATIONALE's category_scores and score
     as missing for every 2017 patient:
       - DIABETES:        "Insulin"/"Non-Insulin"      -> "Yes, insulin"/"Yes, non-insulin"
       - ASACLASS:         "1-No Disturb" ... "5-Moribund" -> "ASA I - Normal/Healthy" ... "ASA V - Moribund"
       - FUNSTATPRESURG:  "Partially Dependent"/"Totally Dependent" (capital D)
                            -> "Partially dependent"/"Totally dependent" (lowercase d)
     These were found by directly sampling values in the real 2017 file,
     not documented anywhere in the PUF user guides.
  4. 2024 no longer asks Hispanic ethnicity as a separate question -- it
     became one of the race options. HISPANIC is re-derived for 2024 from
     the race field (see harmonize_race_and_ethnicity), and RACE / SEX /
     SURGSPECIALTY_BAR spellings are harmonized across years.
"""
import pandas as pd

# ---------------------------------------------------------------------------
# Each year's raw CASEID / OPYEAR column name (before renaming to canonical)
# ---------------------------------------------------------------------------
RAW_CASEID_COLUMN = {
    "2017": "CASEID",
    "2020": "CASEID",
    "2024": "Case Identification Number",
}
RAW_OPYEAR_COLUMN = {
    "2017": "OPYEAR",
    "2020": "OPYEAR",
    "2024": "Year of Operation",
}

# ---------------------------------------------------------------------------
# Header aliases: raw column name (as it appears in that year's file) ->
# canonical short PUF code used everywhere downstream (config.py,
# scoring_engine.py, model_utils.py). Only columns actually needed by
# config.VARIABLE_RATIONALE / config.OUTCOME_VARS / the REOP30-READ30-INTV30
# extras are mapped -- this is deliberately not a full 185-column dictionary.
# ---------------------------------------------------------------------------
HEADER_ALIASES = {
    "2017": {
        # confirmed via real-file inspection; race_PUF/hispanic/HTN_MEDS
        # match your C++ program's alias table exactly.
        "race_PUF": "RACE_PUF",
        "hispanic": "HISPANIC",
        "HTN_MEDS": "NBHTN_MEDS",
        # config.VARIABLE_RATIONALE's "CHRONIC_STEROIDS" entry looks up a
        # column literally named IMMUNOSUPR_THER (the 2021+ PUF name) -- but
        # BOTH 2017 and 2020's real files still use the OLD name
        # CHRONIC_STEROIDS (confirmed by header inspection; the "renamed
        # starting with the 2021 PUF" note in config.py turned out to only
        # be true for 2024, not 2020). Aliased here so all 3 years land on
        # the one column name VARIABLE_RATIONALE expects, with no change
        # needed to VARIABLE_RATIONALE itself.
        "CHRONIC_STEROIDS": "IMMUNOSUPR_THER",
    },
    "2020": {
        # 2020 uses canonical short codes for almost everything this
        # pipeline needs, EXCEPT CHRONIC_STEROIDS -- see the 2017 note above,
        # confirmed true for 2020 too by direct header inspection.
        "CHRONIC_STEROIDS": "IMMUNOSUPR_THER",
    },
    "2024": {
        # 2024 ships human-readable labels for every column. Only the ones
        # this pipeline actually touches are mapped.
        "Sex": "SEX",
        "Age (years)": "AGE",
        "Race": "RACE_PUF",
        "Height": "HGT",
        "Height Unit": "HGTUNIT",
        "Highest Pre-Op Weight recorded": "WGT_HIGH_BAR",
        "Highest Pre-Op Weight recorded Unit": "WGT_HIGH_UNIT_BAR",
        "Highest Recorded Pre-Op BMI": "BMI_HIGH_BAR",
        "Pre-Op Weight closest to bariatric surgery": "WGT_CLOSEST",
        "Pre-Op Weight closest to bariatric surgery Unit": "WGTUNIT_CLOSEST",
        "Pre-Op BMI closest to bariatric surgery": "BMI",
        "Pre-Op Functional Health Status": "FUNSTATPRESURG",
        "Current smoker within one year": "SMOKER",
        "Pre-Op Diabetes Mellitus": "DIABETES",
        "Pre-Op Steroid/Immunosuppressant Use for Chronic Condition": "IMMUNOSUPR_THER",
        "Pre-Op history of COPD": "COPD",
        "Pre-Op Sleep Apnea": "SLEEP_APNEA",
        "Pre-Op GERD": "GERD",
        "Previous Foregut Surgery": "PREVIOUS_SURGERY",
        "Number of Anti-Hypertensive Medications": "NBHTN_MEDS",
        "Pre-Op Hyperlipidemia": "HYPERLIPIDEMIA",
        "Pre-Op Venous Thrombosis Requiring Therapy": "HISTORY_DVT",
        "Pre-Op Venous Stasis": "VENOUS_STASIS",
        "Pre-Op Requiring or on dialysis": "DIALYSIS",
        "Pre-Op Renal Insufficiency": "RENAL_INSUFFICIENCY",
        "Pre-Op Albumin Lab Value (g/dL)": "ALBUMIN",
        "Pre-Op Creatinine Lab Value (mg/dL)": "CREATININE",
        "ASA Class": "ASACLASS",
        "Operation Length (minutes)": "OPLENGTH",
        "Medical specialty of the physician performing the Primary Procedure": "SURGSPECIALTY_BAR",
        "Discharge Destination": "DISCHARGE_DESTINATION",
        "At Least One Reoperation within 30 days of operation": "REOP30",
        "At Least One Readmission within 30 days of operation": "READ30",
        "At Least One Intervention within 30 days of operation": "INTV30",
        "Follow-up data (30 Day) Captured": "FOLLOW_30DAYS_BAR",
        "Post-Op Weight Closest to Day 30": "WGT_CLOSEST30D",
        "Post-Op Weight Closest to Day 30 Unit": "WGTUNIT_CLOSEST30D",
        "Post-Op BMI Closest to Day 30": "BMI_CLOSEST30D",
        "Days from bariatric surgery to Post-Op BMI measurement": "DTBMI_30D",
    },
}

# ---------------------------------------------------------------------------
# Columns genuinely ABSENT for a given year (not renamed -- just not there).
# 01_data_loader.py sets these to NaN for that year's rows rather than
# raising, and the scoring engine redistributes their weight within the
# domain automatically (same mechanism config.py already uses for
# MOBILITY_DEVICE / PRIORITY, which are None for ALL years by design).
# ---------------------------------------------------------------------------
YEAR_COLUMN_ABSENT = {
    "2017": set(),
    "2020": set(),
    # Confirmed via direct header search: no Hispanic/ethnicity column
    # anywhere in the 2024 files. It is RE-DERIVED from the 2024 race field
    # by harmonize_race_and_ethnicity() below, so 2024 still ends up with a
    # HISPANIC value -- this entry just documents that it isn't a raw column.
    "2024": {"HISPANIC"},
}

# ---------------------------------------------------------------------------
# Value normalization: (year, canonical_column) -> {raw_value: normalized}.
# Applied AFTER header renaming, BEFORE scoring. Only 2017 needs this --
# 2020 and 2024 already use the "new registry" wording that
# config.VARIABLE_RATIONALE's category_scores keys are written against.
# ---------------------------------------------------------------------------
VALUE_NORMALIZATION = {
    ("2017", "NBHTN_MEDS"): {
        "3+": "3 or more",  # your C++ program's one hardcoded rule
    },
    ("2017", "DIABETES"): {
        "Insulin": "Yes, insulin",
        "Non-Insulin": "Yes, non-insulin",
    },
    ("2017", "ASACLASS"): {
        "1-No Disturb": "ASA I - Normal/Healthy",
        "2-Mild Disturb": "ASA II - Mild systemic disease",
        "3-Severe Disturb": "ASA III - Severe systemic disease",
        "4-Life Threat": "ASA IV - Severe systemic disease threat to life",
        "5-Moribund": "ASA V - Moribund",
    },
    ("2017", "FUNSTATPRESURG"): {
        "Partially Dependent": "Partially dependent",
        "Totally Dependent": "Totally dependent",
    },
}


# ---------------------------------------------------------------------------
# Cross-year harmonization of the CONTEXT variables (RACE / HISPANIC / SEX /
# SURGSPECIALTY_BAR). These don't carry a hand-set severity score, but they
# ARE model features, so the same real-world category must be spelled the
# same way in all 3 years or the model sees three different "categories".
# All of the below was found by tabulating the real merged file by year.
# ---------------------------------------------------------------------------
SURGSPECIALTY_HARMONIZE = {
    # 2017 capitalizes differently and names "Other" differently from 2020/2024
    "Metabolic and Bariatric Surgeon": "Metabolic and bariatric surgeon",
    "General Surgeon": "General surgeon",
    "Interventional Radiologist": "Interventional radiologist",
    "Other Healthcare Professional": "Other",
}
SEX_HARMONIZE = {"Non-binary (prior to 2025)": "Non-binary"}

HISPANIC_LABEL = "Hispanic or Latino"


def harmonize_race_and_ethnicity(df, year):
    """
    2024 changed how race/ethnicity is collected: there is no separate
    Hispanic question any more -- "Hispanic or Latino" became one of the
    select-all-that-apply RACE options (values like "White~Hispanic or
    Latino"). 2017/2020 asked race and Hispanic ethnicity separately.

    To keep one consistent definition across all 3 years:
      * HISPANIC for 2024 is partly recoverable from the race field: "Yes"
        if "Hispanic or Latino" was one of the selected options. Everyone
        else gets "Not collected (2024)" -- NOT "No". Checked on the real
        data: only 0.9% of 2024 patients select Hispanic as a race option
        vs ~13% answering "Yes" to the separate question in 2017/2020, so
        "didn't select it" clearly does not mean "not Hispanic".
      * RACE for 2024 has the "Hispanic or Latino" component removed (it now
        lives in HISPANIC, same as 2017/2020). A patient who selected ONLY
        "Hispanic or Latino" gets race "Unknown/Not Reported" (no race was
        specified -- we don't guess one).
      * Any multi-selection ("~") and "Race combinations with low frequency"
        -> "Multiracial" in every year.
    """
    race = df["RACE_PUF"].astype("string")

    if year == "2024":
        parts = race.str.split("~")
        has_hisp = parts.apply(lambda p: isinstance(p, list) and HISPANIC_LABEL in p)
        df["HISPANIC"] = "Not collected (2024)"
        df.loc[has_hisp, "HISPANIC"] = "Yes"

        def strip_hisp(p):
            if not isinstance(p, list):
                return pd.NA
            rest = [x for x in p if x != HISPANIC_LABEL]
            if not rest:
                return "Unknown/Not Reported"
            return "~".join(rest)
        race = parts.apply(strip_hisp).astype("string")

    race = race.where(~race.str.contains("~", na=False), "Multiracial")
    race = race.replace({"Race combinations with low frequency": "Multiracial"})
    df["RACE_PUF"] = race
    return df


def harmonize_context_variables(df, year):
    df = harmonize_race_and_ethnicity(df, year)
    if "SURGSPECIALTY_BAR" in df.columns:
        df["SURGSPECIALTY_BAR"] = df["SURGSPECIALTY_BAR"].replace(SURGSPECIALTY_HARMONIZE)
    if "SEX" in df.columns:
        df["SEX"] = df["SEX"].replace(SEX_HARMONIZE)
    return df


def canonical_rename_map(year):
    """Full raw->canonical rename dict for this year, CASEID/OPYEAR included."""
    m = dict(HEADER_ALIASES.get(year, {}))
    m[RAW_CASEID_COLUMN[year]] = "CASEID"
    m[RAW_OPYEAR_COLUMN[year]] = "OPYEAR"
    return m


def apply_value_normalization(df, year):
    """Mutates df in place: fixes 2017's category-string spelling mismatches."""
    for (yr, col), mapping in VALUE_NORMALIZATION.items():
        if yr != year or col not in df.columns:
            continue
        df[col] = df[col].replace(mapping)
    return df
