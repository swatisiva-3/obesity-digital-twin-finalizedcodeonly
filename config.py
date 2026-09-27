"""
config.py
=========
THE ONE FILE YOU EDIT WHEN YOU SWITCH DATASETS. Every other script imports
from here and does not hard-code a file path, column name, weight, or
clinical threshold.

>>> ARCHITECTURE (Sep 24 rebuild) <<<
Earlier versions of this pipeline scored 2017/2020/2024 as three
INDEPENDENT datasets (separate outputs/models per year). This version
MERGES all three years into one combined analytic dataset -- one row per
patient, an OPYEAR column telling you which year they're from -- because
that's what you asked for this time: one pooled 3-year PhenoGraph/UMAP
population, one Optuna run, one MLP, one SHAP explanation, all trained on
the combined cohort. 01_data_loader.py is where the merge happens; it is a
direct port of your C++ multi-year merge program's logic (master-schema
column mapping + a historical alias table + one value normalization),
extended with what real-file inspection turned up -- see column_mapping.py
for the full story and exactly which column names/value spellings differ
across 2017/2020/2024 and why.

Sections in this file, in order:
  1. Real file locations (data/, outputs/, models/)
  2. Case-linkage settings
  3. Outcome variables (BMI/weight at 30 days) + how each year supplies them
  4. The variable rationale (domains, weights, column mapping, clinical ranges)
  5. Domain-level weights
  6. Unit standardization
  7. Optuna weight-learning settings
  8. Train/validation/test split (permanent, 70/15/15)
  9. Model / SHAP / PhenoGraph / baseline settings
"""

import os

# ---------------------------------------------------------------------------
# 1. REAL FILE LOCATIONS
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "MBSAQIP DATA FINAL")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

FILE_DELIMITER = "\t"
CASEID_COLUMN = "CASEID"   # canonical name, AFTER column_mapping.py renames it

YEARS = ["2017", "2020", "2024"]

# Exactly the folder/file names you have on disk right now (confirmed by
# listing the real files, not assumed):
#   data/MBSAQIP DATA FINAL/MBSAQIP_PUF_2017_TXT/2017_MBSAQIP_mainfinal.txt (+ bmi/intv/read/reop)
#   data/MBSAQIP DATA FINAL/MBSAQIP_PUF_2020_TXT/2020_MBSAQIP_mainfinal.txt (+ intv/read/reop)
#   data/MBSAQIP DATA FINAL/MBSAQIP_PUF_2024_TXT/2024_MBSAQIP_mainfinal.txt (+ intv/read/reop)
# 2017 is the only year with a bmifinal file (old MBSAQIP registry format,
# 2016-2019) -- 2020/2024 are "new registry" and don't have one. This is
# also exactly why 2017 needs its 30-day BMI/weight outcome joined in from
# a second file while 2020/2024 already carry it in main -- see Section 3.
YEAR_FOLDERS = {
    "2017": "MBSAQIP_PUF_2017_TXT",
    "2020": "MBSAQIP_PUF_2020_TXT",
    "2024": "MBSAQIP_PUF_2024_TXT",
}
YEAR_FILES = {
    "2017": {
        "main": "2017_MBSAQIP_mainfinal.txt",
        "bmi": "2017_MBSAQIP_bmifinal.txt",   # supplies the 30-day outcome for 2017
        "intv": "2017_MBSAQIP_intvfinal.txt",
        "reop": "2017_MBSAQIP_reopfinal.txt",
        "read": "2017_MBSAQIP_readfinal.txt",
    },
    "2020": {
        "main": "2020_MBSAQIP_mainfinal.txt",   # already has the 30-day outcome
        "intv": "2020_MBSAQIP_intvfinal.txt",
        "reop": "2020_MBSAQIP_reopfinal.txt",
        "read": "2020_MBSAQIP_readfinal.txt",
    },
    "2024": {
        "main": "2024_MBSAQIP_mainfinal.txt",   # already has the 30-day outcome
        "intv": "2024_MBSAQIP_intvfinal.txt",
        "reop": "2024_MBSAQIP_reopfinal.txt",
        "read": "2024_MBSAQIP_readfinal.txt",
    },
}


def year_file_path(year, key):
    """Absolute path to one raw file, or None if that key isn't defined for this year."""
    if year not in YEAR_FILES or key not in YEAR_FILES[year]:
        return None
    return os.path.join(DATA_DIR, YEAR_FOLDERS[year], YEAR_FILES[year][key])


# ---- Every step's input/output file (one pooled 3-year dataset, not one per year) ----
# Step 1 writes TWO files, as asked: all predictor variables, and the 30-day
# BMI/weight outcomes. Steps 2-8 join them on PATIENT_KEY (= OPYEAR_CASEID,
# because CASEIDs are only unique within a year).
PREDICTORS_PATH = os.path.join(OUTPUT_DIR, "01_predictors_all_variables.parquet")   # 01 -> 02
OUTCOMES_PATH = os.path.join(OUTPUT_DIR, "01_outcomes_bmi_weight_30day.parquet")    # 01 -> 02
PREPROCESSED_PATH = os.path.join(OUTPUT_DIR, "02_preprocessed.parquet")             # 02 -> 03
SCORED_PATH = os.path.join(OUTPUT_DIR, "03_scored.parquet")                         # 03 -> 04
FINAL_SCORED_PATH = os.path.join(OUTPUT_DIR, "04_final_scored.parquet")             # 04 -> 05/06/07/08
MERGED_RAW_XLSX_DIR = os.path.join(OUTPUT_DIR, "excel")   # every step's Excel output
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")          # every step's figures

# ---------------------------------------------------------------------------
# 2. CASE-LINKAGE SETTINGS
# ---------------------------------------------------------------------------
# Same reasoning as every earlier version of this pipeline: intv/reop/read
# are event-detail files, not parallel patient rosters -- most patients
# never appear in them. "CASEID in all 4 files" is a small, complication-
# selected cohort, produced as a bonus sheet but never the training set.
#   "main_only"           -> (DEFAULT) every case in main (+ bmi/postop for
#                             2017) with a valid 30-day outcome.
#   "strict_intersection" -> only cases present in every file for that year.
LINK_MODE = "main_only"

# ---------------------------------------------------------------------------
# 3. OUTCOME VARIABLES + how each year supplies them
# ---------------------------------------------------------------------------
# Canonical column names AFTER 01_data_loader.py has standardized everything.
OUTCOME_VARS = {
    "bmi_30d": {
        "column": "BMI_CLOSEST30D",
        "unit_column": None,          # BMI is unitless (kg/m^2)
        "label": "BMI at 30-day follow-up (kg/m^2)",
    },
    "weight_30d": {
        "column": "WGT_CLOSEST30D",
        "unit_column": "WGTUNIT_CLOSEST30D",   # values are "lbs" or "kg"
        "label": "Weight at 30-day follow-up (kg)",
        "standard_unit": "kg",
    },
}
# A row is only usable for outcome modeling if BOTH are present (non-null).
#
# How each year actually supplies these two columns (confirmed by directly
# inspecting the real files):
#   2017 -- NOT in main. 01_data_loader.py joins 2017_MBSAQIP_bmifinal.txt
#           onto main by CASEID and renames its WGT_CLOSEST/WGTUNIT_CLOSEST/
#           BMI columns to WGT_CLOSEST30D/WGTUNIT_CLOSEST30D/BMI_CLOSEST30D.
#           bmifinal's DTBMI column (days from surgery to that measurement;
#           confirmed range 0-30, median 13 in the real file) is kept as
#           DTBMI_30D so you can see how close to day 30 each value actually
#           is -- it is NOT a fixed 30-day window the way 2020/2024 are.
#   2020 -- already in main as WGT_CLOSEST30D / WGTUNIT_CLOSEST30D /
#           BMI_CLOSEST30D. No join needed.
#   2024 -- already in main, but under human-readable labels ("Post-Op
#           Weight Closest to Day 30", etc.) -- renamed by column_mapping.py.

# ---------------------------------------------------------------------------
# 4. VARIABLE RATIONALE
# ---------------------------------------------------------------------------
# Unchanged from every earlier version of this pipeline -- same 4 domains,
# same variables, same weights, same clinical thresholds. What DID change
# with real 2017/2020/2024 data in hand (all handled in column_mapping.py,
# not here, so this section stays identical across dataset generations):
#   - race_PUF/hispanic/HTN_MEDS (2017) and CHRONIC_STEROIDS (2017 + 2020,
#     NOT just 2017) all get aliased to the canonical names below.
#   - 2024 ships human-readable column labels for everything -- aliased.
#   - HISPANIC does not exist anywhere in the 2024 file (not renamed --
#     genuinely absent). Those rows get NaN for HISPANIC after the 3 years
#     are merged; scoring_engine.py already treats a missing value as
#     "doesn't contribute this variable for this row", the same mechanism
#     used for MOBILITY_DEVICE/PRIORITY below.
#   - Three category-string spellings differ in 2017 (DIABETES, ASACLASS,
#     FUNSTATPRESURG) -- normalized in column_mapping.py so they match the
#     category_scores keys below instead of silently going unscored.
FLAG_INELIGIBLE_VARIABLES = {"AGE", "HGT", "RACE", "SEX", "HISPANIC", "OPYEAR", "SURGSPECIALTY_BAR"}
FLAG_THRESHOLD_COUNT = 3
FLAG_WEIGHT_STEP = 2.0

VARIABLE_RATIONALE = {
    "Adiposity": {
        "domain_weight_prior": 25.0,
        "variables": {
            "BMI_HIGH_BAR": {
                "rank": 1, "weight": 30, "column": "BMI_HIGH_BAR", "unit_column": None,
                "source_file": "main", "var_type": "numeric", "direction": "high_is_bad",
                "ideal": 21.7, "tolerance": 5, "severity_span": 25,
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
            "BMI": {
                "rank": 2, "weight": 25, "column": "BMI", "unit_column": None,
                "source_file": "main", "var_type": "numeric", "direction": "two_sided",
                "ideal": 21.7, "tolerance": 5, "severity_span": 25,
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
            "WGT_HIGH_BAR": {
                "rank": 3, "weight": 20, "column": "WGT_HIGH_BAR", "unit_column": "WGT_HIGH_UNIT_BAR",
                "source_file": "main", "var_type": "numeric", "direction": "high_is_bad", "unit_kind": "weight",
                "ideal": "devine_ibw", "tolerance": 5, "severity_span": 40,
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
            "WGT_CLOSEST": {
                "rank": 4, "weight": 10, "column": "WGT_CLOSEST", "unit_column": "WGTUNIT_CLOSEST",
                "source_file": "main", "var_type": "numeric", "direction": "high_is_bad", "unit_kind": "weight",
                "ideal": "devine_ibw", "tolerance": 5, "severity_span": 40,
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
            "MOBILITY_DEVICE": {
                "rank": 5, "weight": 8, "column": None,  # not in 2020/2024 "new registry"; held out for all 3 years for comparability
                "unit_column": None, "source_file": "main", "var_type": "binary",
                "direction": "high_is_bad", "ideal": None, "tolerance": None, "severity_span": None,
                "category_scores": {"Yes": 100, "No": 0},
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
            "VENOUS_STASIS": {
                "rank": 6, "weight": 4, "column": "VENOUS_STASIS", "unit_column": None,
                "source_file": "main", "var_type": "binary", "direction": "high_is_bad",
                "ideal": None, "tolerance": None, "severity_span": None,
                "category_scores": {"Yes": 100, "No": 0},
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
            "HGT": {
                "rank": 7, "weight": 3, "column": "HGT", "unit_column": "HGTUNIT", "unit_kind": "height",
                "source_file": "main", "var_type": "numeric", "direction": "two_sided",
                "ideal": None, "tolerance": None, "severity_span": None,
                "contributes_to_severity_score": False, "flag_direction_logic": "increase",
            },
        },
    },
    "Metabolic": {
        "domain_weight_prior": 25.0,
        "variables": {
            "DIABETES": {
                "rank": 1, "weight": 25, "column": "DIABETES", "unit_column": None,
                "source_file": "main", "var_type": "ordinal", "direction": "high_is_bad",
                "ideal": None, "tolerance": None, "severity_span": None,
                "category_scores": {"No": 0, "Yes, non-insulin": 60, "Yes, insulin": 100},
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
            "HTN_MEDS": {
                "rank": 2, "weight": 18, "column": "NBHTN_MEDS", "unit_column": None,
                "source_file": "main", "var_type": "numeric", "direction": "high_is_bad",
                "ideal": 0, "tolerance": 0, "severity_span": 3,
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
            "HYPERLIPIDEMIA": {
                "rank": 3, "weight": 15, "column": "HYPERLIPIDEMIA", "unit_column": None,
                "source_file": "main", "var_type": "binary", "direction": "high_is_bad",
                "ideal": None, "tolerance": None, "severity_span": None,
                "category_scores": {"Yes": 100, "No": 0},
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
            "SLEEP_APNEA": {
                "rank": 4, "weight": 12, "column": "SLEEP_APNEA", "unit_column": None,
                "source_file": "main", "var_type": "binary", "direction": "high_is_bad",
                "ideal": None, "tolerance": None, "severity_span": None,
                "category_scores": {"Yes": 100, "No": 0},
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
            "RENAL_INSUFFICIENCY": {
                "rank": 5, "weight": 10, "column": "RENAL_INSUFFICIENCY", "unit_column": None,
                "source_file": "main", "var_type": "binary", "direction": "high_is_bad",
                "ideal": None, "tolerance": None, "severity_span": None,
                "category_scores": {"Yes": 100, "No": 0},
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
            "GERD": {
                "rank": 6, "weight": 7, "column": "GERD", "unit_column": None,
                "source_file": "main", "var_type": "binary", "direction": "high_is_bad",
                "ideal": None, "tolerance": None, "severity_span": None,
                "category_scores": {"Yes": 100, "No": 0},
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
            "ALBUMIN": {
                "rank": 7, "weight": 6, "column": "ALBUMIN", "unit_column": None,
                "source_file": "main", "var_type": "numeric", "direction": "low_is_bad",
                "ideal": 4.25, "tolerance": 0.5, "severity_span": 2.0,
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
            "CREATININE": {
                "rank": 8, "weight": 4, "column": "CREATININE", "unit_column": None,
                "source_file": "main", "var_type": "numeric", "direction": "high_is_bad",
                "ideal": 0.85, "tolerance": 0.2, "severity_span": 2.0,
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
            "DIALYSIS": {
                "rank": 9, "weight": 3, "column": "DIALYSIS", "unit_column": None,
                "source_file": "main", "var_type": "binary", "direction": "high_is_bad",
                "ideal": None, "tolerance": None, "severity_span": None,
                "category_scores": {"Yes": 100, "No": 0},
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
        },
    },
    "Behavioral": {
        "domain_weight_prior": 25.0,
        "variables": {
            "FUNSTATPRESURG": {
                "rank": 1, "weight": 30, "column": "FUNSTATPRESURG", "unit_column": None,
                "source_file": "main", "var_type": "ordinal", "direction": "high_is_bad",
                "ideal": None, "tolerance": None, "severity_span": None,
                "category_scores": {
                    "Independent": 0, "Partially dependent": 60,
                    "Totally dependent": 100, "Unknown": None,
                },
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
            "SMOKER": {
                "rank": 2, "weight": 25, "column": "SMOKER", "unit_column": None,
                "source_file": "main", "var_type": "binary", "direction": "high_is_bad",
                "ideal": None, "tolerance": None, "severity_span": None,
                "category_scores": {"Yes": 100, "No": 0},
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
            "MOBILITY_DEVICE": {
                "rank": 3, "weight": 15, "column": None,
                "unit_column": None, "source_file": "main", "var_type": "binary",
                "direction": "high_is_bad", "ideal": None, "tolerance": None, "severity_span": None,
                "category_scores": {"Yes": 100, "No": 0},
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
            "CHRONIC_STEROIDS": {
                "rank": 4, "weight": 10, "column": "IMMUNOSUPR_THER", "unit_column": None,
                "source_file": "main", "var_type": "binary", "direction": "high_is_bad",
                "ideal": None, "tolerance": None, "severity_span": None,
                "category_scores": {"Yes": 100, "No": 0},
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
            "PREVIOUS_SURGERY": {
                "rank": 4, "weight": 10, "column": "PREVIOUS_SURGERY", "unit_column": None,
                "source_file": "main", "var_type": "binary", "direction": "high_is_bad",
                "ideal": None, "tolerance": None, "severity_span": None,
                "category_scores": {"Yes": 100, "No": 0},
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
            "PRIORITY": {
                "rank": 4, "weight": 10, "column": None,
                "unit_column": None, "source_file": "main", "var_type": "binary",
                "direction": "high_is_bad", "ideal": None, "tolerance": None, "severity_span": None,
                "category_scores": {"Yes": 100, "No": 0},
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
        },
    },
    "Socio-Environmental": {
        "domain_weight_prior": 25.0,
        "variables": {
            "AGE": {
                "rank": 1, "weight": 20, "column": "AGE", "unit_column": None,
                "source_file": "main", "var_type": "numeric", "direction": "high_is_bad",
                "ideal": 13, "tolerance": 0, "severity_span": 67,
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
            "ASACLASS": {
                "rank": 1, "weight": 20, "column": "ASACLASS", "unit_column": None,
                "source_file": "main", "var_type": "ordinal", "direction": "high_is_bad",
                "ideal": None, "tolerance": None, "severity_span": None,
                "category_scores": {
                    "ASA I - Normal/Healthy": 0, "ASA II - Mild systemic disease": 25,
                    "ASA III - Severe systemic disease": 60,
                    "ASA IV - Severe systemic disease threat to life": 85,
                    "ASA V - Moribund": 100, "None assigned": None,
                },
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
            "RACE": {
                "rank": 1, "weight": 20, "column": "RACE_PUF", "unit_column": None,
                "source_file": "main", "var_type": "ordinal", "direction": "high_is_bad",
                "ideal": None, "tolerance": None, "severity_span": None, "category_scores": None,
                # Health-equity CONTEXT, not asserted pathology -- not flagged, no hand-set
                # severity number. 2024 recodes/combines some race categories (confirmed:
                # multi-select values like "White~Black or African American" appear only in
                # 2024) -- fine for the model (still a categorical feature), just don't
                # assume category-string continuity across years for this one.
                "contributes_to_severity_score": False, "flag_direction_logic": "increase",
            },
            "SEX": {
                "rank": 2, "weight": 15, "column": "SEX", "unit_column": None,
                "source_file": "main", "var_type": "ordinal", "direction": "high_is_bad",
                "ideal": None, "tolerance": None, "severity_span": None, "category_scores": None,
                "contributes_to_severity_score": False, "flag_direction_logic": "increase",
            },
            "HISPANIC": {
                "rank": 3, "weight": 10, "column": "HISPANIC", "unit_column": None,
                "source_file": "main", "var_type": "ordinal", "direction": "high_is_bad",
                "ideal": None, "tolerance": None, "severity_span": None, "category_scores": None,
                # Genuinely absent from the 2024 file (confirmed by header search, not a
                # rename) -- 2024 patients get NaN here after the merge, handled the same
                # way as MOBILITY_DEVICE/PRIORITY.
                "contributes_to_severity_score": False, "flag_direction_logic": "increase",
            },
            "OPYEAR": {
                "rank": 3, "weight": 10, "column": "OPYEAR", "unit_column": None,
                "source_file": "main", "var_type": "numeric", "direction": "two_sided",
                "ideal": None, "tolerance": None, "severity_span": None,
                "contributes_to_severity_score": False, "flag_direction_logic": "decrease",
            },
            "SURGSPECIALTY_BAR": {
                "rank": 4, "weight": 5, "column": "SURGSPECIALTY_BAR", "unit_column": None,
                "source_file": "main", "var_type": "ordinal", "direction": "high_is_bad",
                "ideal": None, "tolerance": None, "severity_span": None, "category_scores": None,
                "contributes_to_severity_score": False, "flag_direction_logic": "increase",
            },
        },
    },
}

# ---------------------------------------------------------------------------
# 5. DOMAIN-LEVEL WEIGHTS
# ---------------------------------------------------------------------------
DOMAIN_WEIGHTS_PRIOR = {
    "Adiposity": 25.0,
    "Metabolic": 25.0,
    "Behavioral": 25.0,
    "Socio-Environmental": 25.0,
}

# ---------------------------------------------------------------------------
# 6. UNIT STANDARDIZATION
# ---------------------------------------------------------------------------
STANDARD_WEIGHT_UNIT = "kg"
STANDARD_HEIGHT_UNIT = "cm"
LBS_TO_KG = 0.45359237
IN_TO_CM = 2.54

# ---------------------------------------------------------------------------
# 7. OPTUNA WEIGHT-LEARNING SETTINGS
# ---------------------------------------------------------------------------
OPTUNA_N_TRIALS = 50
OPTUNA_SAMPLER = "TPE"
LEARNED_WEIGHTS_PATH = os.path.join(MODEL_DIR, "learned_weights.json")

# ---------------------------------------------------------------------------
# 8. PERMANENT TRAIN / VALIDATION / TEST SPLIT
# ---------------------------------------------------------------------------
# Decided ONCE (04_split_comparison.py) and reused by every script downstream
# (Optuna, the MLP, baselines) so no script ever trains on, or tunes against,
# data another script tested on. Stratified by OPYEAR so all 3 years are
# proportionally represented in train/val/test.
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
SPLIT_INDICES_PATH = os.path.join(MODEL_DIR, "train_val_test_indices.json")

# ---------------------------------------------------------------------------
# 9. MODEL / SHAP / PHENOGRAPH / BASELINE SETTINGS
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
# Fallback MLP settings -- used by 05_train_model.py ONLY if the Optuna MLP
# study (Step 4) hasn't been run yet. Otherwise models/best_mlp_hyperparams.json wins.
MLP_DEFAULT_PARAMS = {
    "n_layers": 2, "n_units": 64, "activation": "relu", "dropout": 0.1, "batch_norm": True,
    "learning_rate": 1e-3, "batch_size": 256, "optimizer": "adamw", "weight_decay": 1e-4,
}

# Set to an integer (e.g. 5000) to run the whole pipeline on a random subset
# for fast iteration/testing. Set to None for the full 3-year cohort.
SAMPLE_SIZE = None

# >>> PUT A CASE ID HERE to get that single patient's SHAP explanation <<<
TARGET_CASEID = None

SHAP_DOMAIN_PLOT_TARGET = "BMI"  # "BMI" or "WEIGHT" -- which target the 4 per-domain SHAP plots explain
SHAP_TOTAL_FEATURES = 26          # the 26 rationale variables with a real column (one-hot SHAP summed back per variable)
SHAP_TOP_N_BEESWARM = 12

# PhenoGraph / UMAP -- now run ONCE on the pooled 3-year cohort (not per
# year). Two outputs: one total-population clustering, one per-domain
# (4-panel) clustering, both over all 3 years together.
PHENOGRAPH_K = 30
PHENOGRAPH_JITTER_STD = 0.02      # tiny noise added to the UMAP LAYOUT only, so identical patients don't sit on one pixel
PHENOGRAPH_MIN_UNIQUE_PROFILES = 200   # below this many distinct patient profiles, a domain is grouped by exact profile instead
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "euclidean"
UMAP_RANDOM_STATE = RANDOM_SEED
CLUSTER_STABILITY_N_RUNS = 3      # PhenoGraph re-runs on random subsamples for ARI/NMI stability (~30 s each per analysis)

# Baseline models (08_baseline_models.py)
BASELINE_RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]

# Derived "early responder" classification, so the MLP can ALSO be scored with
# AUC-ROC / F1 / Precision / Recall (those are classification metrics; the
# model itself predicts continuous BMI/weight). A patient is a responder if
# their ACTUAL 30-day BMI is at least this fraction below pre-op BMI; the
# model's score for that patient is its PREDICTED fractional BMI reduction.
# 5% splits this cohort roughly in half (mean 30-day drop ~2.4 kg/m^2 on ~44).
CLINICAL_RESPONSE_BMI_REDUCTION_FRACTION = 0.05

# ---------------------------------------------------------------------------
# 10. STEP-SPECIFIC SETTINGS (added with the full 02-08 build)
# ---------------------------------------------------------------------------
EXCEL_PREVIEW_ROWS = 5000      # rows in "*_PREVIEW" sheets of per-patient Excel outputs

# --- 04_split_comparison.py: besides the permanent random 70/15/15 split, it
# also runs a TEMPORAL check (train on the older years, test on this one) to
# show how well a model generalizes to a year it has never seen.
TEMPORAL_HOLDOUT_YEAR = "2024"

# --- 04_optuna_weight_learning.py: TWO Optuna studies ---
#  (A) phenotype-score weights (domain + within-domain variable weights),
#      OPTUNA_N_TRIALS trials (Section 7), scored by how well the resulting
#      4 domain scores predict 30-day % BMI change on the VALIDATION set.
#  (B) MLP hyperparameters for predicting 30-day BMI AND weight (one network,
#      two outputs), scored by validation RMSE. Tuned on a random subsample of
#      the training set to keep it tractable on a laptop; the FINAL model in
#      Step 5 is trained on the full training set with the winning settings.
OPTUNA_MLP_N_TRIALS = 30
OPTUNA_MLP_TUNE_TRAIN_ROWS = 60000     # None = use the whole training set (slow)
OPTUNA_MLP_TUNE_VAL_ROWS = 20000
OPTUNA_MLP_MAX_EPOCHS = 25
OPTUNA_MLP_TIMEOUT_SECONDS = 3600      # stop the MLP study after this long even if trials remain
MLP_SEARCH_SPACE = {
    "n_layers": [1, 2, 3],
    "n_units": [32, 64, 128, 256],
    "activation": ["relu", "leaky_relu", "gelu", "tanh"],
    "dropout": (0.0, 0.4),
    "batch_norm": [True, False],
    "learning_rate": (1e-4, 1e-2),     # log-uniform
    "batch_size": [128, 256, 512, 1024],
    "optimizer": ["adam", "adamw", "sgd"],
    "weight_decay": (1e-6, 1e-2),      # log-uniform
}
BEST_MLP_PARAMS_PATH = os.path.join(MODEL_DIR, "best_mlp_hyperparams.json")

# --- 05_train_model.py (final MLP) ---
MLP_MAX_EPOCHS = 150
MLP_EARLY_STOPPING_PATIENCE = 12        # epochs without validation improvement before stopping
MLP_MODEL_PATH = os.path.join(MODEL_DIR, "mlp_bmi_weight_30d.pt")
MLP_PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "mlp_preprocessor.json")
MLP_METRICS_PATH = os.path.join(MODEL_DIR, "mlp_metrics.json")

# --- 06_shap_explain.py ---
SHAP_N_EXPLAIN = 3000         # test-set patients explained (stratified by year)
SHAP_N_BACKGROUND = 300       # training-set background sample for GradientExplainer

# --- 07_phenograph_clustering.py (pooled 3-year) ---
# PhenoGraph + UMAP on 500k+ patients is hours of compute on a laptop, so both
# are fit on a year-stratified sample and every other patient is then
# assigned to its nearest cluster (k-NN in the same scaled feature space).
PHENOGRAPH_SAMPLE_PER_YEAR = 20000
PHENOGRAPH_MIN_CLUSTER_FRACTION = 0.005   # clusters smaller than 0.5% of the sample -> "outlier" (-1)
CLUSTER_STABILITY_SUBSAMPLE = 0.8         # each stability re-run clusters a random 80% of the sample
SILHOUETTE_SAMPLE_SIZE = 10000

# Full per-patient Excel exports (Steps 1, 3, 7) take ~1-3 minutes each at
# 500k+ patients. Set False while iterating to write just the first
# EXCEL_PREVIEW_ROWS rows per year instead (parquet/csv outputs are unaffected).
WRITE_FULL_PATIENT_EXCEL = True

# ---------------------------------------------------------------------------
# QUICK MODE -- `python run_pipeline.py --quick` sets MBSAQIP_QUICK=1 so the
# whole pipeline runs end-to-end in minutes as a smoke test. Never use its
# numbers for results.
# ---------------------------------------------------------------------------
if os.environ.get("MBSAQIP_QUICK") == "1":
    WRITE_FULL_PATIENT_EXCEL = False
    OPTUNA_MLP_TUNE_TRAIN_ROWS = 20000
    OPTUNA_MLP_TUNE_VAL_ROWS = 5000
    OPTUNA_MLP_MAX_EPOCHS = 8
    SHAP_N_EXPLAIN = 600
    PHENOGRAPH_SAMPLE_PER_YEAR = 3000
    CLUSTER_STABILITY_N_RUNS = 2
