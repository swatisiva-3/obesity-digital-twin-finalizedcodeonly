# Obesity Digital Twin: Merged 3-Year MBSAQIP Pipeline (2017 / 2020 / 2024)

This pipeline predicts **BMI and weight 30 days after bariatric surgery**. It runs on the real MBSAQIP PUF files for 2017, 2020 and 2024, merged into **one pooled cohort of 546,731 patients**. It scores every patient on the 4-domain phenotype framework (Adiposity, Metabolic, Behavioral, Socio-Environmental). Optuna tunes both the phenotype weights and the neural network. The model is explained with SHAP, and the pooled 3-year population is clustered with PhenoGraph/UMAP. Every important step writes an Excel file and figures.

**This download already contains the results of a full run on your real data**, in `outputs/` and `models/`. Start with `outputs/RESULTS_SUMMARY.md`. The raw data files are not included, since you already have them; see "Putting the data in place" below.

---

## 1. Folder layout

```
MBSAQIP_PUF_CODE_Updated Sep 24/
├── .gitignore
├── requirements.txt
├── README.md                      <- this file
├── config.py                      <- the ONE file you edit (paths, variables, weights, settings)
├── column_mapping.py              <- the C++ merge program's logic, ported + extended (renames, value fixes)
├── utils.py                       <- shared helpers (logging, units, Excel/figure writers)
├── scoring_engine.py              <- phenotype scoring math (used by Steps 3 and 4)
├── model_utils.py                 <- features, permanent split, metrics, PyTorch MLP (used by Steps 4-8)
├── 00_setup_environment.py
├── 01_data_loader.py
├── 02_preprocessing.py
├── 03_scoring.py
├── 04_optuna_weight_learning.py
├── 04_split_comparison.py
├── 05_train_model.py
├── 06_shap_explain.py
├── 07_phenograph_clustering.py
├── 08_baseline_models.py
├── run_pipeline.py                <- runs 01 -> 08 in order
├── data/
│   └── MBSAQIP DATA FINAL/
│       ├── MBSAQIP_PUF_2017_TXT/   2017_MBSAQIP_{main,bmi,intv,read,reop}final.txt
│       ├── MBSAQIP_PUF_2020_TXT/   2020_MBSAQIP_{main,intv,read,reop}final.txt
│       └── MBSAQIP_PUF_2024_TXT/   2024_MBSAQIP_{main,intv,read,reop}final.txt
├── outputs/                       <- created by the scripts
│   ├── excel/                     <- every Excel output
│   ├── figures/                   <- every plot (PNG, plus 300-dpi PDFs for SHAP)
│   └── *.parquet                  <- the hand-off files between steps
└── models/                        <- split, learned weights, Optuna results, trained MLP
```

**Putting the data in place:** extract your three Google Drive zips (`MBSAQIP_PUF_2017_TXT-...zip`, `..._2020_TXT-...zip`, `..._2024_TXT-...zip`) into `data/MBSAQIP DATA FINAL/`. Each zip already contains a correctly named year folder, so extracting all three there produces exactly the layout above. The file names are referenced in `config.YEAR_FOLDERS` / `config.YEAR_FILES`.

## 2. Running it (Windows)

```bat
cd "C:\Users\swati\Downloads\MBSAQIP_PUF_CODE_Updated Sep 24"
python 00_setup_environment.py          :: once: creates venv\ and installs requirements.txt
venv\Scripts\activate
python run_pipeline.py --quick          :: ~10 min smoke test of every step (numbers not for reporting)
python run_pipeline.py                  :: the real run, ~1 hour on a laptop
```

You can also run any script on its own, in order, e.g. `python 05_train_model.py`. Each script checks that its input exists and tells you which earlier step to run if it doesn't. You can resume a run with `python run_pipeline.py --from 05`. To get one patient's SHAP explanation, run `python run_pipeline.py --steps 06 --patient 2024_<CASEID>`.

Approximate full-run times on a 2-core laptop:

| Step | Time |
|---|---|
| 01 | 4 min |
| 02 | 1 min |
| 03 | 2 min |
| 04 | 15 min (50 weight trials + 30 MLP trials) |
| 04b | under 1 min |
| 05 | 5–15 min |
| 06 | 3 min |
| 07 | 20 min |
| 08 | under 1 min |

The full-data Excel copies in Steps 1, 3 and 7 take 1–3 minutes each. Set `config.WRITE_FULL_PATIENT_EXCEL = False` to write only the first 5,000 rows per year while you're iterating.

## 3. Step inputs and outputs

Each step reads only what the step before it wrote.

| Step | Reads | Writes (hand-off to later steps) | Excel / figures |
|---|---|---|---|
| **01_data_loader** | the 13 raw `.txt` files | `01_predictors_all_variables.parquet` (all variables) and `01_outcomes_bmi_weight_30day.parquet` (30-day BMI + weight). **These two files are what training, validation and testing use.** | `01_merged_all_variables.xlsx`: every variable, all 3 years, one sheet per year. `01_outcomes_bmi_weight_30day.xlsx`: the outcomes, as a separate file. `01_load_summary.xlsx`: per-year counts, column completeness, and every rename or value fix applied. |
| **02_preprocessing** | both 01 files, joined on `PATIENT_KEY` | `02_preprocessed.parquet` | `02_flag_summary.xlsx`, `02_preprocessed_PREVIEW.xlsx`, `02_flag_rates_by_year.png` |
| **03_scoring** | 02 | `03_scored.parquet`, `models/clinical_baseline_weights.json` | `03_phenotype_scores.xlsx` (weights used; domain and total scores per patient and by year), `03_domain_scores_by_year.png` |
| **04_optuna_weight_learning** | 03 + split | `04_final_scored.parquet`, `models/learned_weights.json`, `models/best_mlp_hyperparams.json`, `models/train_val_test_indices.json` | `04_optuna_results.xlsx` (every trial, best settings, hyperparameter importance), Optuna history and importance plots |
| **04_split_comparison** | 04 + split | (checks the split; creates it if missing) | `04_split_comparison.xlsx` (sizes by year, balance/SMD, random vs temporal vs leave-one-year-out), 2 plots |
| **05_train_model** | 04 + split + best hyperparameters | `models/mlp_bmi_weight_30d.pt`, `models/mlp_preprocessor.json`, `models/mlp_metrics.json`, `05_test_predictions.parquet` | `05_model_report.xlsx` (see Section 5), learning curve, actual vs predicted, residuals, ROC, trajectory plots |
| **06_shap_explain** | model + preprocessor + 04 + split | none | `06_shap_importance.xlsx`, `06_shap_global_bar_all_variables_{BMI,WEIGHT}.pdf`, `06_shap_beeswarm_top12_{BMI,WEIGHT}.pdf` (300 dpi), domain plots |
| **07_phenograph_clustering** | 04 (falls back to 03) | `07_cluster_assignments.parquet` | `07_phenograph_clusters.xlsx`, `07_umap_total_3yr.png`, `07_umap_domainwise_3yr.png`, and more |
| **08_baseline_models** | 04 + split + `mlp_metrics.json` | `models/baseline_metrics.json` | `08_baseline_vs_mlp.xlsx`, `08_baseline_vs_mlp.png` |

`PATIENT_KEY` = `OPYEAR_CASEID` (for example `2024_1234567`). CASEIDs are only unique within a year, so this is the join key everywhere.

## 4. Architecture decisions

* **One merged 3-year dataset.** This follows your `config.py`, the C++ merge program, and "one total 3-year population clustering". There is one Optuna run, one MLP, one SHAP analysis and one pooled PhenoGraph, and `OPYEAR` is kept as a column (and a model feature). Everything is still reported **by year** where it matters: test performance, residuals, trajectories, SHAP importance, and cluster composition.
* **The C++ merge program is ported into Python** (`column_mapping.py` + `01_data_loader.py`). It keeps the master schema from the newest year (2024, 185 columns), the alias table (`race_PUF→RACE_PUF`, `hispanic→HISPANIC`, `HTN_MEDS→NBHTN_MEDS`), the `"3+" → "3 or more"` rule, and the non-empty CASEID/OPYEAR requirement. It was extended with everything below.
* **The Excel-to-TSV converter isn't needed.** Your files are already tab-delimited `.txt`, and the loader reads them directly.
* **The permanent 70/15/15 split** (`models/train_val_test_indices.json`) is created once and stratified by year. Every model uses it. Imputation and scaling are fit on training patients only. Optuna uses only the validation set, and the test set is evaluated once per model, for reporting.
* **The MLP predicts the change from pre-op, not the 30-day value directly.** The reported prediction is pre-op value + predicted change = predicted 30-day BMI/weight. This lets the report show accuracy on the 30-day values you asked for *and* on the change itself. The change is the honest number: just copying pre-op BMI forward already gives R² ≈ 0.87 on 30-day BMI.
* **FOLLOWUP_DAY is a model input.** The "30-day" measurement isn't taken on day 30: the median is day 14, with a range of 0–30, in every year. The model needs to know *when* the value was measured, and this input is also what makes the day-by-day trajectory possible.
* **Numeric variables enter the model as real values** (kg, cm, kg/m², g/dL), not as 0–100 severity scores. The severity score caps pre-op BMI at 46.7, which would throw away the information the model needs most. Categorical clinical variables use their severity codes (No=0 … insulin=100). RACE, SEX, HISPANIC and SURGSPECIALTY_BAR are one-hot encoded, and their SHAP values are summed back to one value per variable.

## 5. Where each thing you asked for lives

| You asked for | Where |
|---|---|
| RMSE / MAE / R² | `05_model_report.xlsx` → `1_performance` (train/val/test), `1b_test_by_year` |
| Hidden layers, neurons/layer, activation, dropout, batch normalization | `05_model_report.xlsx` → `2_architecture` |
| Learning rate, batch size, optimizer, weight decay, epochs / early stopping | `05_model_report.xlsx` → `3_training_params` |
| Optuna: number of trials, best validation RMSE, best combination, hyperparameter importance | `05_model_report.xlsx` → `4_optuna`; full detail in `04_optuna_results.xlsx`; `04_optuna_mlp_param_importance.png` |
| Train vs validation vs test, test-set RMSE/MAE/R² | `05_model_report.xlsx` → `5_generalization` |
| Actual vs predicted BMI | `05_actual_vs_predicted_BMI.png` (and `_WEIGHT`), `test_predictions` sheet |
| Residuals, all 3 years in one graph | `05_residuals_all_years.png` |
| AUC-ROC, F1, Precision, Recall | `05_model_report.xlsx` → `6_classification`, `05_roc_curve_responder.png` (see Section 7) |
| BMI and weight trajectory, trendline and slope equation | `05_trajectory_change.png`, `05_trajectory_absolute.png`; `7_trajectory_slopes` (equations) and `7b_trajectory_by_day` sheets |
| SHAP bar of all variables + top-12 beeswarm, 300-dpi PDF | `06_shap_global_bar_all_variables_*.pdf`, `06_shap_beeswarm_top12_*.pdf` |
| 2 PhenoGraph/UMAP analyses: total 3-year and each domain | `07_umap_total_3yr.png`, `07_umap_domainwise_3yr.png` |
| Cluster stability, ARI/NMI across runs | `07_phenograph_clusters.xlsx` → `stability_ARI_NMI`, `07_cluster_stability.png` |
| Cluster size distribution | `cluster_sizes` sheet, `07_cluster_size_distribution.png` |
| Silhouette score | `quality_metrics` sheet (also modularity, Davies-Bouldin, Calinski-Harabasz) |
| UMAP n_neighbors / min_dist / metric / random seed | `umap_settings` sheet; also in the figure titles; set in `config.py` |
| Feature preprocessing/scaling | `preprocessing` sheet (clustering), `3_training_params` (MLP), `model_utils.py` docstring |
| Excel at every important section | `outputs/excel/01_…` through `08_…` |
| Merged file with all variables + separate 30-day BMI/weight file, used for training/validation/testing | `01_merged_all_variables.xlsx` + `01_outcomes_bmi_weight_30day.xlsx`. Their parquet twins feed Step 2 onward. |
| Baseline comparison (your `08_baseline_models.py`) | `08_baseline_models.py`: same logic (mean predictor, Ridge with validation-chosen alpha, test touched once, train-only imputation/scaling), plus a carry-forward baseline and the MLP side by side |

## 6. What inspecting the real files turned up (all handled in code)

1. **2024 uses human-readable column headers** ("Case Identification Number", "Year of Operation", …), not the short PUF codes. They are mapped in `column_mapping.HEADER_ALIASES`.
2. **`CHRONIC_STEROIDS` is still called that in 2017 *and* 2020.** Only 2024's field corresponds to `IMMUNOSUPR_THER`. Both older years are aliased.
3. **2017 spells three categories differently**: DIABETES ("Insulin"), ASACLASS ("3-Severe Disturb"), and FUNSTATPRESURG ("Partially Dependent"). Without a fix these would silently go unscored, so they're normalized to the 2020/2024 wording. SURGSPECIALTY_BAR capitalization and "Non-binary (prior to 2025)" are harmonized too.
4. **2017's outcome file (`bmifinal`) has one row per follow-up visit** (up to 14 per patient). Its `BMI`/`WGT_CLOSEST` columns just repeat the pre-op values. The real follow-up values are `BMI_DISCH`/`WGT_DISCH`, and the loader takes the visit closest to day 30.
5. **`NBHTN_MEDS` is text-capped at "3 or more"** in every year, so it's treated as 3. In 2020/2024 it's blank for patients without hypertension, while 2017 records 0. That's left as missing, not guessed.
6. **2024 dropped the separate Hispanic question.** "Hispanic or Latino" became a race option, but only 0.9% select it, compared with ~13% answering "Yes" to the separate question in 2017/2020. So 2024 HISPANIC is "Yes" when selected and **"Not collected (2024)"** otherwise, never "No". RACE multi-selections become "Multiracial".
7. **Outcome cleaning:** physiologically implausible values (BMI outside 15–120, weight outside 30–350 kg, or a BMI change of more than 30% within 30 days; about 1,850 patients) are set to missing and logged in `02_flag_summary.xlsx`.

## 7. Things to know when you interpret results

* **AUC-ROC / F1 / Precision / Recall** are classification metrics, and this model predicts continuous values. So a clinically meaningful yes/no label is derived: an **early responder** has an actual follow-up BMI at least 5% below pre-op (`config.CLINICAL_RESPONSE_BMI_REDUCTION_FRACTION`). The model's score is its predicted % BMI reduction. That 5% threshold splits this cohort roughly in half, and you can change it in `config.py`.
* **SHAP and correlated features.** Pre-op BMI, pre-op weight, highest BMI and highest weight are strongly correlated, so SHAP can show them pushing in *opposite* directions. For example, a higher pre-op BMI predicts more loss, while a higher *historical peak* at the same current BMI (meaning weight has already come down) predicts less. Read those four together, as the Adiposity domain plot does, rather than one at a time.
* **Identical patients are clustered as one profile.** Many patients have identical profiles in some domains. For example, most patients have every Behavioral variable = 0, and every pre-op BMI above 46.7 has the same Adiposity severity. Graph clustering would chop a pile of identical patients into arbitrary pieces, so PhenoGraph runs on each domain's distinct profiles, and every patient inherits their profile's cluster. A domain with fewer than 200 distinct profiles (in practice, Behavioral) is grouped by exact profile instead, and the figure title and the `method` column in `quality_metrics` say which method was used.
* **PhenoGraph is fit on a 60,000-patient year-balanced sample** (20,000 per year), and every other patient is assigned to their nearest cluster. PhenoGraph finds many fine-grained clusters at k = 30. Raise `config.PHENOGRAPH_K` for fewer, broader clusters.
* **The Optuna MLP search tunes on a 60,000-patient subsample** of the training set so it finishes in minutes. The final Step 5 model is trained on all ~354,000 training patients.

## 8. Changing things

Everything is in `config.py`:

* **Variables, weights and thresholds:** `VARIABLE_RATIONALE`.
* **File names:** `YEAR_FOLDERS` / `YEAR_FILES`.
* **Search size:** `OPTUNA_N_TRIALS`, `OPTUNA_MLP_N_TRIALS`.
* **MLP search space:** `MLP_SEARCH_SPACE`.
* **UMAP/PhenoGraph settings:** `UMAP_*`, `PHENOGRAPH_*`.
* **Responder threshold:** `CLINICAL_RESPONSE_BMI_REDUCTION_FRACTION`.
* **Split:** delete `models/train_val_test_indices.json` to re-split. Leave it alone otherwise, so every model keeps using the same patients.
