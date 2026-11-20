"""
Stage definitions for the obesity digital-twin pipeline.

This module defines stage ordering and known outputs.
It does not execute scripts or modify research artifacts.
"""

STAGES = {
    "00": {
        "script": "00_setup_environment.py",
        "outputs": [],
    },
    "01": {
        "script": "01_data_loader.py",
        "outputs": [
            "outputs/01_raw_extract.parquet",
            "outputs/01_raw_extract.tsv",
            "outputs/01_raw_extract.xlsx",
            "outputs/caseids_in_all_4_files.xlsx",
        ],
    },
    "02": {
        "script": "02_preprocessing.py",
        "outputs": [
            "outputs/02_master_flagged.parquet",
            "outputs/master_flagged_data.csv",
            "outputs/master_flagged_data_PREVIEW.xlsx",
            "outputs/flag_rate_summary.csv",
        ],
    },
    "03": {
        "script": "03_scoring.py",
        "outputs": [
            "outputs/03_scored.parquet",
            "outputs/phenotype_scores.xlsx",
            "models/clinical_baseline_weights.json",
        ],
    },
    "04": {
        "script": "04_optuna_weight_learning_v2.py",
        "outputs": [
            "outputs/04_final_scored_v2.parquet",
            "outputs/optuna_history_v2.png",
            "models/learned_weights_v2.json",
        ],
    },
    "05": {
        "script": "05_train_model_v2.py",
        "outputs": [
            "models/mlp_bmi_30d_v2.pth",
            "models/mlp_weight_30d_v2.pth",
            "models/feature_scaler_v2.joblib",
            "models/feature_metadata_v2.json",
            "models/training_metrics_v2.json",
            "models/train_val_indices_v2.json",
        ],
    },
    "08": {
        "script": "08_baseline_models_v2.py",
        "outputs": [
            "models/model_comparison_metrics_v2.json",
        ],
    },
    "09": {
        "script": "09_xgboost_models_v2.py",
        "outputs": [
            "models/xgboost_metrics_v2.json",
            "outputs/09_xgboost_test_predictions_BMI_v2.csv",
            "outputs/09_xgboost_test_predictions_WEIGHT_v2.csv",
        ],
    },
    "10": {
        "script": "10_paired_model_comparison_v2.py",
        "outputs": [
            "outputs/10_paired_model_comparison_v2.json",
            "outputs/10_paired_test_predictions_BMI_30D_v2.csv",
            "outputs/10_paired_test_predictions_WEIGHT_30D_KG_v2.csv",
        ],
    },
    "audit": {
        "script": "audit_training_leakage_v2.py",
        "outputs": [
            "outputs/training_leakage_audit_v2.json",
            "outputs/training_leakage_audit_v2.txt",
        ],
    },
}

ORDER = list(STAGES)


def select_stages(stage=None, start=None, end=None, all_stages=False):
    """Return selected stage IDs in pipeline order."""

    if all_stages:
        if any(value is not None for value in (stage, start, end)):
            raise ValueError("--all cannot be combined with stage/range selection")
        return ORDER.copy()

    if stage is not None:
        if start is not None or end is not None:
            raise ValueError("--stage cannot be combined with --from/--to")
        if stage not in STAGES:
            raise ValueError(f"Unknown stage: {stage}")
        return [stage]

    if start is None or end is None:
        raise ValueError("Specify --stage, --from and --to, or --all")

    if start not in STAGES or end not in STAGES:
        raise ValueError("Unknown stage in range")

    first = ORDER.index(start)
    last = ORDER.index(end)

    if first > last:
        raise ValueError("Start stage must precede end stage")

    return ORDER[first:last + 1]


# Direct stage dependencies for planning downstream invalidation.
# Audit is treated conservatively as dependent on the modeling results.
DEPENDENCIES = {
    "00": [],
    "01": ["00"],
    "02": ["01"],
    "03": ["02"],
    "04": ["03"],
    "05": ["04"],
    "08": ["04"],
    "09": ["04"],
    "10": ["04", "05", "09"],
    "audit": ["03", "04", "05", "08", "09", "10"],
}


def downstream_stages(selected):
    """Return stages made stale by rerunning selected stages."""
    affected = set(selected)

    changed = True
    while changed:
        changed = False

        for stage, requirements in DEPENDENCIES.items():
            if stage not in affected and any(
                requirement in affected for requirement in requirements
            ):
                affected.add(stage)
                changed = True

    return [stage for stage in ORDER if stage in affected and stage not in selected]
