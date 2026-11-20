"""
09_xgboost_models_v2.py

XGBoost regression experiments for:
    1. BMI_30D
    2. WEIGHT_30D_KG

Uses the existing feature pipeline and permanent data split.
Selects the number of boosting rounds using validation RMSE.
The locked test set is reserved for final evaluation.
"""

import os
import json

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from xgboost import XGBRegressor

import config
from model_utils import build_feature_matrix


def export_prediction_table(y_actual, y_predicted, target_name, output_path):
    import pandas as pd
    import numpy as np

    actual = np.asarray(y_actual).reshape(-1)
    predicted = np.asarray(y_predicted).reshape(-1)

    # Include every test prediction in the CSV.
    table = pd.DataFrame({
        "X_Actual": actual,
        "Y_Predicted": predicted,
        "Error": predicted - actual
    })

    table.to_csv(output_path, index=False)

    # Print only 25 rows so the terminal output stays manageable.
    print("\n" + "=" * 65)
    print(target_name + " — FIRST 25 TEST PREDICTIONS")
    print("=" * 65)
    print(table.head(25).round(3).to_string(index=False))
    print("\nTotal test predictions:", len(table))
    print("Full table saved to:", output_path)
    print("Actual value range:",
          round(float(actual.min()), 3), "to",
          round(float(actual.max()), 3))
    print("Predicted value range:",
          round(float(predicted.min()), 3), "to",
          round(float(predicted.max()), 3))

    return table



def calculate_metrics(y_true, predictions):
    """Calculate standard regression metrics."""

    return {
        "RMSE": float(
            np.sqrt(
                mean_squared_error(y_true, predictions)
            )
        ),
        "MAE": float(
            mean_absolute_error(y_true, predictions)
        ),
        "R2": float(
            r2_score(y_true, predictions)
        ),
        "n": int(len(y_true))
    }


def print_metrics(label, metrics):
    """Print regression metrics."""

    print(
        f"{label:<30} "
        f"RMSE={metrics['RMSE']:.3f}  "
        f"MAE={metrics['MAE']:.3f}  "
        f"R²={metrics['R2']:.3f}"
    )


def prepare_data():
    """Load data and reproduce the baseline's preprocessing."""

    # 1. Load the same scored dataset.
    input_path = os.path.join(
        config.OUTPUT_DIR,
        "04_final_scored_v2.parquet"
    )
    df = pd.read_parquet(input_path)
    print("Loaded rows:", len(df))

    # 2. Build the same feature matrix.
    X_initial, feature_to_variable, _, _ = build_feature_matrix(df)
    feature_columns = list(X_initial.columns)
    X = X_initial.copy()

    # Restore original missing values for severity features.
    for col in feature_columns:
        if col.endswith("__SEVERITY"):
            variable_name = feature_to_variable[col]
            source_col = f"{variable_name}__SEVERITY"

            if source_col in df.columns:
                X[col] = df[source_col]

    # 3. Load the existing permanent split.
    split_path = os.path.join(
        config.MODEL_DIR,
        "train_val_test_indices_v2.json"
    )
    with open(split_path, "r") as f:
        split_data = json.load(f)

    train_idx = np.array(split_data["train_indices"], dtype=int)
    val_idx = np.array(split_data["validation_indices"], dtype=int)
    test_idx = np.array(split_data["test_indices"], dtype=int)

    # 4. Check that the split covers every row without overlap.
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(df)
    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0

    # 5. Recalculate height and operation-year normalization
    #    using training rows only.
    for variable_name in ["HGT", "OPYEAR"]:
        source_col = f"{variable_name}__STD"
        norm_col = f"{variable_name}__NORM"

        train_values = pd.to_numeric(
            df.loc[train_idx, source_col],
            errors="coerce"
        )

        train_mean = train_values.mean()
        train_std = train_values.std()

        if pd.isna(train_std) or train_std == 0:
            X[norm_col] = 0.0
        else:
            X[norm_col] = (
                pd.to_numeric(df[source_col], errors="coerce")
                - train_mean
            ) / train_std

    # 6. Calculate severity-feature medians from training rows only.
    for col in feature_columns:
        if col.endswith("__SEVERITY"):
            train_values = X.loc[train_idx, col]
            median_value = (
                train_values.median()
                if train_values.notna().any()
                else 0.0
            )
            X[col] = X[col].fillna(median_value)

    # Fill any remaining missing feature values.
    X = X.fillna(0.0)

    # 7. Fit scaling on training rows only.
    scaler = StandardScaler()
    scaler.fit(X.loc[train_idx])

    X_scaled = pd.DataFrame(
        scaler.transform(X),
        index=X.index,
        columns=X.columns
    )

    # 8. Load both prediction targets.
    y_bmi = pd.to_numeric(
        df["OUTCOME_BMI_30D"], errors="coerce"
    )
    y_weight = pd.to_numeric(
        df["OUTCOME_WEIGHT_30D_KG"], errors="coerce"
    )

    print("Features:", X_scaled.shape[1])
    print(
        f"Split: {len(train_idx):,} train / "
        f"{len(val_idx):,} validation / {len(test_idx):,} test"
    )

    return (
        X_scaled, y_bmi, y_weight,
        train_idx, val_idx, test_idx
    )


def run_target(target_name, y, X_scaled, train_idx, val_idx, test_idx):
    """Train XGBoost with validation-based early stopping."""

    # Keep only rows with an available target value.
    valid_mask = y.notna()

    train_rows = [
        idx for idx in train_idx if valid_mask.loc[idx]
    ]
    val_rows = [
        idx for idx in val_idx if valid_mask.loc[idx]
    ]
    test_rows = [
        idx for idx in test_idx if valid_mask.loc[idx]
    ]

    # Build separate datasets for each split.
    X_train = X_scaled.loc[train_rows]
    y_train = y.loc[train_rows]

    X_val = X_scaled.loc[val_rows]
    y_val = y.loc[val_rows]

    X_test = X_scaled.loc[test_rows]
    y_test = y.loc[test_rows]

    print(f"\n{'=' * 60}")
    print(f"Target: {target_name}")
    print(f"Rows: {len(train_rows):,} train / "
          f"{len(val_rows):,} validation / "
          f"{len(test_rows):,} test")

    # Define the first XGBoost configuration.
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="rmse",
        early_stopping_rounds=50,
        tree_method="hist",
        n_jobs=4,
        random_state=42
    )

    # Fit on training data; monitor validation data.
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Early stopping chooses the best boosting iteration.
    best_iteration = int(model.best_iteration)
    best_val_rmse = float(model.best_score)

    print(f"Best iteration: {best_iteration}")
    print(f"Best validation RMSE: {best_val_rmse:.3f}")

    # Evaluate the chosen model on validation and test.
    val_predictions = model.predict(X_val)
    test_predictions = model.predict(X_test)

    val_metrics = calculate_metrics(y_val, val_predictions)
    test_metrics = calculate_metrics(y_test, test_predictions)

    print_metrics("XGBoost validation", val_metrics)
    print_metrics("XGBoost FINAL TEST", test_metrics)

    # Export predictions in permanent test-split order.
    filename = (
        "09_xgboost_test_predictions_BMI_v2.csv"
        if target_name == "BMI_30D"
        else "09_xgboost_test_predictions_WEIGHT_v2.csv"
    )

    prediction_path = os.path.join(
        config.OUTPUT_DIR,
        filename
    )

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    export_prediction_table(
        y_actual=y_test,
        y_predicted=test_predictions,
        target_name=target_name,
        output_path=prediction_path
    )

    return {
        "target": target_name,
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "test_rows": len(test_rows),
        "best_iteration": best_iteration,
        "best_validation_rmse": best_val_rmse,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "parameters": model.get_params()
    }



if __name__ == "__main__":
    (
        X_scaled,
        y_bmi,
        y_weight,
        train_idx,
        val_idx,
        test_idx
    ) = prepare_data()

    # Train a separate model for each outcome.
    bmi_results = run_target(
        "BMI_30D",
        y_bmi,
        X_scaled,
        train_idx,
        val_idx,
        test_idx
    )

    weight_results = run_target(
        "WEIGHT_30D_KG",
        y_weight,
        X_scaled,
        train_idx,
        val_idx,
        test_idx
    )

    # Save results separately from the existing baseline.
    output_path = os.path.join(
        config.MODEL_DIR,
        "xgboost_metrics_v2.json"
    )

    results = {
        "BMI_30D": bmi_results,
        "WEIGHT_30D_KG": weight_results
    }

    with open(output_path, "w") as file:
        json.dump(results, file, indent=4)

    print(f"\nSaved XGBoost metrics to: {output_path}")

