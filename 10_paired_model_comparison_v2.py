import importlib
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from model_utils import build_feature_matrix

import config


SEED = 42
BOOTSTRAP_REPS = 1000
OUT_DIR = "outputs"
MODEL_DIR = "models"

os.makedirs(OUT_DIR, exist_ok=True)


class BMI_Model(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.05179372697075629),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.05179372697075629),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.network(x)


def metrics(y, pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "MAE": float(mean_absolute_error(y, pred)),
        "R2": float(r2_score(y, pred)),
        "n": int(len(y)),
    }


def predict_mlp(model, X, device):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(X), 2048):
            batch = torch.tensor(
                X[start:start + 2048],
                dtype=torch.float32,
                device=device,
            )
            chunks.append(model(batch).cpu().numpy().reshape(-1))
    return np.concatenate(chunks)


def bootstrap_difference(y, pred_a, pred_b, reps, rng):
    """Paired bootstrap: both models use the same resampled test rows."""
    n = len(y)
    rmse_diffs = np.empty(reps)
    mae_diffs = np.empty(reps)

    for b in range(reps):
        idx = rng.integers(0, n, size=n)
        ya = y[idx]
        pa = pred_a[idx]
        pb = pred_b[idx]

        rmse_a = np.sqrt(np.mean((pa - ya) ** 2))
        rmse_b = np.sqrt(np.mean((pb - ya) ** 2))
        mae_a = np.mean(np.abs(pa - ya))
        mae_b = np.mean(np.abs(pb - ya))

        rmse_diffs[b] = rmse_a - rmse_b
        mae_diffs[b] = mae_a - mae_b

    return {
        "RMSE_difference_A_minus_B": float(
            np.sqrt(np.mean((pred_a - y) ** 2))
            - np.sqrt(np.mean((pred_b - y) ** 2))
        ),
        "RMSE_difference_95CI": [
            float(np.quantile(rmse_diffs, 0.025)),
            float(np.quantile(rmse_diffs, 0.975)),
        ],
        "MAE_difference_A_minus_B": float(
            np.mean(np.abs(pred_a - y))
            - np.mean(np.abs(pred_b - y))
        ),
        "MAE_difference_95CI": [
            float(np.quantile(mae_diffs, 0.025)),
            float(np.quantile(mae_diffs, 0.975)),
        ],
        "bootstrap_replicates": reps,
    }


def main():
    print("Loading shared feature matrix and locked split...")
    df = pd.read_parquet(
        os.path.join(OUT_DIR, "04_final_scored_v2.parquet")
    )
    print("Loaded rows:", len(df))

    X_initial, feature_to_variable, _, _ = build_feature_matrix(df)

    # Restore the exact feature structure expected by the saved MLP.
    with open(os.path.join(MODEL_DIR, "feature_metadata_v2.json")) as f:
        saved_metadata = json.load(f)

    saved_columns = saved_metadata["feature_columns"]

    missing_columns = [c for c in saved_columns if c not in X_initial.columns]
    extra_columns = [c for c in X_initial.columns if c not in saved_columns]

    expected_missing = {
        "Adiposity_MOBILITY_DEVICE__SEVERITY",
        "Behavioral_MOBILITY_DEVICE__SEVERITY",
    }

    # Accept an exact match or the known historical missing columns.
    # Reject any other missing features or unexpected extras.
    if not set(missing_columns).issubset(expected_missing) or extra_columns:
        raise ValueError(
            f"Unexpected feature mismatch: missing={missing_columns}, "
            f"extra={extra_columns}"
        )

    print(
        f"Feature validation passed: "
        f"{len(saved_columns)} expected columns, "
        f"{len(missing_columns)} historically missing."
    )

    # Match the saved MLP feature order exactly.
    X_initial = X_initial.reindex(columns=saved_columns, fill_value=0.0)

    feature_columns = list(X_initial.columns)
    X = X_initial.copy()

    # Restore missing severity values before train-only imputation.
    for col in feature_columns:
        if col.endswith("__SEVERITY"):
            variable_name = saved_metadata["feature_to_variable"][col]
            source_col = f"{variable_name}__SEVERITY"
            if source_col in df.columns:
                X[col] = df[source_col]

    with open(os.path.join(MODEL_DIR, "train_val_test_indices_v2.json")) as f:
        split_data = json.load(f)

    train_idx = np.array(split_data["train_indices"], dtype=int)
    val_idx = np.array(split_data["validation_indices"], dtype=int)
    test_idx = np.array(split_data["test_indices"], dtype=int)

    assert len(train_idx) + len(val_idx) + len(test_idx) == len(df)
    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0

    for variable_name in ["HGT", "OPYEAR"]:
        source_col = f"{variable_name}__STD"
        norm_col = f"{variable_name}__NORM"
        train_values = pd.to_numeric(
            df.loc[train_idx, source_col], errors="coerce"
        )
        train_mean = train_values.mean()
        train_std = train_values.std()

        if pd.isna(train_std) or train_std == 0:
            X[norm_col] = 0.0
        else:
            X[norm_col] = (
                pd.to_numeric(df[source_col], errors="coerce") - train_mean
            ) / train_std

    for col in feature_columns:
        if col.endswith("__SEVERITY"):
            train_values = X.loc[train_idx, col]
            median_value = (
                train_values.median()
                if train_values.notna().any()
                else 0.0
            )
            X[col] = X[col].fillna(median_value)

    X = X.fillna(0.0)

    scaler = StandardScaler()
    scaler.fit(X.loc[train_idx])
    X_scaled = pd.DataFrame(
        scaler.transform(X),
        index=X.index,
        columns=X.columns,
    )

    y_bmi = pd.to_numeric(df["OUTCOME_BMI_30D"], errors="coerce")
    y_weight = pd.to_numeric(df["OUTCOME_WEIGHT_30D_KG"], errors="coerce")

    print("Features:", X_scaled.shape[1])
    print(
        f"Split: {len(train_idx):,} train / "
        f"{len(val_idx):,} validation / {len(test_idx):,} test"
    )

    df = pd.read_parquet(
        os.path.join(OUT_DIR, "04_final_scored_v2.parquet")
    )

    with open(os.path.join(MODEL_DIR, "feature_metadata_v2.json")) as f:
        metadata = json.load(f)

    with open(os.path.join(MODEL_DIR, "training_metrics_v2.json")) as f:
        mlp_saved_metrics = json.load(f)

    if list(X_scaled.columns) != metadata["feature_columns"]:
        raise ValueError("Feature order differs from saved MLP metadata.")

    saved_scaler = joblib.load(
        os.path.join(MODEL_DIR, "feature_scaler_v2.joblib")
    )
    if int(saved_scaler.n_features_in_) != X_scaled.shape[1]:
        raise ValueError("Saved MLP scaler feature count mismatch.")

    # Independently refit a scaler on the reconstructed training features.
    reconstructed_scaler = StandardScaler()
    reconstructed_scaler.fit(X.loc[train_idx])

    # Compare fitted statistics, not a transform/inverse-transform round trip.
    scaler_checks = {
        "mean": np.max(np.abs(
            saved_scaler.mean_ - reconstructed_scaler.mean_
        )),
        "variance": np.max(np.abs(
            saved_scaler.var_ - reconstructed_scaler.var_
        )),
        "scale": np.max(np.abs(
            saved_scaler.scale_ - reconstructed_scaler.scale_
        )),
    }

    for statistic, max_diff in scaler_checks.items():
        print(
            f"Scaler {statistic} max difference: "
            f"{float(max_diff):.12g}"
        )

    scaler_tolerance = 1e-10
    if any(
        not np.isfinite(max_diff) or max_diff > scaler_tolerance
        for max_diff in scaler_checks.values()
    ):
        raise ValueError(
            "Saved scaler does not match independently reconstructed "
            f"training preprocessing: {scaler_checks}"
        )

    device = torch.device("cpu")
    print("Using CPU for reproducible MLP inference.")




    predictions = {}
    targets = {
        "BMI_30D": y_bmi,
        "WEIGHT_30D_KG": y_weight,
    }

    for target, y_series in targets.items():
        valid_test_rows = [
            int(i) for i in test_idx if pd.notna(y_series.loc[i])
        ]
        y_test = y_series.loc[valid_test_rows].to_numpy(dtype=float)
        X_test = X_scaled.loc[valid_test_rows].to_numpy(dtype=np.float32)

        # Load the previously trained MLP; do not train it again.
        mlp_file = (
            "mlp_bmi_30d_v2.pth"
            if target == "BMI_30D"
            else "mlp_weight_30d_v2.pth"
        )
        mlp = BMI_Model(X_scaled.shape[1]).to(device)
        state = torch.load(
            os.path.join(MODEL_DIR, mlp_file),
            map_location=device,
        )
        mlp.load_state_dict(state)
        pred_mlp = predict_mlp(mlp, X_test, device)

        # Select the RF depth on validation RMSE only, then evaluate test.
        train_rows = [int(i) for i in train_idx if pd.notna(y_series.loc[i])]
        val_rows = [int(i) for i in val_idx if pd.notna(y_series.loc[i])]

        X_train = X_scaled.loc[train_rows].to_numpy()
        y_train = y_series.loc[train_rows].to_numpy(dtype=float)
        X_val = X_scaled.loc[val_rows].to_numpy()
        y_val = y_series.loc[val_rows].to_numpy(dtype=float)

        rf_candidates = [
            {"n_estimators": 60, "max_depth": 12, "min_samples_leaf": 5},
            {"n_estimators": 60, "max_depth": 20, "min_samples_leaf": 5},
        ]

        best_rf = None
        best_config = None
        best_val_rmse = float("inf")

        for params in rf_candidates:
            rf = RandomForestRegressor(
                **params,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            )
            rf.fit(X_train, y_train)
            val_pred = rf.predict(X_val)
            val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
            print(f"{target}: RF {params}, validation RMSE={val_rmse:.6f}")

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_rf = rf
                best_config = params

        pred_rf = best_rf.predict(X_test)

        # Verify existing XGBoost rows against the exact test targets before
        # attaching row identifiers to the predictions.
        xgb_path = os.path.join(
            OUT_DIR,
            "09_xgboost_test_predictions_BMI_v2.csv"
            if target == "BMI_30D"
            else "09_xgboost_test_predictions_WEIGHT_v2.csv",
        )
        xgb_df = pd.read_csv(xgb_path)
        actual_column = xgb_df["X_Actual"].to_numpy(dtype=float)

        if len(xgb_df) != len(y_test) or not np.allclose(
            actual_column, y_test, rtol=1e-5, atol=1e-5
        ):
            raise ValueError(
                f"{target}: XGBoost predictions do not align with test rows."
            )

        pred_xgb = xgb_df["Y_Predicted"].to_numpy(dtype=float)

        actual = y_test
        result = pd.DataFrame({
            "Test_Row_Index": valid_test_rows,
            "CASEID": df.loc[valid_test_rows, "CASEID"].to_numpy(),
            "Actual": actual,
            "MLP_Predicted": pred_mlp,
            "Random_Forest_Predicted": pred_rf,
            "XGBoost_Predicted": pred_xgb,
        })

        output_csv = os.path.join(
            OUT_DIR, f"10_paired_test_predictions_{target}_v2.csv"
        )
        result.to_csv(output_csv, index=False)
        print("Saved aligned predictions:", output_csv)

        model_preds = {
            "MLP": pred_mlp,
            "Random_Forest": pred_rf,
            "XGBoost": pred_xgb,
        }
        predictions[target] = {
            "y": actual,
            "models": model_preds,
            "rf_selected_config": best_config,
            "rf_validation_rmse": best_val_rmse,
        }

        print(f"\n{target} test metrics:")
        for model_name, pred in model_preds.items():
            m = metrics(actual, pred)
            print(
                f"  {model_name:14s} RMSE={m['RMSE']:.6f} "
                f"MAE={m['MAE']:.6f} R2={m['R2']:.6f}"
            )

        if target in mlp_saved_metrics.get("TEST_SET", {}):
            saved = mlp_saved_metrics["TEST_SET"][target]
            reproduced = metrics(actual, pred_mlp)
            print(
                "  MLP saved-vs-reproduced differences:",
                {
                    k: round(reproduced[k] - saved[k], 8)
                    for k in ("RMSE", "MAE", "R2")
                },
            )

    # Paired bootstrap comparisons across the same test patients.
    rng = np.random.default_rng(SEED)
    comparisons = {}

    for target, info in predictions.items():
        y = info["y"]
        models = info["models"]
        comparisons[target] = {
            "n_test": int(len(y)),
            "random_forest_selected_config": info["rf_selected_config"],
            "random_forest_validation_rmse": info["rf_validation_rmse"],
            "pairwise_paired_bootstrap": {},
        }

        names = list(models.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                comparisons[target]["pairwise_paired_bootstrap"][
                    f"{a}_minus_{b}"
                ] = bootstrap_difference(
                    y,
                    models[a],
                    models[b],
                    BOOTSTRAP_REPS,
                    rng,
                )

    json_path = os.path.join(OUT_DIR, "10_paired_model_comparison_v2.json")
    with open(json_path, "w") as f:
        json.dump(comparisons, f, indent=2)

    print("\nSaved paired bootstrap results:", json_path)
    print("Interpretation: negative error differences favor the first model.")
    print("A 95% CI entirely below zero supports lower error for the first model.")
    print("A CI crossing zero does not establish a clear difference.")


if __name__ == "__main__":
    main()
