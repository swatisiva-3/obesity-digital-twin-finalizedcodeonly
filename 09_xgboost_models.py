"""
Stage 09: XGBoost for the finalized pooled MBSAQIP pipeline.

Uses the existing permanent patient-key split, training-only preprocessing,
and the same absolute 30-day outcomes as the baseline models.

Run after the finalized pipeline has produced its scored dataset and split.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import config
import model_utils as mu


def main():
    split_path = Path(config.SPLIT_INDICES_PATH)

    output_dir = Path(config.OUTPUT_DIR)
    model_dir = Path(config.MODEL_DIR)

    metrics_path = model_dir / "xgboost_metrics_pooled.json"
    output_paths = [
        output_dir / f"09_xgboost_test_predictions_{target}_pooled.csv"
        for target in mu.TARGET_NAMES
    ]

    for output_path in [metrics_path, *output_paths]:
        if output_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing result: {output_path}"
            )

    if not split_path.is_file():
        raise RuntimeError(
            "Permanent split is missing. Run the existing pipeline first."
        )

    df, source = mu.load_scored_with_fallback()

    with split_path.open() as f:
        split = json.load(f)

    eligible = df.loc[df["HAS_VALID_OUTCOME"], "PATIENT_KEY"].astype(str)

    mu.validate_split(split, eligible)

    expected = set(eligible)
    groups = [split["patient_keys"][k] for k in
              ("train", "validation", "test")]
    actual = set().union(*(set(g) for g in groups))

    if actual != expected or sum(map(len, groups)) != len(expected):
        raise RuntimeError("Permanent split does not exactly match cohort.")

    if eligible.duplicated().any():
        raise RuntimeError("Duplicate eligible patient keys.")

    parts = mu.split_frames(df, split)

    X_train, _, _ = mu.build_feature_matrix(parts["train"])
    prep = mu.Preprocessor().fit(X_train)

    X = {
        name: prep.transform_X(mu.build_feature_matrix(frame)[0])
        for name, frame in parts.items()
    }

    metrics = {
        "input": source,
        "split_path": str(split_path),
        "models": {}
    }
    predictions = {}

    for target in mu.TARGET_NAMES:
        spec = mu.TARGETS[target]

        y = {
            name: pd.to_numeric(
                frame[spec["outcome"]], errors="coerce"
            ).to_numpy(dtype=float)
            for name, frame in parts.items()
        }

        for name, values in y.items():
            if not np.isfinite(values).all():
                raise RuntimeError(
                    f"{target}: invalid {name} target values."
                )

        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=2000,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            early_stopping_rounds=50,
            eval_metric="rmse",
            random_state=42,
            n_jobs=4,
        )

        model.fit(
            X["train"], y["train"],
            eval_set=[(X["validation"], y["validation"])],
            verbose=False,
        )

        val_pred = model.predict(X["validation"])
        test_pred = model.predict(X["test"])

        metrics["models"][target] = {
            "best_iteration": int(model.best_iteration),
            "validation": mu.regression_metrics(
                y["validation"], val_pred
            ),
            "test": mu.regression_metrics(
                y["test"], test_pred
            ),
        }

        predictions[target] = pd.DataFrame({
            "PATIENT_KEY": parts["test"]["PATIENT_KEY"].astype(str),
            "X_Actual": y["test"],
            "Y_Predicted": test_pred,
            "Error": y["test"] - test_pred,
        })

        print(
            f"{target}: "
            f"validation RMSE={metrics['models'][target]['validation']['RMSE']:.4f}, "
            f"test RMSE={metrics['models'][target]['test']['RMSE']:.4f}"
        )

    # Write results only after both models have finished successfully.
    output_dir = Path(config.OUTPUT_DIR)
    model_dir = Path(config.MODEL_DIR)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = model_dir / "xgboost_metrics_pooled.json"

    # Refuse to overwrite previous research results.
    output_paths = [
        output_dir / f"09_xgboost_test_predictions_{target}_pooled.csv"
        for target in mu.TARGET_NAMES
    ]

    for path in [metrics_path, *output_paths]:
        if path.exists():
            raise FileExistsError(
                f"Output already exists: {path}"
            )

    for target, path in zip(mu.TARGET_NAMES, output_paths):
        predictions[target].to_csv(path, index=False)

    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2, allow_nan=False)

    print("Stage 09 completed.")
    print("Metrics:", metrics_path)


if __name__ == "__main__":
    main()
