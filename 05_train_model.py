"""
05_train_model.py
==================
STEP 5 -- train the final MLP that predicts 30-day BMI AND 30-day weight,
evaluate it on train / validation / test, draw the BMI and weight
trajectories, and write the full model report.

INPUT:  outputs/04_final_scored.parquet       (Step 4; falls back to 03_scored.parquet)
        models/train_val_test_indices.json     (permanent split)
        models/best_mlp_hyperparams.json       (Step 4 study B; falls back to config.MLP_DEFAULT_PARAMS)
OUTPUT: models/mlp_bmi_weight_30d.pt           -> Step 6 (SHAP)
        models/mlp_preprocessor.json            -> Step 6 (train-only imputation/scaling values)
        models/mlp_metrics.json                 -> Step 8 (baseline comparison)
        outputs/05_test_predictions.parquet     -> Step 8
        outputs/excel/05_model_report.xlsx       (every number you asked for, one sheet per category:
            1_performance         RMSE / MAE / R^2 -- train, validation, test (30-day BMI + weight,
                                  and the change from pre-op)
            1b_test_by_year       test-set performance for 2017 / 2020 / 2024 separately
            2_architecture        hidden layers, neurons/layer, activation, dropout, batch norm, params
            3_training_params     learning rate, batch size, optimizer, weight decay, epochs, early stopping
            4_optuna              number of trials, best validation RMSE, best combination, importance
            5_generalization      train vs validation vs test side by side (+ gaps)
            6_classification      AUC-ROC, F1, Precision, Recall (derived early-responder label)
            7_trajectory_slopes   trendline equations (intercept, slope/day, slope/30 days, R^2)
            7b_trajectory_by_day  day-by-day actual vs model-predicted means per year
            training_history      train/validation loss every epoch
            test_predictions      actual vs predicted for every test patient)
        outputs/figures/05_learning_curve.png
        outputs/figures/05_actual_vs_predicted_BMI.png   (+ _WEIGHT)
        outputs/figures/05_residuals_all_years.png       (all 3 years in one graph)
        outputs/figures/05_roc_curve_responder.png
        outputs/figures/05_trajectory_change.png          (BMI + weight change, trendlines + equations)
        outputs/figures/05_trajectory_absolute.png        (mean BMI + weight over days 0-30)
"""
import json
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import model_utils as mu
import utils

YEAR_COLORS = {"2017": "#1f77b4", "2020": "#ff7f0e", "2024": "#2ca02c"}
TRAJ_DAYS = np.arange(0, 31, 1)


def color(yr):
    return YEAR_COLORS.get(str(yr), None)


def perf_rows(split_name, frame, preds):
    rows = []
    for t in mu.TARGET_NAMES:
        spec = mu.TARGETS[t]
        for what, truth, pred in [(f"{t}_30D", frame[spec["outcome"]], preds[f"PRED_{t}_30D"]),
                                  (f"{t}_CHANGE_FROM_PREOP", frame[spec["change"]], preds[f"PRED_{t}_CHANGE"])]:
            m = mu.regression_metrics(truth, pred)
            rows.append({"split": split_name, "target": what,
                         "unit": "kg/m^2" if t == "BMI" else "kg", **m})
    return rows


def linear_fit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3 or np.ptp(x) == 0:
        return np.nan, np.nan, np.nan, len(x)
    b, a = np.polyfit(x, y, 1)
    yhat = a + b * x
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum((y - yhat) ** 2) / ss_tot if ss_tot > 0 else np.nan
    return a, b, r2, len(x)


def trajectories(model, prep, test):
    """Actual (by measured follow-up day) and model-predicted (every day 0-30) trajectories, per year."""
    slope_rows, day_rows = [], []
    X, _, _ = mu.build_feature_matrix(test)
    for yr in sorted(test["OPYEAR"].unique()):
        sub_mask = (test["OPYEAR"] == yr).values
        sub, Xs = test[sub_mask], X[sub_mask]
        preds_by_day = {}
        for d in TRAJ_DAYS:
            Xd = Xs.copy()
            Xd["FOLLOWUP_DAY"] = float(d)
            preds_by_day[d] = mu.predict_outcomes(model, prep, Xd, sub)
        for t in mu.TARGET_NAMES:
            spec = mu.TARGETS[t]
            # actual: each patient's measured change at their own follow-up day
            a, b, r2, n = linear_fit(sub["FOLLOWUP_DAY"], sub[spec["change"]])
            slope_rows.append({"OPYEAR": yr, "target": t, "source": "actual (measured follow-up)",
                               "y": f"{t} change from pre-op", "intercept": a, "slope_per_day": b,
                               "slope_per_30_days": b * 30 if np.isfinite(b) else np.nan, "R2": r2, "n": n,
                               "equation": f"y = {a:.3f} {'+' if b >= 0 else '-'} {abs(b):.4f}*day"})
            # predicted: same patients, model evaluated at every day 0-30
            pred_change = np.array([preds_by_day[d][f"PRED_{t}_CHANGE"].mean() for d in TRAJ_DAYS])
            pred_abs = np.array([preds_by_day[d][f"PRED_{t}_30D"].mean() for d in TRAJ_DAYS])
            a2, b2, r22, _ = linear_fit(TRAJ_DAYS, pred_change)
            slope_rows.append({"OPYEAR": yr, "target": t, "source": "MLP-predicted (test patients, days 0-30)",
                               "y": f"{t} change from pre-op", "intercept": a2, "slope_per_day": b2,
                               "slope_per_30_days": b2 * 30, "R2": r22, "n": int(sub_mask.sum()),
                               "equation": f"y = {a2:.3f} {'+' if b2 >= 0 else '-'} {abs(b2):.4f}*day"})
            a3, b3, r23, _ = linear_fit(TRAJ_DAYS, pred_abs)
            slope_rows.append({"OPYEAR": yr, "target": t, "source": "MLP-predicted (test patients, days 0-30)",
                               "y": f"mean {t} ({'kg/m^2' if t == 'BMI' else 'kg'})", "intercept": a3,
                               "slope_per_day": b3, "slope_per_30_days": b3 * 30, "R2": r23, "n": int(sub_mask.sum()),
                               "equation": f"y = {a3:.2f} {'+' if b3 >= 0 else '-'} {abs(b3):.4f}*day"})
            actual_by_day = sub.groupby(sub["FOLLOWUP_DAY"].round())[[spec["change"], spec["outcome"]]].agg(["mean", "count"])
            for i, d in enumerate(TRAJ_DAYS):
                row = {"OPYEAR": yr, "target": t, "day": int(d),
                       "predicted_mean_change": pred_change[i], "predicted_mean_value": pred_abs[i]}
                if d in actual_by_day.index:
                    row.update({"actual_mean_change": actual_by_day.loc[d, (spec["change"], "mean")],
                                "actual_mean_value": actual_by_day.loc[d, (spec["outcome"], "mean")],
                                "actual_n_measured_that_day": int(actual_by_day.loc[d, (spec["change"], "count")])})
                day_rows.append(row)
    return pd.DataFrame(slope_rows), pd.DataFrame(day_rows)


def plot_trajectories(slopes, by_day, test):
    units = {"BMI": "kg/m^2", "WEIGHT": "kg"}
    # change
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, t in zip(axes, mu.TARGET_NAMES):
        for yr in sorted(by_day["OPYEAR"].unique()):
            d = by_day[(by_day["OPYEAR"] == yr) & (by_day["target"] == t)]
            ok = d["actual_n_measured_that_day"].fillna(0) >= 30
            ax.scatter(d.loc[ok, "day"], d.loc[ok, "actual_mean_change"], s=18, color=color(yr), alpha=0.7)
            sa = slopes[(slopes.OPYEAR == yr) & (slopes.target == t) & slopes.source.str.startswith("actual")].iloc[0]
            sp = slopes[(slopes.OPYEAR == yr) & (slopes.target == t) & slopes.source.str.startswith("MLP")
                        & slopes.y.str.contains("change")].iloc[0]
            ax.plot(TRAJ_DAYS, sa.intercept + sa.slope_per_day * TRAJ_DAYS, color=color(yr), lw=2,
                    label=f"{yr} actual trend: {sa.equation}  (R²={sa.R2:.2f})")
            ax.plot(TRAJ_DAYS, d["predicted_mean_change"], color=color(yr), lw=2, ls="--",
                    label=f"{yr} MLP-predicted: {sp.equation}")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xlabel("days after surgery")
        ax.set_ylabel(f"change from pre-op {'BMI' if t == 'BMI' else 'weight'} ({units[t]})")
        ax.set_title(f"{t} trajectory, days 0-30 (dots = actual mean on that day)")
        ax.legend(fontsize=7.5, loc="lower left")
    fig.suptitle("30-day BMI and weight trajectories by year: trendlines and slope equations")
    fig.tight_layout()
    utils.save_fig(fig, "05_trajectory_change")

    # absolute
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, t in zip(axes, mu.TARGET_NAMES):
        spec = mu.TARGETS[t]
        for yr in sorted(by_day["OPYEAR"].unique()):
            d = by_day[(by_day["OPYEAR"] == yr) & (by_day["target"] == t)]
            s3 = slopes[(slopes.OPYEAR == yr) & (slopes.target == t) & slopes.y.str.startswith("mean")].iloc[0]
            base = test.loc[test["OPYEAR"] == yr, spec["baseline"]].mean()
            ax.scatter([0], [base], marker="s", s=50, color=color(yr), edgecolor="k", zorder=5)
            ax.plot(TRAJ_DAYS, d["predicted_mean_value"], color=color(yr), lw=2,
                    label=f"{yr}: {s3.equation}  (pre-op mean {base:.1f})")
        ax.set_xlabel("days after surgery")
        ax.set_ylabel(f"mean {'BMI' if t == 'BMI' else 'weight'} ({units[t]})")
        ax.set_title(f"MLP-predicted mean {t} over the first 30 days (squares = actual pre-op mean)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    utils.save_fig(fig, "05_trajectory_absolute")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-epochs", type=int, default=config.MLP_MAX_EPOCHS)
    args = ap.parse_args()
    utils.log("=" * 70)
    utils.log("STEP 5: FINAL MLP -- 30-day BMI + weight")
    utils.log("=" * 70)
    df, source = mu.load_scored_with_fallback()
    df["OPYEAR"] = df["OPYEAR"].astype(str)
    utils.log(f"Input: {source}")
    split = mu.get_or_create_split(df)
    parts = mu.split_frames(df, split)
    del df
    train, val, test = parts["train"], parts["validation"], parts["test"]
    params, params_source = mu.load_best_mlp_params()
    utils.log(f"Hyperparameters ({params_source}): {params}")

    Xtr, f2v, f2d = mu.build_feature_matrix(train)
    prep = mu.Preprocessor().fit(Xtr, mu.change_targets(train))
    Xva, _, _ = mu.build_feature_matrix(val)
    t0 = time.time()
    model, history, best_epoch = mu.train_mlp(
        prep.transform_X(Xtr).values, prep.transform_Y(mu.change_targets(train)).values,
        prep.transform_X(Xva).values, prep.transform_Y(mu.change_targets(val)).values,
        params, max_epochs=args.max_epochs, patience=config.MLP_EARLY_STOPPING_PATIENCE, verbose=True)
    train_seconds = time.time() - t0
    utils.log(f"Trained {len(history)} epochs in {train_seconds:.0f}s; best epoch {best_epoch}")

    import torch
    utils.ensure_dir(config.MODEL_DIR)
    torch.save({"state_dict": model.state_dict(), "params": params, "n_in": Xtr.shape[1],
                "n_out": len(mu.TARGET_NAMES)}, config.MLP_MODEL_PATH)
    utils.save_json({**prep.to_dict(), "feature_to_variable": f2v, "feature_to_domain": f2d}, config.MLP_PREPROCESSOR_PATH)

    # ---- predictions + metrics ----
    preds = {}
    perf = []
    for name, frame in parts.items():
        X, _, _ = mu.build_feature_matrix(frame)
        preds[name] = mu.predict_outcomes(model, prep, X, frame)
        perf += perf_rows(name, frame, preds[name])
    perf = pd.DataFrame(perf)
    naive = []
    for t in mu.TARGET_NAMES:
        spec = mu.TARGETS[t]
        m = mu.regression_metrics(test[spec["outcome"]], test[spec["baseline"]])
        naive.append({"split": "test", "target": f"{t}_30D", "model": "naive carry-forward (30-day = pre-op)", **m})
    utils.log("Performance:\n" + perf.round(4).to_string(index=False))

    by_year = []
    for yr in sorted(test["OPYEAR"].unique()):
        mk = (test["OPYEAR"] == yr).values
        for r in perf_rows("test", test[mk], preds["test"][mk]):
            by_year.append({"OPYEAR": yr, **r})
    by_year = pd.DataFrame(by_year)

    gen = perf.pivot_table(index="target", columns="split", values=["RMSE", "MAE", "R2"])
    gen.columns = [f"{m}_{s}" for m, s in gen.columns]
    gen = gen.reset_index()
    gen["RMSE_gap_test_minus_train"] = gen["RMSE_test"] - gen["RMSE_train"]
    gen["R2_gap_train_minus_test"] = gen["R2_train"] - gen["R2_test"]
    gen["verdict"] = np.where(gen["RMSE_gap_test_minus_train"].abs() / gen["RMSE_train"] < 0.05,
                              "generalizes (test within 5% of train RMSE)", "check for over/under-fitting")

    # ---- classification (derived early-responder label) ----
    cls_rows, roc_data = [], {}
    for name in ("validation", "test"):
        frame, p = parts[name], preds[name]
        res, y, score = mu.responder_metrics(frame["BASELINE_BMI"], frame["OUTCOME_BMI_30D"], p["PRED_BMI_30D"])
        cls_rows.append({"split": name, "OPYEAR": "ALL", **res})
        if name == "test":
            roc_data["ALL"] = (y, score)
            for yr in sorted(frame["OPYEAR"].unique()):
                mk = (frame["OPYEAR"] == yr).values
                r2, y2, s2 = mu.responder_metrics(frame.loc[mk, "BASELINE_BMI"], frame.loc[mk, "OUTCOME_BMI_30D"],
                                                  p.loc[mk, "PRED_BMI_30D"])
                cls_rows.append({"split": name, "OPYEAR": yr, **r2})
                roc_data[yr] = (y2, s2)
    cls = pd.DataFrame(cls_rows)
    utils.log("Early-responder classification:\n" + cls[["split", "OPYEAR", "AUC_ROC", "F1", "Precision", "Recall"]].round(4).to_string(index=False))

    # ---- architecture / training / optuna sheets ----
    arch = pd.DataFrame([
        {"item": "Model type", "value": "Feed-forward multilayer perceptron (PyTorch), 2 outputs"},
        {"item": "Input features", "value": Xtr.shape[1]},
        {"item": "Input variables", "value": f"{len(set(f2v.values()))} ({len(set(f2v.values())) - 1} rationale variables + FOLLOWUP_DAY)"},
        {"item": "Hidden layers", "value": int(params["n_layers"])},
        {"item": "Neurons per hidden layer", "value": int(params["n_units"])},
        {"item": "Activation function", "value": params["activation"]},
        {"item": "Dropout rate", "value": round(float(params["dropout"]), 4)},
        {"item": "Batch normalization", "value": bool(params["batch_norm"])},
        {"item": "Output layer", "value": "Linear, 2 units: standardized 30-day BMI change, standardized 30-day weight change"},
        {"item": "Prediction", "value": "pre-op value + predicted change = predicted 30-day BMI / weight"},
        {"item": "Trainable parameters", "value": mu.count_parameters(model)},
        {"item": "Layer-by-layer", "value": " -> ".join(str(m) for m in model)},
    ])
    train_params = pd.DataFrame([
        {"item": "Learning rate", "value": float(params["learning_rate"])},
        {"item": "Batch size", "value": int(params["batch_size"])},
        {"item": "Optimizer", "value": params["optimizer"] + (" (momentum 0.9, Nesterov)" if params["optimizer"] == "sgd" else "")},
        {"item": "Weight decay (L2)", "value": float(params["weight_decay"])},
        {"item": "Loss", "value": "mean squared error on standardized targets"},
        {"item": "Gradient clipping", "value": "global norm 5.0"},
        {"item": "Max epochs", "value": args.max_epochs},
        {"item": "Early stopping", "value": f"patience {config.MLP_EARLY_STOPPING_PATIENCE} epochs on validation MSE; best weights restored"},
        {"item": "Epochs trained", "value": len(history)},
        {"item": "Best epoch", "value": best_epoch},
        {"item": "Training time (s)", "value": round(train_seconds, 1)},
        {"item": "Training rows", "value": len(train)},
        {"item": "Validation rows (early stopping)", "value": len(val)},
        {"item": "Test rows (touched once, for reporting)", "value": len(test)},
        {"item": "Imputation / scaling", "value": "median imputation + z-score, both fit on TRAINING rows only"},
        {"item": "Random seed", "value": config.RANDOM_SEED},
        {"item": "Hyperparameter source", "value": params_source},
    ])
    if os.path.exists(config.BEST_MLP_PARAMS_PATH):
        b = utils.load_json(config.BEST_MLP_PARAMS_PATH)
        optuna_sheet = pd.DataFrame(
            [{"item": "Number of trials", "value": b["n_trials"]},
             {"item": "Completed / pruned", "value": f"{b.get('n_complete')} / {b.get('n_pruned')}"},
             {"item": "Best trial", "value": b["best_trial_number"]},
             {"item": "Best validation RMSE, 30-day BMI (kg/m^2)", "value": b["best_validation_rmse_BMI_30D"]},
             {"item": "Best validation RMSE, 30-day weight (kg)", "value": b["best_validation_rmse_WEIGHT_30D"]},
             {"item": "Best validation MSE (standardized, objective)", "value": b["best_validation_mse_standardized"]},
             {"item": "Tuning rows (train / validation)", "value": f"{b['tuning_rows']['train']} / {b['tuning_rows']['validation']}"}]
            + [{"item": f"Best: {k}", "value": v} for k, v in b["best_params"].items()]
            + [{"item": f"Importance: {k}", "value": round(v, 4)} for k, v in b.get("hyperparameter_importance", {}).items()])
    else:
        optuna_sheet = pd.DataFrame([{"item": "Optuna MLP study", "value": "not run -- run 04_optuna_weight_learning.py"}])

    slopes, by_day = trajectories(model, prep, test)
    utils.log("Trajectory slopes:\n" + slopes[["OPYEAR", "target", "source", "y", "equation", "slope_per_30_days", "R2"]].round(4).to_string(index=False))

    test_out = pd.concat([test[["PATIENT_KEY", "CASEID", "OPYEAR", "FOLLOWUP_DAY", "BASELINE_BMI", "OUTCOME_BMI_30D",
                                "BASELINE_WEIGHT_KG", "OUTCOME_WEIGHT_30D_KG"]].reset_index(drop=True),
                          preds["test"].reset_index(drop=True)], axis=1)
    test_out["RESIDUAL_BMI_30D"] = test_out["OUTCOME_BMI_30D"] - test_out["PRED_BMI_30D"]
    test_out["RESIDUAL_WEIGHT_30D"] = test_out["OUTCOME_WEIGHT_30D_KG"] - test_out["PRED_WEIGHT_30D"]
    test_out.to_parquet(os.path.join(config.OUTPUT_DIR, "05_test_predictions.parquet"), index=False)

    utils.save_json({"performance": perf.to_dict("records"), "test_by_year": by_year.to_dict("records"),
                     "naive_carry_forward_test": naive, "classification": cls.to_dict("records"),
                     "params": params, "params_source": params_source, "best_epoch": best_epoch,
                     "epochs_trained": len(history)}, config.MLP_METRICS_PATH)

    utils.save_excel_sheets(utils.excel_path("05_model_report.xlsx"), {
        "1_performance": pd.concat([perf.assign(model="MLP"), pd.DataFrame(naive)], ignore_index=True),
        "1b_test_by_year": by_year, "2_architecture": arch.astype(str), "3_training_params": train_params.astype(str),
        "4_optuna": optuna_sheet.astype(str), "5_generalization": gen, "6_classification": cls,
        "7_trajectory_slopes": slopes, "7b_trajectory_by_day": by_day,
        "training_history": pd.DataFrame(history), "test_predictions": test_out})

    # ---- figures ----
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(h["epoch"], h["train_mse"], label="train")
    ax.plot(h["epoch"], h["val_mse"], label="validation")
    ax.axvline(best_epoch, color="k", ls=":", label=f"best epoch ({best_epoch})")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE (standardized targets)"); ax.legend()
    ax.set_title("MLP learning curve (early stopping on validation)")
    utils.save_fig(fig, "05_learning_curve")

    rng = np.random.RandomState(config.RANDOM_SEED)
    samp = test_out.iloc[rng.choice(len(test_out), min(15000, len(test_out)), replace=False)]
    for t, unit in (("BMI", "kg/m^2"), ("WEIGHT", "kg")):
        spec = mu.TARGETS[t]
        years = sorted(test_out["OPYEAR"].unique())
        fig, axes = plt.subplots(1, len(years), figsize=(5.2 * len(years), 5), sharex=True, sharey=True)
        lo, hi = np.nanpercentile(test_out[spec["outcome"]], [0.5, 99.5])
        for ax, yr in zip(np.atleast_1d(axes), years):
            s = samp[samp["OPYEAR"] == yr]
            r = by_year[(by_year.OPYEAR == yr) & (by_year.target == f"{t}_30D")].iloc[0]
            ax.scatter(s[spec["outcome"]], s[f"PRED_{t}_30D"], s=4, alpha=0.3, color=color(yr))
            ax.plot([lo, hi], [lo, hi], "k--", lw=1)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_title(f"{yr}: RMSE {r.RMSE:.2f}, MAE {r.MAE:.2f}, R² {r.R2:.3f}")
            ax.set_xlabel(f"actual 30-day {'BMI' if t == 'BMI' else 'weight'} ({unit})")
        np.atleast_1d(axes)[0].set_ylabel(f"predicted 30-day {'BMI' if t == 'BMI' else 'weight'} ({unit})")
        fig.suptitle(f"Actual vs predicted 30-day {t} (test set)")
        fig.tight_layout()
        utils.save_fig(fig, f"05_actual_vs_predicted_{t}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for col, (t, unit) in enumerate((("BMI", "kg/m^2"), ("WEIGHT", "kg"))):
        for yr in sorted(test_out["OPYEAR"].unique()):
            s = samp[samp["OPYEAR"] == yr]
            full = test_out[test_out["OPYEAR"] == yr][f"RESIDUAL_{t}_30D"]
            axes[0, col].scatter(s[f"PRED_{t}_30D"], s[f"RESIDUAL_{t}_30D"], s=3, alpha=0.25, color=color(yr), label=yr)
            lim = np.nanpercentile(np.abs(test_out[f"RESIDUAL_{t}_30D"]), 99.5)
            axes[1, col].hist(full.clip(-lim, lim), bins=100, histtype="step", density=True, lw=1.8, color=color(yr),
                              label=f"{yr}: mean {full.mean():+.3f}, SD {full.std():.3f}")
        axes[0, col].axhline(0, color="k", lw=1)
        axes[0, col].set_xlabel(f"predicted 30-day {'BMI' if t == 'BMI' else 'weight'} ({unit})")
        axes[0, col].set_ylabel(f"residual = actual - predicted ({unit})")
        axes[0, col].set_title(f"{t}: residual vs predicted, all 3 years")
        axes[0, col].legend(markerscale=4)
        axes[1, col].axvline(0, color="k", lw=1)
        axes[1, col].set_xlabel(f"residual ({unit})")
        axes[1, col].set_ylabel("density")
        axes[1, col].set_title(f"{t}: residual distribution by year")
        axes[1, col].legend(fontsize=8)
    fig.suptitle("Prediction residuals on the test set -- 2017, 2020 and 2024 in one graph (histogram tails clipped at the 99.5th percentile)")
    fig.tight_layout()
    utils.save_fig(fig, "05_residuals_all_years")

    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for key, (y, s) in roc_data.items():
        fpr, tpr, _ = roc_curve(y, s)
        auc = cls[(cls.split == "test") & (cls.OPYEAR == key)]["AUC_ROC"].iloc[0]
        ax.plot(fpr, tpr, lw=2.5 if key == "ALL" else 1.5, color="k" if key == "ALL" else color(key),
                label=f"{'all years' if key == 'ALL' else key}: AUC {auc:.3f}")
    ax.plot([0, 1], [0, 1], ":", color="grey")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title(f"ROC -- early responder (>= {100*config.CLINICAL_RESPONSE_BMI_REDUCTION_FRACTION:.0f}% BMI drop by follow-up)")
    ax.legend()
    utils.save_fig(fig, "05_roc_curve_responder")

    plot_trajectories(slopes, by_day, test)
    utils.log("Step 5 complete.")


if __name__ == "__main__":
    main()
