"""
04_optuna_weight_learning.py
=============================
STEP 4 -- two Optuna studies, both scored ONLY on the validation split of
the permanent 70/15/15 split (the test split is never touched here).

INPUT:  outputs/03_scored.parquet                   (Step 3)
        models/clinical_baseline_weights.json        (Step 3 -- seeds study A)
        models/train_val_test_indices.json           (created here on first run if missing;
                                                      04_split_comparison.py reports on it)
OUTPUT: models/learned_weights.json                  -> learned phenotype weights (study A)
        models/best_mlp_hyperparams.json             -> best MLP settings (study B) -> Step 5
        outputs/04_final_scored.parquet              -> domain/total scores re-computed with the
                                                        learned weights -> Steps 5, 6, 7, 8
        outputs/excel/04_optuna_results.xlsx          (every trial of both studies, best
                                                        settings, hyperparameter importance)
        outputs/figures/04_optuna_weight_history.png
        outputs/figures/04_optuna_mlp_history.png
        outputs/figures/04_optuna_mlp_param_importance.png

STUDY A -- PHENOTYPE WEIGHTS. Searches the within-domain variable weights
and the 4 domain weights. Each candidate weight set -> every patient's
4 domain scores + total score (scoring_engine, the same math as Step 3) ->
a Ridge regression from those 5 scores to 30-day % BMI change, fit on the
TRAIN split -> RMSE on the VALIDATION split (minimized). Trial #0 is the
clinical weights from Step 3 (and, if a previous run exists, trial #1 is
last run's learned weights -- continual learning); TPE is free to move away
from both.

STUDY B -- MLP HYPERPARAMETERS for predicting 30-day BMI and weight (one
network, two outputs). Searches layers, neurons per layer, activation,
dropout, batch normalization, learning rate, batch size, optimizer and
weight decay (config.MLP_SEARCH_SPACE), with early stopping and median
pruning. Objective = validation MSE on the standardized targets (both
outcomes weighted equally); every trial also records validation RMSE for
30-day BMI (kg/m^2) and 30-day weight (kg) in real units.

Run: python 04_optuna_weight_learning.py [--weight-trials N] [--mlp-trials N] [--skip-mlp]
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import optuna
from sklearn.linear_model import Ridge
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import model_utils as mu
import scoring_engine
import utils

optuna.logging.set_verbosity(optuna.logging.WARNING)
DOMAINS = list(config.VARIABLE_RATIONALE.keys())


# ---------------------------------------------------------------------------
# Study A -- phenotype weights
# ---------------------------------------------------------------------------
def flatten(variable_weights, domain_weights):
    flat = {f"{d}__{v}": w for d, vw in variable_weights.items() for v, w in vw.items()}
    flat.update({f"DOMAIN__{d}": w for d, w in domain_weights.items()})
    return flat


def unflatten(flat, structure):
    vw = {d: scoring_engine.renormalize_weights({v: flat[f"{d}__{v}"] for v in vs}) for d, vs in structure.items()}
    dw = scoring_engine.renormalize_weights({d: flat[f"DOMAIN__{d}"] for d in structure})
    return vw, dw


def score_features(df, vw, dw):
    ds, total = scoring_engine.full_scoring_pipeline(df, vw, dw)
    return pd.concat([ds, total.rename("TOTAL")], axis=1).fillna(0.0)


def run_weight_study(train, val, n_trials):
    base = utils.load_json(os.path.join(config.MODEL_DIR, "clinical_baseline_weights.json"))
    structure = base["variable_weights"]
    y_tr, y_va = train["BMI_PCT_CHANGE_30D"].values, val["BMI_PCT_CHANGE_30D"].values

    def objective(trial):
        flat = {}
        for d, vs in structure.items():
            for v in vs:
                flat[f"{d}__{v}"] = trial.suggest_float(f"{d}__{v}", 0.0, 100.0)
            flat[f"DOMAIN__{d}"] = trial.suggest_float(f"DOMAIN__{d}", 0.0, 100.0)
        vw, dw = unflatten(flat, structure)
        m = Ridge(alpha=1.0).fit(score_features(train, vw, dw), y_tr)
        pred = m.predict(score_features(val, vw, dw))
        return float(np.sqrt(np.mean((y_va - pred) ** 2)))

    study = optuna.create_study(direction="minimize", study_name="phenotype_weights",
                                sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED))
    study.enqueue_trial(flatten(structure, base["domain_weights"]))
    if os.path.exists(config.LEARNED_WEIGHTS_PATH):
        prev = utils.load_json(config.LEARNED_WEIGHTS_PATH)
        try:
            study.enqueue_trial(flatten(prev["variable_weights"], prev["domain_weights"]))
            utils.log("  warm-starting from the previous run's learned weights (continual learning)")
        except KeyError:
            pass
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    clinical_rmse = study.trials[0].value
    vw, dw = unflatten(study.best_params, structure)
    utils.log(f"  study A: {n_trials} trials in {time.time()-t0:.0f}s | validation RMSE (30-day % BMI change): "
              f"clinical weights {clinical_rmse:.4f} -> best {study.best_value:.4f} (trial {study.best_trial.number})")
    return study, vw, dw, clinical_rmse


# ---------------------------------------------------------------------------
# Study B -- MLP hyperparameters
# ---------------------------------------------------------------------------
def sample_params(trial):
    s = config.MLP_SEARCH_SPACE
    return {
        "n_layers": trial.suggest_categorical("n_layers", s["n_layers"]),
        "n_units": trial.suggest_categorical("n_units", s["n_units"]),
        "activation": trial.suggest_categorical("activation", s["activation"]),
        "dropout": trial.suggest_float("dropout", *s["dropout"]),
        "batch_norm": trial.suggest_categorical("batch_norm", s["batch_norm"]),
        "learning_rate": trial.suggest_float("learning_rate", *s["learning_rate"], log=True),
        "batch_size": trial.suggest_categorical("batch_size", s["batch_size"]),
        "optimizer": trial.suggest_categorical("optimizer", s["optimizer"]),
        "weight_decay": trial.suggest_float("weight_decay", *s["weight_decay"], log=True),
    }


def run_mlp_study(train, val, n_trials):
    rng = np.random.RandomState(config.RANDOM_SEED)
    tr = train if not config.OPTUNA_MLP_TUNE_TRAIN_ROWS or len(train) <= config.OPTUNA_MLP_TUNE_TRAIN_ROWS \
        else train.iloc[rng.choice(len(train), config.OPTUNA_MLP_TUNE_TRAIN_ROWS, replace=False)]
    va = val if not config.OPTUNA_MLP_TUNE_VAL_ROWS or len(val) <= config.OPTUNA_MLP_TUNE_VAL_ROWS \
        else val.iloc[rng.choice(len(val), config.OPTUNA_MLP_TUNE_VAL_ROWS, replace=False)]
    utils.log(f"  study B tunes on {len(tr):,} train / {len(va):,} validation patients "
              f"(config.OPTUNA_MLP_TUNE_*_ROWS); Step 5 retrains the winner on the full training set")

    Xtr, _, _ = mu.build_feature_matrix(tr)
    Xva, _, _ = mu.build_feature_matrix(va)
    prep = mu.Preprocessor().fit(Xtr, mu.change_targets(tr))
    Xtr_s, Xva_s = prep.transform_X(Xtr).values, prep.transform_X(Xva).values
    Ytr_s, Yva_s = prep.transform_Y(mu.change_targets(tr)).values, prep.transform_Y(mu.change_targets(va)).values

    def objective(trial):
        params = sample_params(trial)
        model, hist, best_epoch = mu.train_mlp(Xtr_s, Ytr_s, Xva_s, Yva_s, params,
                                               max_epochs=config.OPTUNA_MLP_MAX_EPOCHS, patience=5, trial=trial)
        pred = prep.inverse_Y(mu.predict(model, Xva_s))
        truth = prep.inverse_Y(Yva_s)
        for j, t in enumerate(mu.TARGET_NAMES):
            trial.set_user_attr(f"val_RMSE_{t}_30D", float(np.sqrt(np.mean((truth[:, j] - pred[:, j]) ** 2))))
        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("n_parameters", mu.count_parameters(model))
        return float(np.mean((pred - truth) ** 2 / prep.y_stds.values ** 2))

    study = optuna.create_study(direction="minimize", study_name="mlp_hyperparameters",
                                sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3))
    study.enqueue_trial(dict(config.MLP_DEFAULT_PARAMS))
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, timeout=config.OPTUNA_MLP_TIMEOUT_SECONDS,
                   show_progress_bar=False, catch=(RuntimeError, ValueError))
    utils.log(f"  study B: {len(study.trials)} trials in {time.time()-t0:.0f}s | best validation MSE (standardized) "
              f"{study.best_value:.4f} | val RMSE BMI {study.best_trial.user_attrs['val_RMSE_BMI_30D']:.3f} kg/m^2, "
              f"weight {study.best_trial.user_attrs['val_RMSE_WEIGHT_30D']:.3f} kg")
    return study, len(tr), len(va)


def importance_table(study, top=None):
    try:
        imp = optuna.importance.get_param_importances(study)
    except Exception as e:  # e.g. too few completed trials
        utils.log(f"  (hyperparameter importance unavailable: {e})")
        return pd.DataFrame(columns=["parameter", "importance"])
    df = pd.DataFrame({"parameter": list(imp.keys()), "importance": list(imp.values())})
    return df.head(top) if top else df


def history_plot(study, name, title, ylabel):
    vals = [t.value for t in study.trials if t.value is not None]
    nums = [t.number for t in study.trials if t.value is not None]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(nums, vals, s=18, alpha=0.6, label="trial")
    ax.plot(nums, np.minimum.accumulate(vals), color="crimson", lw=2, label="best so far")
    ax.set_xlabel("trial")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    utils.save_fig(fig, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight-trials", type=int, default=config.OPTUNA_N_TRIALS)
    ap.add_argument("--mlp-trials", type=int, default=config.OPTUNA_MLP_N_TRIALS)
    ap.add_argument("--skip-mlp", action="store_true")
    args = ap.parse_args()

    utils.log("=" * 70)
    utils.log("STEP 4: OPTUNA -- (A) phenotype weights, (B) MLP hyperparameters")
    utils.log("=" * 70)
    df = utils.read_model_columns(config.SCORED_PATH)
    split = mu.get_or_create_split(df)
    parts = mu.split_frames(df, split)
    train, val = parts["train"], parts["validation"]
    utils.log(f"Permanent split: train {len(train):,} | validation {len(val):,} | test {len(parts['test']):,} "
              f"(test is NOT used in this step)")

    # ---- Study A ----
    utils.log("Study A: phenotype weights ...")
    study_a, vw, dw, clinical_rmse = run_weight_study(train, val, args.weight_trials)
    utils.save_json({"variable_weights": vw, "domain_weights": dw, "best_validation_rmse": study_a.best_value,
                     "clinical_weights_validation_rmse": clinical_rmse,
                     "objective": "Ridge(domain scores + total) -> 30-day % BMI change, validation RMSE",
                     "n_trials": len(study_a.trials)}, config.LEARNED_WEIGHTS_PATH)

    ds, total = scoring_engine.full_scoring_pipeline(df, vw, dw)
    for d in DOMAINS:
        df[f"{d}_SCORE_CLINICAL"] = df[f"{d}_SCORE"]
        df[f"{d}_SCORE"] = ds[f"{d}_SCORE"]
    df["TOTAL_PHENOTYPE_SCORE_CLINICAL"] = df["TOTAL_PHENOTYPE_SCORE"]
    df["TOTAL_PHENOTYPE_SCORE"] = total
    df.to_parquet(config.FINAL_SCORED_PATH, index=False)
    del parts, train, val
    utils.log(f"Saved {config.FINAL_SCORED_PATH} (scores recomputed with learned weights; clinical-weight "
              f"scores kept as *_CLINICAL columns)")
    history_plot(study_a, "04_optuna_weight_history", "Optuna study A: phenotype weight learning",
                 "validation RMSE, 30-day % BMI change")

    base = utils.load_json(os.path.join(config.MODEL_DIR, "clinical_baseline_weights.json"))
    wrows = [{"domain": d, "variable": v, "clinical_weight_pct": round(base["variable_weights"][d][v], 2),
              "learned_weight_pct": round(w, 2)} for d, vs in vw.items() for v, w in vs.items()]
    wrows += [{"domain": d, "variable": "(DOMAIN WEIGHT)", "clinical_weight_pct": base["domain_weights"][d],
               "learned_weight_pct": round(w, 2)} for d, w in dw.items()]
    sheets = {
        "A_summary": pd.DataFrame([{"study": "A: phenotype weights", "n_trials": len(study_a.trials),
                                    "objective": "validation RMSE of Ridge(4 domain scores + total -> 30-day % BMI change)",
                                    "clinical_weights_val_RMSE": clinical_rmse, "best_val_RMSE": study_a.best_value,
                                    "best_trial": study_a.best_trial.number}]),
        "A_learned_weights": pd.DataFrame(wrows),
        "A_all_trials": study_a.trials_dataframe(),
        "A_param_importance_top20": importance_table(study_a, top=20),
    }

    # ---- Study B ----
    if not args.skip_mlp:
        utils.log("Study B: MLP hyperparameters (predicting 30-day BMI and weight) ...")
        p2 = mu.split_frames(df, split)
        study_b, n_tr, n_va = run_mlp_study(p2["train"], p2["validation"], args.mlp_trials)
        imp_b = importance_table(study_b)
        best = study_b.best_trial
        utils.save_json({
            "best_params": best.params,
            "best_validation_mse_standardized": best.value,
            "best_validation_rmse_BMI_30D": best.user_attrs.get("val_RMSE_BMI_30D"),
            "best_validation_rmse_WEIGHT_30D": best.user_attrs.get("val_RMSE_WEIGHT_30D"),
            "best_trial_number": best.number,
            "n_trials": len(study_b.trials),
            "n_complete": sum(t.state == optuna.trial.TrialState.COMPLETE for t in study_b.trials),
            "n_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study_b.trials),
            "tuning_rows": {"train": n_tr, "validation": n_va},
            "hyperparameter_importance": dict(zip(imp_b["parameter"], imp_b["importance"])),
            "search_space": {k: list(v) if isinstance(v, (list, tuple)) else v for k, v in config.MLP_SEARCH_SPACE.items()},
        }, config.BEST_MLP_PARAMS_PATH)
        history_plot(study_b, "04_optuna_mlp_history", "Optuna study B: MLP hyperparameter search",
                     "validation MSE (standardized BMI + weight change)")
        if len(imp_b):
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.barh(imp_b["parameter"][::-1], imp_b["importance"][::-1], color="steelblue")
            ax.set_xlabel("importance (fANOVA)")
            ax.set_title("MLP hyperparameter importance (Optuna study B)")
            utils.save_fig(fig, "04_optuna_mlp_param_importance")
        sheets.update({
            "B_summary": pd.DataFrame([{
                "study": "B: MLP hyperparameters", "n_trials": len(study_b.trials),
                "n_complete": sum(t.state == optuna.trial.TrialState.COMPLETE for t in study_b.trials),
                "n_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study_b.trials),
                "best_trial": best.number, "best_val_MSE_standardized": best.value,
                "best_val_RMSE_BMI_30D_kg_m2": best.user_attrs.get("val_RMSE_BMI_30D"),
                "best_val_RMSE_WEIGHT_30D_kg": best.user_attrs.get("val_RMSE_WEIGHT_30D"),
                "tuning_train_rows": n_tr, "tuning_validation_rows": n_va}]),
            "B_best_hyperparameters": pd.DataFrame([{"hyperparameter": k, "value": v} for k, v in best.params.items()]),
            "B_param_importance": imp_b,
            "B_all_trials": study_b.trials_dataframe(),
        })
    for k, v in sheets.items():
        for c in v.columns:
            if pd.api.types.is_datetime64_any_dtype(v[c]) or pd.api.types.is_timedelta64_dtype(v[c]):
                v[c] = v[c].astype(str)
    utils.save_excel_sheets(utils.excel_path("04_optuna_results.xlsx"), sheets)
    utils.log("Step 4 complete.")


if __name__ == "__main__":
    main()
