"""
model_utils.py
===============
Everything the modeling steps (04, 05, 06, 07, 08) share, so they all build
IDENTICAL features, use the SAME permanent split, and score models the SAME
way. (Lives outside the numbered scripts because `import 04_...` isn't valid
Python.)

FEATURES -- one or more model columns per rationale variable (26 variables):
  * numeric variables (BMI, BMI_HIGH_BAR, WGT_HIGH_BAR, WGT_CLOSEST, HGT,
    ALBUMIN, CREATININE, NBHTN_MEDS, AGE, OPYEAR) -> the unit-standardized
    raw value (`<VAR>__STD`: kg, cm, g/dL ...). NOT the 0-100 severity: the
    severity score is clipped at 100 (pre-op BMI saturates at 46.7), which
    would throw away exactly the information a 30-day BMI model needs most.
  * binary/ordinal clinical variables (DIABETES, ASACLASS, SMOKER, ...) ->
    their 0-100 severity code (an ordered numeric encoding, e.g. DIABETES
    No=0, non-insulin=60, insulin=100).
  * context categoricals with no severity (RACE, SEX, HISPANIC,
    SURGSPECIALTY_BAR) -> one-hot columns. SHAP (Step 6) sums those back
    into one value per variable, so every plot still shows 26 variables.
PLUS one time covariate, FOLLOWUP_DAY (days from surgery to the "30-day"
measurement). It isn't a phenotype variable, but the "30-day" value is not
taken on a fixed day (median day 14, range 0-30 in every year), so the model
needs to know WHEN the measurement was taken. It is also what lets Step 5
draw a model-predicted BMI/weight trajectory across days 0-30.
Missing values are imputed with TRAINING-SET medians and everything is
standardized with TRAINING-SET mean/std (Preprocessor below) -- validation
and test patients never influence those numbers.

TARGETS -- the MLP predicts the 30-day CHANGE from pre-op (BMI change and
weight change, standardized), and the reported prediction is
pre-op value + predicted change = predicted 30-day BMI / weight. That's the
same thing as predicting the 30-day values, but much easier to learn, and
it lets us report accuracy both on the 30-day values (what you asked for)
and on the change itself (the harder, more honest number -- a model that
just copies pre-op BMI forward already gets R^2 ~0.97 on 30-day BMI).
"""
import json
import os

import numpy as np
import pandas as pd

import config
import utils

TARGETS = {
    "BMI": {"outcome": "OUTCOME_BMI_30D", "baseline": "BASELINE_BMI", "change": "BMI_CHANGE_30D",
            "label": "30-day BMI (kg/m^2)"},
    "WEIGHT": {"outcome": "OUTCOME_WEIGHT_30D_KG", "baseline": "BASELINE_WEIGHT_KG", "change": "WEIGHT_CHANGE_30D_KG",
               "label": "30-day weight (kg)"},
}
TARGET_NAMES = list(TARGETS.keys())
CONTEXT_MISSING_LABEL = "Missing"
TIME_DOMAIN = "Follow-up timing"


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def variable_specs():
    """[(domain, var_name, spec)] for every rationale variable that has a real column."""
    out = []
    for domain, d in config.VARIABLE_RATIONALE.items():
        for var, spec in d["variables"].items():
            if spec["column"] is not None:
                out.append((domain, var, spec))
    return out


def build_feature_matrix(df):
    """
    Returns (X, feature_to_variable, feature_to_domain). X still contains NaN
    -- impute with a Preprocessor fit on the training split.
    """
    cols, f2v, f2d = {}, {}, {}
    for domain, var, spec in variable_specs():
        contributes = spec.get("contributes_to_severity_score", True)
        if spec["var_type"] == "numeric":
            src = f"{var}__STD"
            if src not in df.columns:
                continue
            cols[src] = pd.to_numeric(df[src], errors="coerce")
            f2v[src], f2d[src] = var, domain
        elif contributes and spec.get("category_scores"):
            src = f"{var}__SEVERITY"
            cols[src] = pd.to_numeric(df[src], errors="coerce")
            f2v[src], f2d[src] = var, domain
        else:
            src = f"{var}__STD"
            if src not in df.columns:
                continue
            cats = df[src].astype("string").fillna(CONTEXT_MISSING_LABEL)
            for level in sorted(cats.unique()):
                name = f"{var}={level}"
                cols[name] = (cats == level).astype(float)
                f2v[name], f2d[name] = var, domain
    if "FOLLOWUP_DAY" in df.columns:
        cols["FOLLOWUP_DAY"] = pd.to_numeric(df["FOLLOWUP_DAY"], errors="coerce")
        f2v["FOLLOWUP_DAY"], f2d["FOLLOWUP_DAY"] = "FOLLOWUP_DAY", TIME_DOMAIN
    X = pd.DataFrame(cols, index=df.index)
    return X, f2v, f2d


class Preprocessor:
    """Train-only median imputation + standardization for X, and standardization for Y."""

    def fit(self, X, Y=None):
        self.columns = list(X.columns)
        self.medians = X.median().fillna(0.0)
        Xi = X.fillna(self.medians)
        self.means = Xi.mean()
        self.stds = Xi.std().replace(0, 1.0).fillna(1.0)
        if Y is not None:
            self.y_columns = list(Y.columns)
            self.y_means = Y.mean()
            self.y_stds = Y.std().replace(0, 1.0)
        return self

    def transform_X(self, X):
        X = X.reindex(columns=self.columns, fill_value=0.0)
        return ((X.fillna(self.medians) - self.means) / self.stds).astype(np.float32)

    def transform_Y(self, Y):
        return ((Y[self.y_columns] - self.y_means) / self.y_stds).astype(np.float32)

    def inverse_Y(self, arr):
        return arr * self.y_stds.values + self.y_means.values

    def to_dict(self):
        d = {"columns": self.columns, "medians": self.medians.to_dict(),
             "means": self.means.to_dict(), "stds": self.stds.to_dict()}
        if hasattr(self, "y_columns"):
            d.update({"y_columns": self.y_columns, "y_means": self.y_means.to_dict(), "y_stds": self.y_stds.to_dict()})
        return d

    @classmethod
    def from_dict(cls, d):
        p = cls()
        p.columns = d["columns"]
        p.medians = pd.Series(d["medians"]).reindex(p.columns)
        p.means = pd.Series(d["means"]).reindex(p.columns)
        p.stds = pd.Series(d["stds"]).reindex(p.columns)
        if "y_columns" in d:
            p.y_columns = d["y_columns"]
            p.y_means = pd.Series(d["y_means"]).reindex(p.y_columns)
            p.y_stds = pd.Series(d["y_stds"]).reindex(p.y_columns)
        return p


def change_targets(df):
    return df[[TARGETS[t]["change"] for t in TARGET_NAMES]]


# ---------------------------------------------------------------------------
# Permanent 70/15/15 split (stratified by year), keyed by PATIENT_KEY
# ---------------------------------------------------------------------------
def create_split(df):
    rng = np.random.RandomState(config.RANDOM_SEED)
    eligible = df[df["HAS_VALID_OUTCOME"]]
    parts = {"train": [], "validation": [], "test": []}
    for yr, sub in eligible.groupby("OPYEAR"):
        keys = rng.permutation(np.asarray(sub["PATIENT_KEY"].astype(str).tolist(), dtype=object))
        n = len(keys)
        n_tr = int(round(n * config.TRAIN_FRACTION))
        n_va = int(round(n * config.VALIDATION_FRACTION))
        parts["train"] += keys[:n_tr].tolist()
        parts["validation"] += keys[n_tr:n_tr + n_va].tolist()
        parts["test"] += keys[n_tr + n_va:].tolist()
    split = {
        "description": "Permanent train/validation/test split of patients with a valid 30-day BMI+weight "
                       "outcome, stratified by OPYEAR. Keys are PATIENT_KEY (= OPYEAR_CASEID). Created once; "
                       "every model in this pipeline uses it. Delete this file to re-split.",
        "fractions": {"train": config.TRAIN_FRACTION, "validation": config.VALIDATION_FRACTION, "test": config.TEST_FRACTION},
        "random_seed": config.RANDOM_SEED,
        "n_eligible": int(len(eligible)),
        "counts": {k: len(v) for k, v in parts.items()},
        "patient_keys": parts,
    }
    validate_split(split, eligible["PATIENT_KEY"])
    utils.save_json(split, config.SPLIT_INDICES_PATH)
    utils.log(f"Created permanent split -> {config.SPLIT_INDICES_PATH}: {split['counts']}")
    return split


def validate_split(split, eligible_keys):
    tr, va, te = (set(split["patient_keys"][k]) for k in ("train", "validation", "test"))
    assert not (tr & va) and not (tr & te) and not (va & te), "split sets overlap"
    missing = set(eligible_keys) - (tr | va | te)
    assert not missing, f"{len(missing)} eligible patients are in no split"


def get_or_create_split(df):
    """Load the permanent split if it exists (and still matches the data), else create it."""
    if os.path.exists(config.SPLIT_INDICES_PATH):
        split = utils.load_json(config.SPLIT_INDICES_PATH)
        eligible = set(df.loc[df["HAS_VALID_OUTCOME"], "PATIENT_KEY"])
        in_split = set().union(*[set(v) for v in split["patient_keys"].values()])
        if eligible == in_split:
            return split
        utils.log("WARNING: saved split no longer matches the eligible patients (data changed?) -- re-creating it")
    return create_split(df)


def split_frames(df, split):
    idx = df.set_index("PATIENT_KEY", drop=False)
    return {k: idx.loc[split["patient_keys"][k]].reset_index(drop=True) for k in ("train", "validation", "test")}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def regression_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[m], y_pred[m]
    resid = y_true - y_pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {"RMSE": float(np.sqrt(np.mean(resid ** 2))), "MAE": float(np.mean(np.abs(resid))),
            "R2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"), "n": int(m.sum())}


def responder_metrics(baseline_bmi, actual_bmi, predicted_bmi, threshold=None):
    """AUC-ROC / F1 / precision / recall for the derived 'early responder' label (config.py Section 9)."""
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
    thr = config.CLINICAL_RESPONSE_BMI_REDUCTION_FRACTION if threshold is None else threshold
    b, a, p = (np.asarray(x, float) for x in (baseline_bmi, actual_bmi, predicted_bmi))
    m = np.isfinite(b) & np.isfinite(a) & np.isfinite(p) & (b > 0)
    actual_red, pred_red = (b[m] - a[m]) / b[m], (b[m] - p[m]) / b[m]
    y, yhat = (actual_red >= thr).astype(int), (pred_red >= thr).astype(int)
    out = {"threshold_fraction": thr, "n": int(m.sum()), "prevalence": float(y.mean()),
           "AUC_ROC": float(roc_auc_score(y, pred_red)) if 0 < y.mean() < 1 else float("nan"),
           "F1": float(f1_score(y, yhat, zero_division=0)),
           "Precision": float(precision_score(y, yhat, zero_division=0)),
           "Recall": float(recall_score(y, yhat, zero_division=0)),
           "Accuracy": float(accuracy_score(y, yhat))}
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    out.update({"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)})
    return out, y, pred_red


# ---------------------------------------------------------------------------
# PyTorch MLP (two outputs: BMI change, weight change)
# ---------------------------------------------------------------------------
def _torch():
    import torch
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    return torch


def build_mlp(n_in, n_out, params):
    torch = _torch()
    nn = torch.nn
    acts = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "gelu": nn.GELU, "tanh": nn.Tanh}
    layers, width = [], n_in
    for _ in range(int(params["n_layers"])):
        layers.append(nn.Linear(width, int(params["n_units"])))
        if params.get("batch_norm"):
            layers.append(nn.BatchNorm1d(int(params["n_units"])))
        layers.append(acts[params["activation"]]())
        if params.get("dropout", 0) > 0:
            layers.append(nn.Dropout(float(params["dropout"])))
        width = int(params["n_units"])
    layers.append(nn.Linear(width, n_out))
    return nn.Sequential(*layers)


def make_optimizer(model, params):
    torch = _torch()
    lr, wd = float(params["learning_rate"]), float(params["weight_decay"])
    if params["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if params["optimizer"] == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9, nesterov=True)


def predict(model, X, batch_size=8192):
    torch = _torch()
    model.eval()
    out = []
    with torch.no_grad():
        Xt = torch.as_tensor(np.array(X, dtype=np.float32))
        for i in range(0, len(Xt), batch_size):
            out.append(model(Xt[i:i + batch_size]).numpy())
    return np.vstack(out)


def train_mlp(Xtr, Ytr, Xva, Yva, params, max_epochs, patience, seed=None, trial=None, verbose=False):
    """
    Mini-batch training with early stopping on validation MSE (standardized
    targets). Returns (model at best epoch, history list, best_epoch).
    If an Optuna `trial` is passed, reports each epoch and supports pruning.
    """
    torch = _torch()
    seed = config.RANDOM_SEED if seed is None else seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr_t, Ytr_t = torch.as_tensor(np.array(Xtr, dtype=np.float32)), torch.as_tensor(np.array(Ytr, dtype=np.float32))
    Xva_np, Yva_np = np.asarray(Xva, np.float32), np.asarray(Yva, np.float32)
    model = build_mlp(Xtr_t.shape[1], Ytr_t.shape[1], params)
    opt = make_optimizer(model, params)
    loss_fn = torch.nn.MSELoss()
    bs = int(params["batch_size"])
    best, best_state, best_epoch, bad, history = np.inf, None, 0, 0, []
    g = torch.Generator().manual_seed(seed)
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr_t), generator=g)
        tot = 0.0
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            if len(b) < 2:
                continue  # BatchNorm needs >1 sample
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[b]), Ytr_t[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(b)
        train_loss = tot / len(perm)
        val_loss = float(np.mean((predict(model, Xva_np) - Yva_np) ** 2))
        if not np.isfinite(val_loss):
            break
        history.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss})
        if verbose:
            utils.log(f"    epoch {epoch:3d}  train MSE {train_loss:.4f}  val MSE {val_loss:.4f}")
        if val_loss < best - 1e-5:
            best, best_epoch, bad = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()
        if bad >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history, best_epoch


def predict_outcomes(model, prep, X_df, baseline_df):
    """Model output (standardized changes) -> predicted 30-day BMI and weight in real units."""
    changes = prep.inverse_Y(predict(model, prep.transform_X(X_df).values))
    out = {}
    for j, t in enumerate(TARGET_NAMES):
        out[f"PRED_{t}_CHANGE"] = changes[:, j]
        out[f"PRED_{t}_30D"] = baseline_df[TARGETS[t]["baseline"]].values + changes[:, j]
    return pd.DataFrame(out, index=X_df.index)


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def load_best_mlp_params():
    if os.path.exists(config.BEST_MLP_PARAMS_PATH):
        return utils.load_json(config.BEST_MLP_PARAMS_PATH)["best_params"], "optuna"
    return dict(config.MLP_DEFAULT_PARAMS), "config.MLP_DEFAULT_PARAMS (Optuna MLP study not run yet)"


def load_scored_with_fallback():
    """Step 4's Optuna-weighted scores if available, else Step 3's clinical-weight scores."""
    if os.path.exists(config.FINAL_SCORED_PATH):
        return pd.read_parquet(config.FINAL_SCORED_PATH), "04_final_scored (Optuna-learned weights)"
    return utils.load_modeling_frame(config.SCORED_PATH), "03_scored (clinical weights; Step 4 not run yet)"
