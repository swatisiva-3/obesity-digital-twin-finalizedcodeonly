"""
06_shap_explain.py
===================
STEP 6 -- SHAP explanations of the Step 5 MLP.

INPUT:  models/mlp_bmi_weight_30d.pt        (Step 5)
        models/mlp_preprocessor.json         (Step 5 -- the SAME train-only imputation/scaling)
        outputs/04_final_scored.parquet      (Step 4; falls back to 03_scored.parquet)
        models/train_val_test_indices.json   (explained patients come from TEST, background from TRAIN)
OUTPUT: outputs/figures/06_shap_global_bar_all_variables_BMI.pdf     (all variables, 300 dpi)
        outputs/figures/06_shap_beeswarm_top12_BMI.pdf               (top 12, 300 dpi)
        outputs/figures/06_shap_global_bar_all_variables_WEIGHT.pdf  / 06_shap_beeswarm_top12_WEIGHT.pdf
        outputs/figures/06_shap_domain_bar_{BMI,WEIGHT}.png          (importance summed per domain)
        outputs/figures/06_shap_domain_<Domain>_BMI.png              (one beeswarm per domain, 4 total)
        outputs/figures/06_shap_domain_importance_by_year.png        (2017 vs 2020 vs 2024)
        outputs/figures/06_shap_patient_<PATIENT_KEY>.png            (only with --patient)
        outputs/excel/06_shap_importance.xlsx

WHAT IS EXPLAINED: the network predicts each patient's CHANGE from pre-op
(Step 5 docstring); predicted 30-day value = pre-op value + that change. SHAP
here explains the predicted change (in kg/m^2 for BMI and kg for weight) --
i.e. what drives how much a patient is predicted to lose -- which is the
part of the prediction the model actually learned. (The pre-op value itself
carries forward 1-for-1 and needs no explanation.)

HOW: shap.GradientExplainer (expected gradients) on the PyTorch model, with
a year-stratified sample of test patients explained against a training-set
background. The one-hot columns of RACE / SEX / HISPANIC / SURGSPECIALTY_BAR
are summed back into one SHAP value per variable (SHAP values are additive),
so every plot shows one row per variable: the 26 rationale variables plus
FOLLOWUP_DAY. In the beeswarm, those 4 categorical variables are coloured by
category code (not a meaningful high/low scale).

Run: python 06_shap_explain.py [--patient 2024_12345]
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

import config
import model_utils as mu
import utils


def load_model():
    import torch
    ck = torch.load(config.MLP_MODEL_PATH, weights_only=False)
    model = mu.build_mlp(ck["n_in"], ck["n_out"], ck["params"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model


def stratified_sample(frame, n, seed):
    per = max(1, n // frame["OPYEAR"].nunique())
    return pd.concat([g.sample(n=min(per, len(g)), random_state=seed) for _, g in frame.groupby("OPYEAR")],
                     ignore_index=True)


def aggregate_to_variables(shap_feat, X_raw, f2v, variables):
    """Sum one-hot SHAP columns per variable; build a numeric 'data' matrix for colouring."""
    vals, data = [], []
    for v in variables:
        cols = [c for c in X_raw.columns if f2v.get(c) == v]
        idx = [X_raw.columns.get_loc(c) for c in cols]
        vals.append(shap_feat[:, idx].sum(axis=1))
        if len(cols) == 1 and "=" not in cols[0]:
            data.append(X_raw[cols[0]].values.astype(float))
        else:  # one-hot group -> category code of the active column
            data.append(np.argmax(X_raw[cols].values, axis=1).astype(float))
    return np.column_stack(vals), np.column_stack(data)


def save_current(name, formats=("pdf", "png")):
    fig = plt.gcf()
    utils.save_fig(fig, name, dpi=300, formats=formats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default=config.TARGET_CASEID,
                    help="PATIENT_KEY (e.g. 2024_1234567) for a single-patient waterfall plot")
    args = ap.parse_args()
    utils.log("=" * 70)
    utils.log("STEP 6: SHAP EXPLANATIONS")
    utils.log("=" * 70)
    import torch

    model = load_model()
    pj = utils.load_json(config.MLP_PREPROCESSOR_PATH)
    prep = mu.Preprocessor.from_dict(pj)
    f2v, f2d = pj["feature_to_variable"], pj["feature_to_domain"]

    df, source = mu.load_scored_with_fallback()
    df["OPYEAR"] = df["OPYEAR"].astype(str)
    split = mu.get_or_create_split(df)
    parts = mu.split_frames(df, split)
    explain = stratified_sample(parts["test"], config.SHAP_N_EXPLAIN, config.RANDOM_SEED)
    if args.patient and args.patient not in set(explain["PATIENT_KEY"]):
        extra = df[df["PATIENT_KEY"] == str(args.patient)]
        if len(extra):
            explain = pd.concat([explain, extra], ignore_index=True)
        else:
            utils.log(f"WARNING: --patient {args.patient} not found (use PATIENT_KEY = OPYEAR_CASEID)")
    background = parts["train"].sample(n=config.SHAP_N_BACKGROUND, random_state=config.RANDOM_SEED)
    del df, parts

    X_raw = mu.build_feature_matrix(explain)[0].reindex(columns=prep.columns, fill_value=0.0)
    X_bg = mu.build_feature_matrix(background)[0].reindex(columns=prep.columns, fill_value=0.0)
    Xs = torch.as_tensor(prep.transform_X(X_raw).values)
    Xb = torch.as_tensor(prep.transform_X(X_bg).values)
    utils.log(f"Explaining {len(explain):,} test patients ({explain['OPYEAR'].value_counts().to_dict()}) "
              f"against {len(background)} training-set background patients; {X_raw.shape[1]} model features")

    explainer = shap.GradientExplainer(model, Xb)
    sv = explainer.shap_values(Xs, nsamples=200, rseed=config.RANDOM_SEED)
    sv = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)  # (n, features, outputs)
    with torch.no_grad():
        base_std = model(Xb).numpy().mean(axis=0)

    variables = list(dict.fromkeys(f2v[c] for c in prep.columns))
    var_domain = {f2v[c]: f2d[c] for c in prep.columns}
    # clean X for colouring: numeric features in real units (median-imputed), one-hots 0/1
    X_color = X_raw.fillna(prep.medians)

    importance_rows, per_patient, expl_by_target = [], {}, {}
    for j, t in enumerate(mu.TARGET_NAMES):
        scale = float(prep.y_stds.iloc[j])
        vals, data = aggregate_to_variables(sv[:, :, j] * scale, X_color, f2v, variables)
        base = float(base_std[j] * scale + prep.y_means.iloc[j])
        expl = shap.Explanation(values=vals, base_values=np.full(len(vals), base), data=data,
                                feature_names=variables)
        expl_by_target[t] = expl
        per_patient[t] = pd.DataFrame(vals, columns=variables).assign(PATIENT_KEY=explain["PATIENT_KEY"].values,
                                                                      OPYEAR=explain["OPYEAR"].values)
        unit = "kg/m^2" if t == "BMI" else "kg"
        n_all = len(variables)

        # --- your requested pattern: all-variable bar + top-12 beeswarm, 300-dpi PDFs ---
        plt.figure()
        shap.plots.bar(expl, max_display=n_all, show=False)
        plt.title(f"Global SHAP importance, all {n_all} variables -- predicted 30-day {t} change ({unit})")
        save_current(f"06_shap_global_bar_all_variables_{t}")
        plt.figure()
        shap.plots.beeswarm(expl, max_display=config.SHAP_TOP_N_BEESWARM, show=False)
        plt.title(f"SHAP beeswarm, top {config.SHAP_TOP_N_BEESWARM} variables -- predicted 30-day {t} change ({unit})")
        save_current(f"06_shap_beeswarm_top12_{t}")

        mean_abs = np.abs(vals).mean(axis=0)
        for k, v in enumerate(variables):
            row = {"target": t, "variable": v, "domain": var_domain[v], "mean_abs_SHAP": mean_abs[k],
                   "mean_SHAP": vals[:, k].mean(), "unit": unit}
            for yr in sorted(explain["OPYEAR"].unique()):
                row[f"mean_abs_SHAP_{yr}"] = np.abs(vals[explain["OPYEAR"].values == yr, k]).mean()
            importance_rows.append(row)

        # domain-wise bar
        dom = pd.DataFrame({"domain": [var_domain[v] for v in variables], "imp": mean_abs}).groupby("domain")["imp"].sum().sort_values()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(dom.index, dom.values, color="slateblue")
        ax.set_xlabel(f"sum of mean |SHAP| over the domain's variables ({unit})")
        ax.set_title(f"Domain-wise SHAP importance -- predicted 30-day {t} change")
        utils.save_fig(fig, f"06_shap_domain_bar_{t}", formats=("png", "pdf"))

    imp = pd.DataFrame(importance_rows)
    imp["rank"] = imp.groupby("target")["mean_abs_SHAP"].rank(ascending=False).astype(int)
    imp = imp.sort_values(["target", "rank"])

    # per-domain beeswarms for the configured target
    t = config.SHAP_DOMAIN_PLOT_TARGET
    expl = expl_by_target[t]
    for domain in list(config.VARIABLE_RATIONALE.keys()):
        idx = [k for k, v in enumerate(variables) if var_domain[v] == domain]
        if not idx:
            continue
        sub = shap.Explanation(values=expl.values[:, idx], base_values=expl.base_values, data=expl.data[:, idx],
                               feature_names=[variables[k] for k in idx])
        plt.figure()
        shap.plots.beeswarm(sub, max_display=len(idx), show=False)
        plt.title(f"{domain} domain -- SHAP for predicted 30-day {t} change (all 3 years)")
        save_current(f"06_shap_domain_{domain}_{t}", formats=("png",))

    # domain importance by year
    years = sorted(explain["OPYEAR"].unique())
    dyr = imp[imp.target == t].groupby("domain")[[f"mean_abs_SHAP_{y}" for y in years]].sum()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.8 / len(years)
    x = np.arange(len(dyr))
    for i, y in enumerate(years):
        ax.bar(x + i * w, dyr[f"mean_abs_SHAP_{y}"], width=w, label=y)
    ax.set_xticks(x + w * (len(years) - 1) / 2)
    ax.set_xticklabels(dyr.index, rotation=15)
    ax.set_ylabel("sum of mean |SHAP|")
    ax.set_title(f"Does what drives predicted 30-day {t} change differ by year?")
    ax.legend(title="year")
    utils.save_fig(fig, "06_shap_domain_importance_by_year")

    # clinical weight vs learned weight vs SHAP
    import os
    lw = utils.load_json(config.LEARNED_WEIGHTS_PATH) if os.path.exists(config.LEARNED_WEIGHTS_PATH) else None
    comp = []
    for d, dd in config.VARIABLE_RATIONALE.items():
        for v, spec in dd["variables"].items():
            if spec["column"] is None:
                continue
            r = imp[(imp.target == t) & (imp.variable == v)]
            comp.append({"domain": d, "variable": v, "clinical_weight_within_domain_pct": spec["weight"],
                         "optuna_learned_weight_pct": (lw["variable_weights"].get(d, {}).get(v) if lw else None),
                         "contributes_to_severity_score": spec.get("contributes_to_severity_score", True),
                         f"mean_abs_SHAP_{t}": float(r["mean_abs_SHAP"].iloc[0]) if len(r) else None,
                         f"SHAP_rank_{t}": int(r["rank"].iloc[0]) if len(r) else None})

    # single patient
    if args.patient and args.patient in set(explain["PATIENT_KEY"]):
        k = int(np.where(explain["PATIENT_KEY"].values == args.patient)[0][0])
        for tt in mu.TARGET_NAMES:
            plt.figure()
            shap.plots.waterfall(expl_by_target[tt][k], max_display=15, show=False)
            plt.title(f"Patient {args.patient}: why the model predicts this 30-day {tt} change")
            save_current(f"06_shap_patient_{args.patient}_{tt}", formats=("png",))

    utils.save_excel_sheets(utils.excel_path("06_shap_importance.xlsx"), {
        "variable_importance": imp,
        "domain_importance": imp.groupby(["target", "domain"])["mean_abs_SHAP"].sum().reset_index(),
        f"clinical_vs_SHAP_{t}": pd.DataFrame(comp),
        "patient_SHAP_BMI": per_patient["BMI"], "patient_SHAP_WEIGHT": per_patient["WEIGHT"],
        "notes": pd.DataFrame({"note": [
            "SHAP explains the model's predicted CHANGE from pre-op (kg/m^2 for BMI, kg for weight).",
            f"Explained: {len(explain):,} test patients (stratified by year); background: {config.SHAP_N_BACKGROUND} training patients.",
            "Method: shap.GradientExplainer (expected gradients) on the PyTorch MLP; one-hot SHAP summed per variable.",
            f"Input: {source}."]})})
    utils.log("Top variables:\n" + imp[imp["rank"] <= 8][["target", "rank", "variable", "domain", "mean_abs_SHAP"]].round(4).to_string(index=False))
    utils.log("Step 6 complete.")


if __name__ == "__main__":
    main()
