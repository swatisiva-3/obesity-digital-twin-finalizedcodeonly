"""
scoring_engine.py
==================
Shared scoring logic used by BOTH 03_scoring.py (the clinical-weight
baseline) and 04_optuna_weight_learning.py (which calls this same engine
hundreds of times with different candidate weight sets while it searches).
It lives in its own file, instead of inside 03_scoring.py, because Python
module names can't start with a digit -- 04 needs to import this logic, and
`import 03_scoring` isn't valid syntax.

Nothing here is dataset-specific; it all operates on the __SEVERITY columns
that 02_preprocessing.py already computed, plus whatever weight dictionary
it's handed.
"""

import numpy as np
import pandas as pd
import config


def available_variables(domain_name):
    """
    Return {var_name: spec} for every variable in a domain that (a) exists
    in this dataset (column is not None) and (b) is meant to contribute a
    severity number (contributes_to_severity_score == True). This is where
    MOBILITY_DEVICE/PRIORITY (not in this PUF) and RACE/SEX/HISPANIC/OPYEAR/
    SURGSPECIALTY_BAR (context-only, see config.py) get excluded.
    """
    domain = config.VARIABLE_RATIONALE[domain_name]["variables"]
    return {
        name: spec for name, spec in domain.items()
        if spec["column"] is not None and spec.get("contributes_to_severity_score", True)
    }


def renormalize_weights(weights_dict):
    """Rescale a {name: weight} dict so its values sum to 100, preserving ratios."""
    total = sum(weights_dict.values())
    if total <= 0:
        n = len(weights_dict)
        return {k: 100.0 / n for k in weights_dict} if n else {}
    return {k: v / total * 100.0 for k, v in weights_dict.items()}


def default_variable_weights():
    """
    Build the starting {domain: {var_name: weight}} structure straight from
    config.VARIABLE_RATIONALE, with any dataset-unavailable variable already
    dropped and the remaining weights renormalized back to sum to 100 within
    each domain. This is the pure "clinical rationale" weight set, before any
    flag-count nudge or Optuna search touches it.
    """
    weights = {}
    for domain_name in config.VARIABLE_RATIONALE:
        avail = available_variables(domain_name)
        raw = {name: spec["weight"] for name, spec in avail.items()}
        weights[domain_name] = renormalize_weights(raw)
    return weights


def default_domain_weights():
    return dict(config.DOMAIN_WEIGHTS_PRIOR)


def compute_domain_scores(df, variable_weights):
    """
    Given the preprocessed dataframe (with __SEVERITY columns from Step 2)
    and a {domain: {var_name: weight}} dict, compute one 0-100 score per
    domain per patient. Missing values are handled PER PATIENT: if a given
    patient is missing (say) their albumin severity, that variable's weight
    is dropped from THAT patient's weighted average and the remaining
    weights for that patient are renormalized -- a patient isn't penalized
    or rewarded just because a lab wasn't drawn.
    Returns a DataFrame with one column per domain: "<Domain>_SCORE".
    """
    domain_score_cols = {}
    for domain_name, var_weights in variable_weights.items():
        if not var_weights:
            domain_score_cols[f"{domain_name}_SCORE"] = pd.Series(np.nan, index=df.index)
            continue
        severity_cols = {v: f"{v}__SEVERITY" for v in var_weights if f"{v}__SEVERITY" in df.columns}
        if not severity_cols:
            domain_score_cols[f"{domain_name}_SCORE"] = pd.Series(np.nan, index=df.index)
            continue

        sev = df[list(severity_cols.values())].copy()
        sev.columns = list(severity_cols.keys())
        w = pd.Series({v: var_weights[v] for v in severity_cols})

        weighted = sev.mul(w, axis=1)
        weight_mask = sev.notna().mul(w, axis=1)  # weight contributed only where severity is present
        numerator = weighted.sum(axis=1, skipna=True)
        denominator = weight_mask.sum(axis=1)
        domain_score = np.where(denominator > 0, numerator / denominator, np.nan)
        domain_score_cols[f"{domain_name}_SCORE"] = pd.Series(domain_score, index=df.index)

    return pd.DataFrame(domain_score_cols)


def compute_total_score(domain_scores_df, domain_weights):
    """
    Combine the 4 domain scores into ONE total score, 0-100, using the same
    per-row missing-data renormalization logic as compute_domain_scores.
    """
    domains = list(domain_weights.keys())
    cols = [f"{d}_SCORE" for d in domains]
    present_cols = [c for c in cols if c in domain_scores_df.columns]
    scores = domain_scores_df[present_cols]
    w = pd.Series({f"{d}_SCORE": domain_weights[d] for d in domains if f"{d}_SCORE" in present_cols})

    weighted = scores.mul(w, axis=1)
    weight_mask = scores.notna().mul(w, axis=1)
    numerator = weighted.sum(axis=1, skipna=True)
    denominator = weight_mask.sum(axis=1)
    return pd.Series(np.where(denominator > 0, numerator / denominator, np.nan), index=domain_scores_df.index)


def apply_flag_count_weight_nudge(df, variable_weights):
    """
    Your rule: "if more than N cases are flagged on a variable, nudge its
    weight up or down (per that variable's own flag_direction_logic in
    config.py) before Optuna ever sees it." This is a ONE-TIME adjustment
    to the clinical starting weights -- not something Optuna itself does.
    Optuna, in 04_optuna_weight_learning.py, then treats THIS adjusted set
    as its initial sample point and is free to move away from it in any
    direction as it searches for whatever weights best predict the real
    30-day outcomes.
    """
    adjusted = {domain: dict(weights) for domain, weights in variable_weights.items()}
    for domain_name, var_weights in variable_weights.items():
        avail = available_variables(domain_name)
        for var_name in var_weights:
            flag_col = f"{var_name}__FLAG"
            if flag_col not in df.columns:
                continue
            n_flagged = int(df[flag_col].sum())
            if n_flagged > config.FLAG_THRESHOLD_COUNT:
                direction = avail.get(var_name, {}).get("flag_direction_logic", "increase")
                step = config.FLAG_WEIGHT_STEP if direction == "increase" else -config.FLAG_WEIGHT_STEP
                adjusted[domain_name][var_name] = max(0.0, adjusted[domain_name][var_name] + step)
        adjusted[domain_name] = renormalize_weights(adjusted[domain_name])
    return adjusted


def full_scoring_pipeline(df, variable_weights, domain_weights):
    """
    Run the whole scoring stack and return:
      - domain_scores_df: one "<Domain>_SCORE" column per domain
      - total_score: one Series, 0-100
    This is the single function both 03_scoring.py and
    04_optuna_weight_learning.py call, so the scoring MATH is defined in
    exactly one place.
    """
    domain_scores_df = compute_domain_scores(df, variable_weights)
    total_score = compute_total_score(domain_scores_df, domain_weights)
    return domain_scores_df, total_score
