"""
07_phenograph_clustering.py
============================
STEP 7 -- PhenoGraph clustering + UMAP on the POOLED 3-year population
(2017 + 2020 + 2024 together), exactly two main analyses:

  A. TOTAL  -- one clustering of all patients on their 4 phenotype domain
               scores (Adiposity, Metabolic, Behavioral, Socio-Environmental).
  B. DOMAIN -- one clustering PER DOMAIN, each on that domain's own
               variable-level severity scores, all patients, all 3 years.

INPUT:  outputs/04_final_scored.parquet   (Step 4; falls back to 03_scored.parquet --
                                           clustering never needs outcome data)
OUTPUT: outputs/07_cluster_assignments.parquet   (every patient: TOTAL_CLUSTER + one cluster per domain)
        outputs/excel/07_phenograph_clusters.xlsx
            quality_metrics      clusters, outliers, modularity, silhouette, Davies-Bouldin,
                                 Calinski-Harabasz -- for every analysis
            stability_ARI_NMI    ARI / NMI of each re-run vs. the reference clustering
            cluster_sizes        size distribution (fit sample and all patients)
            cluster_by_year      % of each year's patients in each cluster
            cluster_profiles     mean domain scores / BMI / 30-day % BMI change per cluster
            umap_settings        n_neighbors, min_dist, metric, random seed, ...
            preprocessing        features, imputation, scaling, jitter, PhenoGraph k, min size
            assignments_*        per-patient cluster labels (one sheet per year)
        outputs/figures/07_umap_total_3yr.png           (A: clusters | by year)
        outputs/figures/07_umap_domainwise_3yr.png      (B: 4 panels, one per domain)
        outputs/figures/07_umap_total_domain_gradients.png
        outputs/figures/07_cluster_size_distribution.png
        outputs/figures/07_cluster_stability.png

WHY A SAMPLE: PhenoGraph + UMAP on 546k patients would take hours on a
laptop, so both are fit on config.PHENOGRAPH_SAMPLE_PER_YEAR patients from
EACH year (equal year representation), then every remaining patient is
assigned to its nearest cluster with a k-nearest-neighbour classifier in the
same scaled feature space. Clusters smaller than
config.PHENOGRAPH_MIN_CLUSTER_FRACTION of the sample are labelled -1
("outlier / tiny cluster").

STABILITY: PhenoGraph is re-run config.CLUSTER_STABILITY_N_RUNS times, each
on a different random config.CLUSTER_STABILITY_SUBSAMPLE of the sample; each
re-run is compared with the reference clustering on the patients they share
using the Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI).
~1 = the same clusters come back every time; ~0 = clusters are noise.
"""
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import phenograph
import umap
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score, silhouette_score,
                             davies_bouldin_score, calinski_harabasz_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import config
import model_utils as mu
import utils

DOMAINS = list(config.VARIABLE_RATIONALE.keys())
SCORE_COLS = [f"{d}_SCORE" for d in DOMAINS]
YEAR_COLORS = {"2017": "#1f77b4", "2020": "#ff7f0e", "2024": "#2ca02c"}


def domain_feature_columns(domain):
    cols = []
    for var, spec in config.VARIABLE_RATIONALE[domain]["variables"].items():
        if spec["column"] is not None and spec.get("contributes_to_severity_score", True):
            cols.append(f"{var}__SEVERITY")
    return cols


def prepare(frame, cols, scaler=None, medians=None):
    X = frame[cols].astype(float)
    medians = X.median() if medians is None else medians
    X = X.fillna(medians)
    scaler = StandardScaler().fit(X) if scaler is None else scaler
    return scaler.transform(X), scaler, medians


def run_phenograph(X, seed):
    """
    Cluster patients. Many patients share an IDENTICAL profile in the
    low-cardinality domains (e.g. ~80% of patients have every Behavioral
    variable = 0). Feeding thousands of identical points to PhenoGraph makes
    it cut them into arbitrary pieces (their k-nearest neighbours are a
    random choice among ties), so we cluster the DISTINCT profiles instead
    and map every patient to its profile's cluster:
      * >= config.PHENOGRAPH_MIN_UNIQUE_PROFILES distinct profiles -> PhenoGraph
        on the distinct profiles;
      * fewer than that -> each distinct profile is its own group (there's
        nothing for graph clustering to discover among a handful of points).
    Clusters with fewer patients than PHENOGRAPH_MIN_CLUSTER_FRACTION of the
    input are labelled -1. Returns (labels, modularity Q, method, n_profiles).
    """
    import contextlib, io
    uniq, inverse = np.unique(np.round(X, 4), axis=0, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    n_u = len(uniq)
    if n_u >= config.PHENOGRAPH_MIN_UNIQUE_PROFILES:
        with contextlib.redirect_stdout(io.StringIO()):   # PhenoGraph is very chatty
            lab_u, _, Q = phenograph.cluster(uniq, k=min(config.PHENOGRAPH_K, n_u - 1), seed=seed, n_jobs=1)
        lab_u, method = np.asarray(lab_u), "PhenoGraph (Louvain) on distinct patient profiles"
    else:
        lab_u, Q, method = np.arange(n_u), np.nan, "exact-profile grouping (too few distinct profiles for graph clustering)"
    labels = lab_u[inverse]
    min_size = max(2, int(config.PHENOGRAPH_MIN_CLUSTER_FRACTION * len(X)))
    sizes = pd.Series(labels).value_counts()
    small = sizes[sizes < min_size].index
    labels = np.where(np.isin(labels, small) | (labels < 0), -1, labels)
    keep = pd.Series(labels[labels >= 0]).value_counts().index.tolist()   # relabel: largest = 0
    remap = {old: new for new, old in enumerate(keep)}
    return np.array([remap.get(l, -1) for l in labels]), float(Q), method, n_u


def quality(X, labels, seed):
    m = labels >= 0
    out = {"n_clusters": int(len(set(labels[m]))), "n_outlier_patients_in_sample": int((~m).sum()),
           "silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    if out["n_clusters"] >= 2:
        rng = np.random.RandomState(seed)
        idx = np.where(m)[0]
        sub = rng.choice(idx, min(config.SILHOUETTE_SAMPLE_SIZE, len(idx)), replace=False)
        out["silhouette"] = float(silhouette_score(X[sub], labels[sub]))
        out["davies_bouldin"] = float(davies_bouldin_score(X[m], labels[m]))
        out["calinski_harabasz"] = float(calinski_harabasz_score(X[m], labels[m]))
    return out


def stability(X, ref_labels, analysis):
    rows = []
    rng = np.random.RandomState(config.RANDOM_SEED + 1)
    for r in range(config.CLUSTER_STABILITY_N_RUNS):
        idx = rng.choice(len(X), int(config.CLUSTER_STABILITY_SUBSAMPLE * len(X)), replace=False)
        lab = run_phenograph(X[idx], seed=config.RANDOM_SEED + 100 + r)[0]
        ref = ref_labels[idx]
        rows.append({"analysis": analysis, "run": r + 1, "n_patients": len(idx), "n_clusters": int(len(set(lab[lab >= 0]))),
                     "ARI_vs_reference": float(adjusted_rand_score(ref, lab)),
                     "NMI_vs_reference": float(normalized_mutual_info_score(ref, lab))})
    return rows


def assign_all(X_sample, labels, X_all):
    m = labels >= 0
    knn = KNeighborsClassifier(n_neighbors=15, n_jobs=1).fit(X_sample[m], labels[m])
    return knn.predict(X_all)


PALETTE = [plt.get_cmap(n)(i) for n in ("tab20", "tab20b", "tab20c") for i in range(20)]


def scatter_clusters(ax, emb, labels, title):
    """UMAP scatter coloured by cluster, with each cluster's number written at its centre."""
    out = labels < 0
    if out.any():
        ax.scatter(emb[out, 0], emb[out, 1], s=1, c="lightgrey", alpha=0.4, label="outlier / tiny cluster (-1)")
    for k in sorted(set(labels[~out])):
        m = labels == k
        ax.scatter(emb[m, 0], emb[m, 1], s=1, color=PALETTE[k % len(PALETTE)], alpha=0.6)
        cx, cy = np.median(emb[m, 0]), np.median(emb[m, 1])
        ax.text(cx, cy, str(k), fontsize=7, weight="bold", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    if out.any():
        ax.legend(markerscale=8, fontsize=7, loc="best")


def analyse(name, sample, full, cols):
    """Scale, cluster, embed, score, stability-test, and assign every patient. Returns a dict of results."""
    t0 = time.time()
    Xs, scaler, med = prepare(sample, cols)
    rng = np.random.RandomState(config.RANDOM_SEED)
    Xs_j = Xs + rng.normal(0, config.PHENOGRAPH_JITTER_STD, Xs.shape)   # display-only jitter for UMAP
    labels, Q, method, n_profiles = run_phenograph(Xs, seed=config.RANDOM_SEED)
    q = quality(Xs, labels, config.RANDOM_SEED)
    if not method.startswith("PhenoGraph"):
        # patients inside an exact-profile group are identical, so silhouette/DB/CH are trivially "perfect"
        q.update({"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan})
    q.update({"analysis": name, "features": ", ".join(cols), "modularity_Q": Q, "fit_sample_size": len(sample),
              "method": method, "distinct_profiles_in_sample": n_profiles})
    emb = umap.UMAP(n_neighbors=config.UMAP_N_NEIGHBORS, min_dist=config.UMAP_MIN_DIST, metric=config.UMAP_METRIC,
                    random_state=config.UMAP_RANDOM_STATE).fit_transform(Xs_j)
    stab = stability(Xs, labels, name)
    Xall, _, _ = prepare(full, cols, scaler, med)
    all_labels = assign_all(Xs, labels, Xall)
    utils.log(f"  {name}: {q['n_clusters']} clusters, Q={Q:.3f}, silhouette={q['silhouette']:.3f}, "
              f"ARI={np.mean([s['ARI_vs_reference'] for s in stab]):.3f}, "
              f"NMI={np.mean([s['NMI_vs_reference'] for s in stab]):.3f} ({time.time()-t0:.0f}s)")
    return {"labels": labels, "emb": emb, "quality": q, "stability": stab, "all_labels": all_labels, "cols": cols}


def main():
    utils.log("=" * 70)
    utils.log("STEP 7: POOLED 3-YEAR PHENOGRAPH + UMAP")
    utils.log("=" * 70)
    full, source = mu.load_scored_with_fallback()
    full["OPYEAR"] = full["OPYEAR"].astype(str)
    utils.log(f"Input: {source} -- {len(full):,} patients (all patients; outcomes not required)")
    sample = pd.concat([g.sample(n=min(config.PHENOGRAPH_SAMPLE_PER_YEAR, len(g)), random_state=config.RANDOM_SEED)
                        for _, g in full.groupby("OPYEAR")], ignore_index=True)
    utils.log(f"Fit sample: {len(sample):,} ({sample['OPYEAR'].value_counts().sort_index().to_dict()})")

    results = {"TOTAL": analyse("TOTAL (4 domain scores)", sample, full, SCORE_COLS)}
    for d in DOMAINS:
        cols = [c for c in domain_feature_columns(d) if c in full.columns]
        results[d] = analyse(f"DOMAIN: {d}", sample, full, cols)

    # ---- assignments ----
    assign = full[["PATIENT_KEY", "CASEID", "OPYEAR"]].copy()
    assign["TOTAL_CLUSTER"] = results["TOTAL"]["all_labels"]
    for d in DOMAINS:
        assign[f"{d}_CLUSTER"] = results[d]["all_labels"]
    assign["IN_UMAP_FIT_SAMPLE"] = assign["PATIENT_KEY"].isin(set(sample["PATIENT_KEY"]))
    emb_df = pd.DataFrame({"PATIENT_KEY": sample["PATIENT_KEY"],
                           "UMAP1_TOTAL": results["TOTAL"]["emb"][:, 0], "UMAP2_TOTAL": results["TOTAL"]["emb"][:, 1]})
    assign = assign.merge(emb_df, on="PATIENT_KEY", how="left")
    assign.to_parquet(f"{config.OUTPUT_DIR}/07_cluster_assignments.parquet", index=False)

    # ---- tables ----
    qual = pd.DataFrame([r["quality"] for r in results.values()])
    qual = qual[["analysis", "method", "distinct_profiles_in_sample", "n_clusters", "n_outlier_patients_in_sample",
                 "modularity_Q", "silhouette", "davies_bouldin", "calinski_harabasz", "fit_sample_size", "features"]]
    stab = pd.DataFrame([s for r in results.values() for s in r["stability"]])
    stab_summary = stab.groupby("analysis")[["ARI_vs_reference", "NMI_vs_reference"]].agg(["mean", "std"]).round(4)
    stab_summary.columns = [f"{a}_{b}" for a, b in stab_summary.columns]
    qual = qual.merge(stab_summary.reset_index(), on="analysis", how="left")

    size_rows, year_rows = [], []
    label_cols = {"TOTAL": "TOTAL_CLUSTER", **{d: f"{d}_CLUSTER" for d in DOMAINS}}
    for key, col in label_cols.items():
        fit_sizes = pd.Series(results[key]["labels"]).value_counts()
        all_sizes = assign[col].value_counts()
        for k in sorted(set(all_sizes.index) | set(fit_sizes.index)):
            size_rows.append({"analysis": key, "cluster": int(k), "n_fit_sample": int(fit_sizes.get(k, 0)),
                              "n_all_patients": int(all_sizes.get(k, 0)),
                              "pct_all_patients": round(100 * all_sizes.get(k, 0) / len(assign), 3)})
        ct = pd.crosstab(assign[col], assign["OPYEAR"], normalize="columns").mul(100).round(2)
        for k, row in ct.iterrows():
            year_rows.append({"analysis": key, "cluster": int(k), **{f"pct_of_{y}": row[y] for y in ct.columns}})
    sizes, by_year = pd.DataFrame(size_rows), pd.DataFrame(year_rows)

    prof_src = full[["PATIENT_KEY"] + SCORE_COLS + ["TOTAL_PHENOTYPE_SCORE", "BASELINE_BMI", "BMI_PCT_CHANGE_30D",
                                                   "WEIGHT_CHANGE_30D_KG", "AGE__STD"]].merge(
        assign[["PATIENT_KEY", "TOTAL_CLUSTER"]], on="PATIENT_KEY")
    profiles = prof_src.groupby("TOTAL_CLUSTER").agg(
        n=("PATIENT_KEY", "size"), **{f"mean_{c}": (c, "mean") for c in SCORE_COLS + ["TOTAL_PHENOTYPE_SCORE", "BASELINE_BMI",
                                                                                      "BMI_PCT_CHANGE_30D", "WEIGHT_CHANGE_30D_KG", "AGE__STD"]}
    ).round(3).reset_index()

    umap_settings = pd.DataFrame([
        {"setting": "UMAP n_neighbors", "value": config.UMAP_N_NEIGHBORS},
        {"setting": "UMAP min_dist", "value": config.UMAP_MIN_DIST},
        {"setting": "UMAP metric", "value": config.UMAP_METRIC},
        {"setting": "UMAP random seed", "value": config.UMAP_RANDOM_STATE},
        {"setting": "UMAP n_components", "value": 2},
        {"setting": "UMAP fit sample", "value": f"{len(sample):,} patients ({config.PHENOGRAPH_SAMPLE_PER_YEAR:,} per year)"},
        {"setting": "UMAP role", "value": "visualization only -- clusters come from PhenoGraph, not from the UMAP layout"},
    ])
    preprocessing = pd.DataFrame([
        {"setting": "TOTAL features", "value": ", ".join(SCORE_COLS) + " (Optuna-weighted if Step 4 has run)"},
        *[{"setting": f"{d} features", "value": ", ".join(results[d]['cols'])} for d in DOMAINS],
        {"setting": "Missing values", "value": "median imputation (fit sample medians)"},
        {"setting": "Scaling", "value": "StandardScaler (z-score) fit on the fit sample, applied to all patients"},
        {"setting": "Identical patients", "value": "clustered as one distinct profile (see run_phenograph docstring); "
                                                   f"profiles rounded to 4 decimals; PhenoGraph needs >= {config.PHENOGRAPH_MIN_UNIQUE_PROFILES} distinct profiles"},
        {"setting": "UMAP display jitter", "value": f"Gaussian noise, SD {config.PHENOGRAPH_JITTER_STD} (standardized units) -- layout only, not clustering"},
        {"setting": "PhenoGraph k", "value": config.PHENOGRAPH_K},
        {"setting": "PhenoGraph clustering", "value": "Louvain on the Jaccard-weighted kNN graph (phenograph.cluster)"},
        {"setting": "Min cluster size", "value": f"{100*config.PHENOGRAPH_MIN_CLUSTER_FRACTION:.1f}% of fit sample; smaller -> -1"},
        {"setting": "Assignment of other patients", "value": "15-nearest-neighbour classifier in the same scaled space"},
        {"setting": "Stability", "value": f"{config.CLUSTER_STABILITY_N_RUNS} re-runs on random {100*config.CLUSTER_STABILITY_SUBSAMPLE:.0f}% subsamples, ARI/NMI vs reference"},
        {"setting": "Input", "value": source},
    ])
    sheets = {"quality_metrics": qual, "stability_ARI_NMI": stab, "cluster_sizes": sizes, "cluster_by_year": by_year,
              "cluster_profiles_TOTAL": profiles, "umap_settings": umap_settings.astype(str),
              "preprocessing": preprocessing.astype(str)}
    per = assign if config.WRITE_FULL_PATIENT_EXCEL else assign.groupby("OPYEAR", group_keys=False).head(config.EXCEL_PREVIEW_ROWS)
    for yr, sub in per.groupby("OPYEAR"):
        sheets[f"assignments_{yr}"] = sub
    utils.save_excel_sheets(utils.excel_path("07_phenograph_clusters.xlsx"), sheets)
    utils.log("\n" + qual.drop(columns=["features", "method"]).round(3).to_string(index=False))

    # ---- figures ----
    emb = results["TOTAL"]["emb"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    scatter_clusters(axes[0], emb, results["TOTAL"]["labels"],
                     f"TOTAL phenotype clusters, 2017+2020+2024 pooled ({results['TOTAL']['quality']['n_clusters']} clusters)")
    for yr in sorted(sample["OPYEAR"].unique()):
        m = (sample["OPYEAR"] == yr).values
        axes[1].scatter(emb[m, 0], emb[m, 1], s=1, alpha=0.35, color=YEAR_COLORS.get(yr), label=yr)
    axes[1].set_title("Same embedding coloured by year (well mixed = same phenotypes every year)")
    axes[1].legend(markerscale=10); axes[1].set_xticks([]); axes[1].set_yticks([])
    fig.suptitle(f"PhenoGraph + UMAP on the 4 domain scores (n_neighbors={config.UMAP_N_NEIGHBORS}, "
                 f"min_dist={config.UMAP_MIN_DIST}, metric={config.UMAP_METRIC}, seed={config.UMAP_RANDOM_STATE})")
    fig.tight_layout()
    utils.save_fig(fig, "07_umap_total_3yr")

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    for ax, d in zip(axes.ravel(), DOMAINS):
        r = results[d]
        if r["quality"]["method"].startswith("PhenoGraph"):
            sub = f"PhenoGraph, silhouette {r['quality']['silhouette']:.2f}"
        else:
            sub = f"only {r['quality']['distinct_profiles_in_sample']} distinct profiles -> grouped by exact profile"
        scatter_clusters(ax, r["emb"], r["labels"], f"{d}: {r['quality']['n_clusters']} clusters ({sub})")
    fig.suptitle("Domain-wise PhenoGraph clustering, all patients 2017+2020+2024 pooled\n"
                 "(each domain clustered and embedded on its own variables' severity scores)")
    fig.tight_layout()
    utils.save_fig(fig, "07_umap_domainwise_3yr")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, c in zip(axes.ravel(), SCORE_COLS):
        sc = ax.scatter(emb[:, 0], emb[:, 1], s=1, c=sample[c], cmap="viridis", alpha=0.6)
        fig.colorbar(sc, ax=ax, label="score (0-100)")
        ax.set_title(c.replace("_SCORE", " score"))
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Domain score gradients over the TOTAL embedding (3 years pooled)")
    fig.tight_layout()
    utils.save_fig(fig, "07_umap_total_domain_gradients")

    fig, axes = plt.subplots(1, 5, figsize=(24, 4.5))
    for ax, (key, col) in zip(axes, label_cols.items()):
        s = sizes[(sizes.analysis == key) & (sizes.cluster >= 0)].sort_values("cluster")
        ax.bar(s["cluster"].astype(str), s["pct_all_patients"], color="darkcyan")
        ax.set_title(f"{key}: {len(s)} clusters")
        ax.set_xlabel("cluster"); ax.set_ylabel("% of all patients")
        ax.tick_params(axis="x", labelsize=6, rotation=90)
    fig.suptitle("Cluster size distribution (all 546k patients after assignment)")
    fig.tight_layout()
    utils.save_fig(fig, "07_cluster_size_distribution")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    names = list(stab["analysis"].unique())
    x = np.arange(len(names))
    for i, metric in enumerate(["ARI_vs_reference", "NMI_vs_reference"]):
        g = stab.groupby("analysis")[metric]
        ax.bar(x + i * 0.38, g.mean().reindex(names), yerr=g.std().reindex(names), width=0.38, capsize=4,
               label=metric.split("_")[0])
    ax.set_xticks(x + 0.19); ax.set_xticklabels(names, rotation=15, fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("agreement with reference clustering")
    ax.set_title(f"PhenoGraph stability across {config.CLUSTER_STABILITY_N_RUNS} subsampled re-runs")
    ax.legend()
    utils.save_fig(fig, "07_cluster_stability")
    utils.log("Step 7 complete.")


if __name__ == "__main__":
    main()
