"""Content-grouped CV, paired-bootstrap, and cross-fitted residualization utilities shared by every stage
decomposition / mechanism-analysis script and by the reviewer-validation extension. Single source of truth --
moved here (not duplicated) once a second consumer outside scripts/ needed them. No statistical procedure here
was changed by the move; see git history for provenance (originally in scripts/stage_decomposition_analysis.py
and scripts/vae_curvature_redundancy_analysis.py)."""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

C = 0.1
REPS = 10
FOLDS = 5
SEED = 0
N_BOOT = 1000


def make():
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced", random_state=17))


def sub(d, gen):
    x = d[(d.label == 0) | (d.generator == gen)]
    ok = x.groupby("content_id").label.nunique()
    return x[x.content_id.isin(ok[ok == 2].index)].reset_index(drop=True)


def cohen_paired(x, feat):
    pw = x.pivot_table(index="content_id", columns="label", values=feat)
    diff = (pw[1] - pw[0]).dropna()
    return float(diff.mean() / (diff.std(ddof=1) + 1e-12)), diff


def boot_ci(vals, stat_fn, n=N_BOOT, seed=0):
    arr = [stat_fn(np.random.default_rng(seed + 1 + i).choice(vals, len(vals), replace=True)) for i in range(n)]
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def grouped_cv_auc(x, cols, reps=REPS, folds=FOLDS, seed=0):
    X = np.nan_to_num(x[cols].values); y = x.label.values; ids = np.array(sorted(x.content_id.unique()))
    oof = np.zeros(len(x)); aucs = []
    for r in range(reps):
        perm = np.random.default_rng(seed + r).permutation(ids)
        fo = x.content_id.map({c: i % folds for i, c in enumerate(perm)}).values
        p = np.zeros(len(x))
        for k in range(folds):
            te = fo == k
            p[te] = make().fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y, p)); oof += p / reps
    return float(np.mean(aucs)), oof


def content_boot_ci(x, score, n=N_BOOT, seed=0):
    ids = np.array(sorted(x.content_id.unique())); by = {c: np.where(x.content_id.values == c)[0] for c in ids}
    y = x.label.values; rng = np.random.default_rng(seed)
    vals = [roc_auc_score(y[ix], score[ix]) for ix in (np.concatenate([by[c] for c in rng.choice(ids, len(ids))]) for _ in range(n))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_grouped_cv(x, cols_a, cols_b, reps=REPS, folds=FOLDS, seed=0):
    X_a = np.nan_to_num(x[cols_a].values); X_b = np.nan_to_num(x[cols_b].values); y = x.label.values
    ids = np.array(sorted(x.content_id.unique())); oof_a = np.zeros(len(x)); oof_b = np.zeros(len(x)); aucs_a = []; aucs_b = []
    for r in range(reps):
        perm = np.random.default_rng(seed + r).permutation(ids)
        fo = x.content_id.map({c: i % folds for i, c in enumerate(perm)}).values
        pa = np.zeros(len(x)); pb = np.zeros(len(x))
        for k in range(folds):
            te = fo == k
            pa[te] = make().fit(X_a[~te], y[~te]).predict_proba(X_a[te])[:, 1]
            pb[te] = make().fit(X_b[~te], y[~te]).predict_proba(X_b[te])[:, 1]
        aucs_a.append(roc_auc_score(y, pa)); aucs_b.append(roc_auc_score(y, pb)); oof_a += pa / reps; oof_b += pb / reps
    return float(np.mean(aucs_a)), float(np.mean(aucs_b)), oof_a, oof_b


def paired_diff_ci(x, oof_a, oof_b, n=N_BOOT, seed=0):
    ids = np.array(sorted(x.content_id.unique())); by = {c: np.where(x.content_id.values == c)[0] for c in ids}
    y = x.label.values; rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        ix = np.concatenate([by[c] for c in rng.choice(ids, len(ids))])
        diffs.append(roc_auc_score(y[ix], oof_b[ix]) - roc_auc_score(y[ix], oof_a[ix]))
    return float(np.mean(diffs)), (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def fold_assignment(x, r, folds=FOLDS, seed=SEED):
    ids = np.array(sorted(x.content_id.unique()))
    perm = np.random.default_rng(seed + r).permutation(ids)
    return x.content_id.map({c: i % folds for i, c in enumerate(perm)}).values


def cross_fitted_residualize(x, C_obs, predictor_matrix, alpha=1.0, reps=REPS, folds=FOLDS, seed=SEED):
    n = len(x)
    pred_matrix = np.full((reps, n), np.nan)
    for r in range(reps):
        fo = fold_assignment(x, r, folds, seed)
        for k in range(folds):
            te = fo == k; tr = ~te
            scaler = StandardScaler().fit(predictor_matrix[tr])
            Xtr = scaler.transform(predictor_matrix[tr]); Xte = scaler.transform(predictor_matrix[te])
            ridge = Ridge(alpha=alpha).fit(Xtr, C_obs[tr])
            pred_matrix[r, te] = ridge.predict(Xte)
    resid_matrix = C_obs[None, :] - pred_matrix
    return pred_matrix, resid_matrix, np.nanmean(pred_matrix, axis=0), C_obs - np.nanmean(pred_matrix, axis=0)


def cohen_paired_array(x, values):
    xx = x.copy(); xx["_val"] = values
    pw = xx.pivot_table(index="content_id", columns="label", values="_val")
    diff = (pw[1] - pw[0]).dropna()
    return float(diff.mean() / (diff.std(ddof=1) + 1e-12)), diff


def cv_oof_probs(x, cols, reps=REPS, folds=FOLDS):
    X = np.nan_to_num(x[cols].values); y = x.label.values; n = len(x)
    oof = np.zeros(n)
    for r in range(reps):
        fo = fold_assignment(x, r, folds)
        p = np.zeros(n)
        for k in range(folds):
            te = fo == k
            p[te] = make().fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
        oof += p / reps
    return oof


def metric_val(metric, y, p):
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    if metric == "auroc": return roc_auc_score(y, p)
    if metric == "logloss": return log_loss(y, pc, labels=[0, 1])
    if metric == "brier": return brier_score_loss(y, p)


def paired_metric_diff_ci(x, oof_a, oof_b, metric, n=N_BOOT, seed=0):
    ids = np.array(sorted(x.content_id.unique())); by = {c: np.where(x.content_id.values == c)[0] for c in ids}
    y = x.label.values; rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        ix = np.concatenate([by[c] for c in rng.choice(ids, len(ids))])
        diffs.append(metric_val(metric, y[ix], oof_b[ix]) - metric_val(metric, y[ix], oof_a[ix]))
    return float(np.mean(diffs)), (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def paired_probe_delta_effect_ci(ids, diffs_a, diffs_b, n=N_BOOT, seed=0):
    """Bootstrap the difference in paired Cohen's d between two measuring instruments (e.g. two probes) applied
    to the SAME content ids, using one shared resample per draw so the comparison is genuinely paired rather
    than inferred from two separate, independently-resampled CIs. diffs_a/diffs_b: pandas Series of per-content
    (fake - real) differences, indexed by content_id (the second return value of cohen_paired), for the same
    generator under instrument A and instrument B respectively. Returns (mean delta_d, 95% CI) where
    delta_d = d_b - d_a."""
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n):
        resampled = rng.choice(ids, len(ids), replace=True)
        da = diffs_a.loc[resampled].values; db = diffs_b.loc[resampled].values
        d_a = da.mean() / (da.std(ddof=1) + 1e-12)
        d_b = db.mean() / (db.std(ddof=1) + 1e-12)
        deltas.append(d_b - d_a)
    return float(np.mean(deltas)), (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))


def paired_probe_delta_auroc_ci(x_a, oof_a_base, oof_a_ext, x_b, oof_b_base, oof_b_ext, n=N_BOOT, seed=0):
    """Bootstrap the difference, across two measuring instruments, of the incremental AUROC gain from adding one
    feature on top of a base feature set -- i.e. does an instrument swap change how much incremental predictive
    value a feature contributes, not just its raw effect size. x_a/x_b: dataframes (content_id, label) for
    instrument A/B, covering the same content ids. oof_*_base/ext: out-of-fold predictions (already fit once via
    grouped_cv_auc/paired_grouped_cv, not refit per bootstrap draw) aligned to x_a's/x_b's row order. One shared
    content-id resample per draw. Returns (mean delta_delta_AUROC, 95% CI) where
    delta_delta_AUROC = (AUROC_b_ext - AUROC_b_base) - (AUROC_a_ext - AUROC_a_base)."""
    ids = np.array(sorted(x_a.content_id.unique()))
    by_a = {c: np.where(x_a.content_id.values == c)[0] for c in ids}
    by_b = {c: np.where(x_b.content_id.values == c)[0] for c in ids}
    y_a = x_a.label.values; y_b = x_b.label.values
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n):
        resampled = rng.choice(ids, len(ids), replace=True)
        ix_a = np.concatenate([by_a[c] for c in resampled]); ix_b = np.concatenate([by_b[c] for c in resampled])
        delta_a = roc_auc_score(y_a[ix_a], oof_a_ext[ix_a]) - roc_auc_score(y_a[ix_a], oof_a_base[ix_a])
        delta_b = roc_auc_score(y_b[ix_b], oof_b_ext[ix_b]) - roc_auc_score(y_b[ix_b], oof_b_base[ix_b])
        deltas.append(delta_b - delta_a)
    return float(np.mean(deltas)), (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))
