"""Experiment 2 (reviewer validation): is the generator-dependent path_length effect stable across measuring
instruments, or does it follow probe-generator affinity? See
docs/research_history/reviewer_validation/REVIEWER_VALIDATION_PLAN.md for the full preregistered protocol,
the SDXL probe's exact configuration, and the decision rule -- frozen before this script was run.

Conditioning mode is fixed to mode A (correct caption, the standard v1.0 protocol) for both probes, chosen in
the plan before Experiment 1's results were known. SD1.5-probe values are reused verbatim from
results/stage_decomposition/panel_features_dit.json (no recompute). SDXL-probe values are computed once (from
the GPU-dependent path below) and then persisted to the compact, committed
results/reviewer_validation/probe_swap_features.csv -- every subsequent run (including
scripts/reproduce/final_analysis.py) reads that committed table and never needs a probe or a GPU again. Raw
per-image intermediate outputs go to results/reviewer_validation/cache/ (gitignored, resumable).

The direct cross-probe comparisons (delta_d, delta_delta_auroc) resample the same 60 content ids once per
bootstrap draw for both probes, rather than inferring probe-dependence from two independently-resampled CIs --
see synthimage.analysis.cv.paired_probe_delta_effect_ci / paired_probe_delta_auroc_ci."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from synthimage.analysis.cv import (cohen_paired, boot_ci, paired_grouped_cv, paired_diff_ci,
    paired_probe_delta_effect_ci, paired_probe_delta_auroc_ci)

CACHE = Path("results/reviewer_validation/cache"); CACHE.mkdir(parents=True, exist_ok=True)
OUT = Path("results/reviewer_validation"); PLOTS = OUT / "plots"; PLOTS.mkdir(parents=True, exist_ok=True)
MANIFEST = "data/caption_matched/manifest.csv"
SD15_PROBE_SOURCE = "results/stage_decomposition/panel_features_dit.json"
SDXL_CACHE_FILE = CACHE / "panel_features_sdxl_probe.json"
FEATURES_TABLE = OUT / "probe_swap_features.csv"
VAE_FEATS = ["lpips_ae", "pixel_mse_ae", "latent_mse_ae"]
FEATURE_NAMES = VAE_FEATS + ["score_norm_step0", "path_length", "diffpath_curvature"]
GENERATORS = ["sd15", "sdxl", "pixart_dit", "amused"]  # primary: sd15, sdxl, pixart_dit; secondary: amused


def extract_sdxl_probe(manifest: pd.DataFrame) -> list[dict]:
    """GPU-dependent. Only runs if results/reviewer_validation/cache/panel_features_sdxl_probe.json is
    incomplete -- once probe_swap_features.csv is committed, nothing here needs to run again."""
    done = json.loads(SDXL_CACHE_FILE.read_text()) if SDXL_CACHE_FILE.exists() else []
    have = {(r["generator"], r["content_id"]) for r in done}
    pending = [r for _, r in manifest.iterrows() if (r["generator"], r["content_id"]) not in have]
    print(f"[sdxl probe] {len(done)}/{len(manifest)} already done; {len(pending)} to extract")
    if not pending:
        return done
    import lpips
    from synthimage.data.loading import load
    from synthimage.probes.sdxl import SDXLProbe
    from synthimage.features.panel_v2 import path_length, diffpath_curvature, score_norm_step0, vae_only
    probe = SDXLProbe(steps=6)
    device = probe.device
    net = lpips.LPIPS(net="vgg").to(device).eval()
    for i, r in enumerate(pending, 1):
        x = load(r["path"], 256, 17)
        res = probe.invert_reconstruct(x, r["caption"], mode="caption", guidance=1.0, capture_predictions=True)
        feats = {"path_length": path_length(res["forward"]), "diffpath_curvature": diffpath_curvature(res["cond_scores"]),
                  "score_norm_step0": score_norm_step0(res["cond_scores"]), **vae_only(probe, x, net)}
        assert not any(v != v for v in feats.values()), f"NaN produced for sdxl-probe:{r['generator']}:{r['content_id']}"
        done.append({"generator": r["generator"], "content_id": r["content_id"], "label": int(r["label"]), "features": feats})
        SDXL_CACHE_FILE.write_text(json.dumps(done, indent=1))
        if i % 20 == 0 or i == len(pending):
            print(f"[sdxl probe] {i}/{len(pending)}", flush=True)
    return done


def sub(d, gen):
    x = d[(d.label == 0) | (d.generator == gen)]
    ok = x.groupby("content_id").label.nunique()
    return x[x.content_id.isin(ok[ok == 2].index)].reset_index(drop=True)


def to_frame(probe_name: str, rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"probe": probe_name, "generator": r["generator"], "content_id": r["content_id"],
                           "label": r["label"], **{f: r["features"][f] for f in FEATURE_NAMES}} for r in rows])


def build_features_table() -> pd.DataFrame:
    """Extraction stage: SD1.5-probe values reused verbatim, SDXL-probe values from cache (running
    extract_sdxl_probe first if the cache is incomplete). Writes the compact, committed feature table."""
    sd15_rows = json.load(open(SD15_PROBE_SOURCE))
    sd15_frame = to_frame("sd15", sd15_rows)

    manifest = pd.read_csv(MANIFEST)
    sdxl_rows = extract_sdxl_probe(manifest)
    sdxl_frame = to_frame("sdxl", sdxl_rows)

    all_df = pd.concat([sd15_frame, sdxl_frame], ignore_index=True)
    assert set(all_df.groupby(["generator", "content_id"]).probe.nunique().unique()) == {2}, \
        "every generator/content_id pair must have both probes"
    all_df.to_csv(FEATURES_TABLE, index=False)
    return all_df


def analyze(all_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per-probe, per-generator effect sizes and incremental AUROC. Returns the summary rows and, keyed by
    (probe, generator), the (dataframe, oof_base, oof_ext) needed by direct_cross_probe_tests so the
    VAE+score / VAE+score+path_length CV fit is never repeated."""
    rows = []; oof_store = {}
    for probe_name in ["sd15", "sdxl"]:
        for gen in GENERATORS:
            x = sub(all_df[all_df.probe == probe_name], gen)
            d_val, diffs = cohen_paired(x, "path_length")
            lo, hi = boot_ci(diffs.values, lambda a: a.mean() / (a.std(ddof=1) + 1e-12))
            auc = roc_auc_score(x.label, x.path_length); auc_df = max(auc, 1 - auc)

            auc_a, auc_b, oa, ob = paired_grouped_cv(x, VAE_FEATS, VAE_FEATS + ["path_length"])
            md_vae, ci_vae = paired_diff_ci(x, oa, ob)

            score_feats = VAE_FEATS + ["score_norm_step0"]
            auc_a2, auc_b2, oa2, ob2 = paired_grouped_cv(x, score_feats, score_feats + ["path_length"])
            md_score, ci_score = paired_diff_ci(x, oa2, ob2)
            oof_store[(probe_name, gen)] = (x, oa2, ob2)

            rows.append({
                "probe": probe_name, "generator": gen, "n_pairs": x.content_id.nunique(),
                "paired_cohens_d": d_val, "d_ci_lo": lo, "d_ci_hi": hi,
                "univariate_auroc_direction_free": auc_df,
                "auroc_vae": auc_a, "auroc_vae_plus_path_length": auc_b,
                "incremental_auroc_vae_path_length": md_vae, "incremental_ci_lo": ci_vae[0], "incremental_ci_hi": ci_vae[1],
                "auroc_vae_score": auc_a2, "auroc_vae_score_path_length": auc_b2,
                "incremental_auroc_vae_score_path_length": md_score,
                "incremental_score_ci_lo": ci_score[0], "incremental_score_ci_hi": ci_score[1],
            })
        print(probe_name, "done", flush=True)
    return pd.DataFrame(rows), oof_store


def direct_cross_probe_tests(all_df: pd.DataFrame, oof_store: dict) -> pd.DataFrame:
    """Direct paired bootstrap comparisons between the SD1.5 and SDXL probes, using one shared content-id
    resample per draw for both probes (not two separately-resampled CIs eyeballed for overlap). Reuses the
    VAE+score / VAE+score+path_length out-of-fold predictions already fit once in analyze()."""
    rows = []
    for gen in GENERATORS:
        x_sd15 = sub(all_df[all_df.probe == "sd15"], gen)
        x_sdxl = sub(all_df[all_df.probe == "sdxl"], gen)
        ids = np.array(sorted(x_sd15.content_id.unique()))
        assert ids.tolist() == sorted(x_sdxl.content_id.unique()), "probes must cover identical content ids"

        _, diffs_sd15 = cohen_paired(x_sd15, "path_length")
        _, diffs_sdxl = cohen_paired(x_sdxl, "path_length")
        delta_d, delta_d_ci = paired_probe_delta_effect_ci(ids, diffs_sd15, diffs_sdxl)

        x_a, oa_base, oa_ext = oof_store[("sd15", gen)]
        x_b, ob_base, ob_ext = oof_store[("sdxl", gen)]
        delta_delta_auroc, dd_ci = paired_probe_delta_auroc_ci(x_a, oa_base, oa_ext, x_b, ob_base, ob_ext)

        rows.append({"generator": gen, "delta_d_sdxl_minus_sd15": delta_d,
                     "delta_d_ci_lo": delta_d_ci[0], "delta_d_ci_hi": delta_d_ci[1],
                     "delta_delta_auroc_sdxl_minus_sd15": delta_delta_auroc,
                     "delta_delta_auroc_ci_lo": dd_ci[0], "delta_delta_auroc_ci_hi": dd_ci[1]})
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame) -> None:
    mat = summary.pivot(index="generator", columns="probe", values="paired_cohens_d").reindex(GENERATORS)[["sd15", "sdxl"]]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-np.abs(mat.values).max(), vmax=np.abs(mat.values).max())
    ax.set_xticks(range(2)); ax.set_xticklabels(["SD1.5 probe", "SDXL probe"])
    ax.set_yticks(range(len(GENERATORS))); ax.set_yticklabels(GENERATORS)
    for i in range(len(GENERATORS)):
        for j in range(2):
            ax.text(j, i, f"{mat.values[i, j]:.2f}", ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, label="paired Cohen's d (path_length)")
    ax.set_title("path_length effect: probe x generator")
    fig.tight_layout(); fig.savefig(PLOTS / "probe_swap_matrix.png", dpi=140); plt.close(fig)


def run_from_features(features_path: Path = FEATURES_TABLE) -> None:
    """No GPU, no probe, no model download: reads the committed compact feature table and reproduces the
    summary table, direct cross-probe tests, and figure. This is the entry point
    scripts/reproduce/final_analysis.py calls."""
    all_df = pd.read_csv(features_path)
    summary, oof_store = analyze(all_df)
    direct = direct_cross_probe_tests(all_df, oof_store)
    summary = summary.merge(direct, on="generator", how="left")
    summary.to_csv(OUT / "probe_swap_summary.csv", index=False)
    plot(summary)
    print(summary.round(4).to_string(index=False))
    print("\nwrote", OUT / "probe_swap_summary.csv", "and", PLOTS / "probe_swap_matrix.png")


def main():
    """Full pipeline: extracts SDXL-probe values if not already cached, writes the compact committed feature
    table, then runs the same GPU-free analysis as run_from_features. `--from-features` skips straight to the
    GPU-free analysis (used by scripts/reproduce/final_analysis.py, which must never touch a probe)."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-features", action="store_true",
                     help="skip extraction; analyze the already-committed probe_swap_features.csv only")
    a = ap.parse_args()
    if not a.from_features:
        build_features_table()
    run_from_features()


if __name__ == "__main__":
    main()
