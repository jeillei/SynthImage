"""Experiment 1 (reviewer validation): does the path_length real-vs-generated effect depend on supplying the
image's own correct generation caption to the frozen SD1.5 probe? See
docs/research_history/reviewer_validation/REVIEWER_VALIDATION_PLAN.md for the full preregistered protocol,
derangement definition, and decision rule -- frozen before this script was run.

Mode A (correct caption) is reused verbatim from results/stage_decomposition/panel_features_dit.json (already
committed, same protocol). Modes B (null) and C (shuffled) are computed once with the same frozen SD15Probe
(same steps/guidance/canonicalization -- only the text conditioning changes) and then persisted to the compact,
committed results/reviewer_validation/conditioning_features.csv -- every subsequent run (including
scripts/reproduce/final_analysis.py) reads that committed table and never needs a probe or a GPU again. Raw
per-image intermediate outputs go to results/reviewer_validation/cache/ (gitignored, resumable)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from synthimage.analysis.cv import cohen_paired, boot_ci, paired_grouped_cv, paired_diff_ci

CACHE = Path("results/reviewer_validation/cache"); CACHE.mkdir(parents=True, exist_ok=True)
OUT = Path("results/reviewer_validation"); PLOTS = OUT / "plots"; PLOTS.mkdir(parents=True, exist_ok=True)
MANIFEST = "data/caption_matched/manifest.csv"
CORRECT_SOURCE = "results/stage_decomposition/panel_features_dit.json"
FEATURES_TABLE = OUT / "conditioning_features.csv"
VAE_FEATS = ["lpips_ae", "pixel_mse_ae", "latent_mse_ae"]
GENERATORS = ["sd15", "sdxl", "pixart_dit", "amused"]  # primary: sd15, sdxl, pixart_dit; secondary: amused


def deranged_captions(manifest: pd.DataFrame) -> dict:
    """Single fixed 60-cycle over sorted content ids: shuffled_caption(id at i) = caption(id at (i+1) mod 60).
    Deterministic, one-to-one, no fixed point (shift=1 != 0 mod 60), fixed before any feature extraction."""
    caption_of = manifest.drop_duplicates("content_id").set_index("content_id").caption.to_dict()
    ids = sorted(caption_of)
    n = len(ids)
    return {ids[i]: caption_of[ids[(i + 1) % n]] for i in range(n)}


def extract_mode(manifest: pd.DataFrame, mode: str, shuffled: dict | None) -> list[dict]:
    """GPU-dependent. mode: 'null' or 'shuffled'. Resumable: skips rows already cached. Only runs if
    results/reviewer_validation/cache/panel_features_{mode}.json is incomplete -- once
    conditioning_features.csv is committed, nothing here needs to run again."""
    out_path = CACHE / f"panel_features_{mode}.json"
    done = json.loads(out_path.read_text()) if out_path.exists() else []
    have = {(r["generator"], r["content_id"]) for r in done}
    pending = [r for _, r in manifest.iterrows() if (r["generator"], r["content_id"]) not in have]
    print(f"[{mode}] {len(done)}/{len(manifest)} already done; {len(pending)} to extract")
    if not pending:
        return done
    from synthimage.data.loading import load
    from synthimage.probes.sd15 import SD15Probe
    from synthimage.features.panel_v2 import path_length, diffpath_curvature, score_norm_step0
    probe = SD15Probe(steps=6)
    for i, r in enumerate(pending, 1):
        x = load(r["path"], 256, 17)
        caption = "" if mode == "null" else shuffled[r["content_id"]]
        res = probe.invert_reconstruct(x, caption, mode=("null" if mode == "null" else "caption"),
                                        guidance=1.0, capture_predictions=True)
        feats = {"path_length": path_length(res["forward"]), "diffpath_curvature": diffpath_curvature(res["cond_scores"]),
                  "score_norm_step0": score_norm_step0(res["cond_scores"])}
        assert not any(v != v for v in feats.values()), f"NaN produced for {mode}:{r['generator']}:{r['content_id']}"
        done.append({"generator": r["generator"], "content_id": r["content_id"], "label": int(r["label"]), "features": feats})
        out_path.write_text(json.dumps(done, indent=1))
        if i % 20 == 0 or i == len(pending):
            print(f"[{mode}] {i}/{len(pending)}", flush=True)
    return done


def build_condition_frame(condition: str, rows: list[dict], vae_by_key: dict) -> pd.DataFrame:
    recs = []
    for r in rows:
        key = (r["generator"], r["content_id"])
        rec = {"condition": condition, "generator": r["generator"], "content_id": r["content_id"], "label": r["label"], **r["features"]}
        rec.update(vae_by_key[key])
        recs.append(rec)
    return pd.DataFrame(recs)


def sub(d, gen):
    x = d[(d.label == 0) | (d.generator == gen)]
    ok = x.groupby("content_id").label.nunique()
    return x[x.content_id.isin(ok[ok == 2].index)].reset_index(drop=True)


def build_features_table() -> pd.DataFrame:
    """Extraction stage: mode A reused verbatim, modes B/C from cache (running extract_mode first if the cache
    is incomplete). Writes the compact, committed feature table."""
    manifest = pd.read_csv(MANIFEST)
    shuffled = deranged_captions(manifest)

    correct_rows = json.load(open(CORRECT_SOURCE))
    vae_by_key = {(r["generator"], r["content_id"]): {f: r["features"][f] for f in VAE_FEATS} for r in correct_rows}
    correct_frame = pd.DataFrame([{"condition": "correct", "generator": r["generator"], "content_id": r["content_id"],
                                    "label": r["label"], "path_length": r["features"]["path_length"],
                                    "diffpath_curvature": r["features"]["diffpath_curvature"],
                                    "score_norm_step0": r["features"]["score_norm_step0"], **vae_by_key[(r["generator"], r["content_id"])]}
                                   for r in correct_rows])

    null_rows = extract_mode(manifest, "null", None)
    shuffled_rows = extract_mode(manifest, "shuffled", shuffled)
    null_frame = build_condition_frame("null", null_rows, vae_by_key)
    shuffled_frame = build_condition_frame("shuffled", shuffled_rows, vae_by_key)

    all_df = pd.concat([correct_frame, null_frame, shuffled_frame], ignore_index=True)
    assert set(all_df.groupby(["generator", "content_id"]).condition.nunique().unique()) == {3}, \
        "every generator/content_id pair must have all three conditions"
    all_df.to_csv(FEATURES_TABLE, index=False)
    return all_df


def analyze(all_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    d_correct_by_gen = {}
    for gen in GENERATORS:
        x = sub(all_df[all_df.condition == "correct"], gen)
        d_correct_by_gen[gen], _ = cohen_paired(x, "path_length")

    for gen in GENERATORS:
        for cond in ["correct", "null", "shuffled"]:
            x = sub(all_df[all_df.condition == cond], gen)
            d_val, diffs = cohen_paired(x, "path_length")
            lo, hi = boot_ci(diffs.values, lambda a: a.mean() / (a.std(ddof=1) + 1e-12))
            auc = roc_auc_score(x.label, x.path_length); auc_df = max(auc, 1 - auc)
            frac_match = float((np.sign(diffs.values) == np.sign(d_correct_by_gen[gen])).mean())

            auc_a, auc_b, oa, ob = paired_grouped_cv(x, VAE_FEATS, VAE_FEATS + ["path_length"])
            md_vae, ci_vae = paired_diff_ci(x, oa, ob)

            score_feats = VAE_FEATS + ["score_norm_step0"]
            auc_a2, auc_b2, oa2, ob2 = paired_grouped_cv(x, score_feats, score_feats + ["path_length"])
            md_score, ci_score = paired_diff_ci(x, oa2, ob2)

            rows.append({
                "generator": gen, "condition": cond, "n_pairs": x.content_id.nunique(),
                "paired_cohens_d": d_val, "d_ci_lo": lo, "d_ci_hi": hi,
                "univariate_auroc_direction_free": auc_df, "frac_matched_correct_direction": frac_match,
                "retention_vs_correct": float(d_val / d_correct_by_gen[gen]) if d_correct_by_gen[gen] != 0 else float("nan"),
                "auroc_vae": auc_a, "auroc_vae_plus_path_length": auc_b,
                "incremental_auroc_vae_path_length": md_vae, "incremental_ci_lo": ci_vae[0], "incremental_ci_hi": ci_vae[1],
                "auroc_vae_score": auc_a2, "auroc_vae_score_path_length": auc_b2,
                "incremental_auroc_vae_score_path_length": md_score,
                "incremental_score_ci_lo": ci_score[0], "incremental_score_ci_hi": ci_score[1],
            })
        print(gen, "done", flush=True)
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame) -> None:
    conds = ["correct", "null", "shuffled"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.25
    colors = {"correct": "#4477aa", "null": "#ee6677", "shuffled": "#ccbb44"}
    xi = np.arange(len(GENERATORS))
    for i, cond in enumerate(conds):
        s = summary[summary.condition == cond].set_index("generator").reindex(GENERATORS)
        off = (i - 1) * w
        ax.bar(xi + off, s.paired_cohens_d, width=w, label=cond, color=colors[cond],
               yerr=[s.paired_cohens_d - s.d_ci_lo, s.d_ci_hi - s.paired_cohens_d], capsize=3)
    ax.axhline(0, color="k", lw=.7); ax.set_xticks(xi); ax.set_xticklabels(GENERATORS)
    ax.set_ylabel("paired Cohen's d (path_length)"); ax.legend(title="conditioning")
    ax.set_title("path_length effect under correct / null / shuffled caption conditioning")
    fig.tight_layout(); fig.savefig(PLOTS / "conditioning_ablation.png", dpi=140); plt.close(fig)


def run_from_features(features_path: Path = FEATURES_TABLE) -> None:
    """No GPU, no probe, no model download: reads the committed compact feature table and reproduces the
    summary table and figure. This is the entry point scripts/reproduce/final_analysis.py calls."""
    # keep_default_na=False: the "null" conditioning-mode label is a literal string, not a missing value --
    # pandas' default na_values list otherwise silently turns it into NaN.
    all_df = pd.read_csv(features_path, keep_default_na=False)
    summary = analyze(all_df)
    summary.to_csv(OUT / "conditioning_summary.csv", index=False)
    plot(summary)
    print(summary.round(3).to_string(index=False))
    print("\nwrote", OUT / "conditioning_summary.csv", "and", PLOTS / "conditioning_ablation.png")


def main():
    """Full pipeline: extracts null/shuffled values if not already cached, writes the compact committed feature
    table, then runs the same GPU-free analysis as run_from_features. `--from-features` skips straight to the
    GPU-free analysis (used by scripts/reproduce/final_analysis.py, which must never touch a probe)."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-features", action="store_true",
                     help="skip extraction; analyze the already-committed conditioning_features.csv only")
    a = ap.parse_args()
    if not a.from_features:
        build_features_table()
    run_from_features()


if __name__ == "__main__":
    main()
