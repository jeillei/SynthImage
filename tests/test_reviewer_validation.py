"""Scientific-correctness tests for the reviewer-validation extension (conditioning ablation + probe swap).
No GPU/model weights needed -- these test the derangement logic, the cross-probe interface contract, and the
paired cross-probe bootstrap machinery, not feature extraction itself. See
docs/research_history/reviewer_validation/REVIEWER_VALIDATION_PLAN.md."""
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.reviewer_validation.conditioning_ablation import deranged_captions, FEATURES_TABLE as CONDITIONING_FEATURES
from scripts.reviewer_validation.probe_swap import FEATURES_TABLE as PROBE_SWAP_FEATURES
from synthimage.probes.sd15 import SD15Probe
from synthimage.probes.sdxl import SDXLProbe
from synthimage.analysis.cv import paired_probe_delta_effect_ci, paired_probe_delta_auroc_ci


def _fake_manifest(n=60):
    return pd.DataFrame({"content_id": [f"c{i:03d}" for i in range(n)], "caption": [f"caption {i}" for i in range(n)]})


def test_derangement_is_a_bijection_with_no_fixed_point():
    m = _fake_manifest()
    shuffled = deranged_captions(m)
    caption_of = m.set_index("content_id").caption.to_dict()
    assert set(shuffled) == set(caption_of)  # every content id gets an assignment
    assert sorted(shuffled.values()) == sorted(caption_of.values())  # bijection: same multiset of captions
    for cid, cap in shuffled.items():
        assert cap != caption_of[cid], f"{cid} mapped to its own caption"


def test_derangement_is_deterministic_across_calls_and_row_order():
    m = _fake_manifest()
    a = deranged_captions(m)
    b = deranged_captions(m.sample(frac=1, random_state=7).reset_index(drop=True))  # shuffled row order
    assert a == b


def test_sd15_and_sdxl_probes_expose_the_same_measurement_interface():
    """path_length/diffpath_curvature/score_norm_step0/vae_only in panel_v2.py are written against one probe
    interface; a cross-probe comparison is only meaningful if both probes satisfy it identically."""
    required = {"encode_image", "decode_latent", "embeds", "invert_reconstruct"}
    assert required <= set(dir(SD15Probe))
    assert required <= set(dir(SDXLProbe))


def test_paired_probe_delta_effect_ci_uses_one_shared_resample_for_both_probes():
    """If probe A and B produced literally identical per-content diffs, delta_d must be exactly zero for every
    bootstrap draw -- that only holds if both probes are indexed by the SAME resampled content ids each draw.
    An implementation that drew two independent resamples would generally NOT get exactly zero every time."""
    ids = np.array([f"c{i:02d}" for i in range(30)])
    diffs = pd.Series(np.random.default_rng(0).normal(0, 1, size=30), index=ids)
    mean_delta, ci = paired_probe_delta_effect_ci(ids, diffs, diffs, n=200, seed=1)
    assert mean_delta == 0.0
    assert ci == (0.0, 0.0)


def test_paired_probe_delta_auroc_ci_uses_one_shared_resample_for_both_probes():
    """Same logic as above, for the incremental-AUROC-difference test: identical (x, oof) inputs for both
    probes must give exactly zero for every draw."""
    n = 30
    x = pd.DataFrame({"content_id": [f"c{i:02d}" for i in range(n)] * 2, "label": [0] * n + [1] * n})
    rng = np.random.default_rng(0)
    oof_base = rng.uniform(0, 1, size=2 * n)
    oof_ext = rng.uniform(0, 1, size=2 * n)
    mean_dd, ci = paired_probe_delta_auroc_ci(x, oof_base, oof_ext, x, oof_base, oof_ext, n=200, seed=1)
    assert mean_dd == 0.0
    assert ci == (0.0, 0.0)


def test_committed_feature_tables_have_expected_columns_and_complete_pairing():
    """The compact, committed feature tables scripts/reproduce/final_analysis.py reads (no GPU) must have every
    (generator, content_id) pairing complete across all three conditioning modes / both probes -- an incomplete
    row would silently corrupt the content-grouped pairing every analysis function relies on."""
    if not CONDITIONING_FEATURES.exists() or not PROBE_SWAP_FEATURES.exists():
        return  # nothing to check before the tables are first generated locally
    cond = pd.read_csv(CONDITIONING_FEATURES, keep_default_na=False)
    expected_cond_cols = {"condition", "generator", "content_id", "label", "path_length", "diffpath_curvature",
                          "score_norm_step0", "lpips_ae", "pixel_mse_ae", "latent_mse_ae"}
    assert expected_cond_cols <= set(cond.columns)
    assert set(cond.groupby(["generator", "content_id"]).condition.nunique().unique()) == {3}

    swap = pd.read_csv(PROBE_SWAP_FEATURES, keep_default_na=False)
    expected_swap_cols = {"probe", "generator", "content_id", "label", "path_length", "diffpath_curvature",
                          "score_norm_step0", "lpips_ae", "pixel_mse_ae", "latent_mse_ae"}
    assert expected_swap_cols <= set(swap.columns)
    assert set(swap.groupby(["generator", "content_id"]).probe.nunique().unique()) == {2}


def test_final_analysis_includes_reviewer_validation_stage_gpu_free():
    """scripts/reproduce/final_analysis.py must reproduce the v1.1 reviewer-validation results too, and must
    invoke them with --from-features so it never touches a probe or downloads a model."""
    from scripts.reproduce.final_analysis import STEPS
    rel_paths = {step[1]: step[2] for step in STEPS}
    assert rel_paths.get("scripts/reviewer_validation/conditioning_ablation.py") == ["--from-features"]
    assert rel_paths.get("scripts/reviewer_validation/probe_swap.py") == ["--from-features"]
