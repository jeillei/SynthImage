# REVIEWER_VALIDATION_RESULTS

Reports both reviewer-validation experiments exactly as preregistered in
`REVIEWER_VALIDATION_PLAN.md`, using its decision rules and reporting negative results plainly. Written
incrementally: the Experiment 1 section was committed before Experiment 2 was run, and Experiment 2's conditioning
mode (correct caption) was fixed in the plan before Experiment 1's C1 result was known.

## Experiment 1 — conditioning ablation

**Result: C1 — conditioning-independent.** The `path_length` real-vs-generated effect for SD1.5 and SDXL is
essentially unchanged whether the SD1.5 probe is conditioned on the image's own correct caption, an empty
("null") caption, or a caption belonging to a different, deterministically deranged content id.

| generator | condition | paired Cohen's d | 95% CI | retention vs. correct |
|---|---|---:|---:|---:|
| sd15 | correct | −0.333 | [−0.551, −0.111] | 1.00 |
| sd15 | null | −0.324 | [−0.549, −0.103] | 0.975 |
| sd15 | shuffled | −0.327 | [−0.553, −0.101] | 0.982 |
| sdxl | correct | −0.358 | [−0.607, −0.105] | 1.00 |
| sdxl | null | −0.367 | [−0.622, −0.117] | 1.028 |
| sdxl | shuffled | −0.376 | [−0.628, −0.121] | 1.052 |
| pixart_dit (secondary) | correct | +0.380 | [0.155, 0.674] | 1.00 |
| pixart_dit (secondary) | null | +0.380 | [0.155, 0.670] | 0.998 |
| pixart_dit (secondary) | shuffled | +0.370 | [0.145, 0.661] | 0.973 |
| amused (secondary) | correct | +1.390 | [1.144, 1.793] | 1.00 |
| amused (secondary) | null | +1.393 | [1.144, 1.797] | 1.002 |
| amused (secondary) | shuffled | +1.389 | [1.142, 1.787] | 0.999 |

Retention ratios for every generator, under both null and shuffled conditioning, land within ±5% of 1.0 — far
inside the ±30% band the plan set as the threshold for "conditioning-independent." Full table (including
univariate AUROC and both incremental comparisons):
`results/reviewer_validation/conditioning_summary.csv`. Figure: `results/reviewer_validation/plots/conditioning_ablation.png`
(three conditioning bars per generator, visually indistinguishable).

**Incremental value.** `VAE+score` vs `VAE+score+path_length` (the more powered of the two preregistered
incremental comparisons) stays positive with a 95% CI excluding zero for SD1.5 and SDXL under **all three**
conditioning modes:

| generator | condition | Δ AUROC (VAE+score → +path_length) | 95% CI |
|---|---|---:|---:|
| sd15 | correct | +0.088 | [0.016, 0.156] |
| sd15 | null | +0.085 | [0.013, 0.154] |
| sd15 | shuffled | +0.082 | [0.011, 0.149] |
| sdxl | correct | +0.060 | [0.026, 0.100] |
| sdxl | null | +0.060 | [0.025, 0.100] |
| sdxl | shuffled | +0.060 | [0.025, 0.100] |

**One honest caveat.** The narrower `VAE` vs `VAE+path_length` comparison (no score features) for **SD1.5
specifically** has a 95% CI that includes zero under null (−0.003, 0.160) and shuffled (−0.004, 0.159)
conditioning, where it excluded zero under correct conditioning (0.001, 0.163) — barely, in both directions. The
point estimate itself barely moves (0.080 → 0.076 → 0.075), so this reads as a comparison sitting right at the
edge of what a bootstrap CI can resolve at n=60 rather than a real effect change; it is reported here rather
than smoothed over. SDXL's equivalent comparison stays clearly positive in all three conditions.

PixArt-Sigma's pattern (small positive `path_length` effect but a **negative**, CI-excluding-zero incremental
contribution — i.e. adding `path_length` on top of VAE+score measurably *hurts* AUROC for this generator) is
unchanged across conditioning modes, consistent with the v1.0 finding that trajectory information does not help
for PixArt-Sigma. aMUSEd's incremental comparison crosses zero in all three conditions (VAE alone already
separates aMUSEd almost perfectly), also unchanged.

**Interpretation.** The `path_length` effect for SD1.5- and SDXL-generated images is not primarily a
caption–image semantic-compatibility artifact: removing the correct caption entirely, or replacing it with a
caption that describes a different photograph, leaves the effect size and (with one narrow, borderline exception
for SD1.5) the incremental predictive value essentially intact. This directly addresses reviewer concern 1 and
strengthens the provenance interpretation of the v1.0 result.

## Experiment 2 — probe swap

**Result: P4 — mixed.** The generator-level *incremental-value* pattern is stable across the SD1.5 and SDXL
probes, but raw `path_length` effect *magnitude* is probe-dependent in a way that is not captured by a simple
symmetric probe-family-affinity model (P2) — the amplification under the SDXL probe is not mirrored by a
corresponding SD1.5-probe preference for its own family, and the generator most affected besides SDXL itself
(aMUSEd) shares no architecture family with either probe. This section reports the direct paired cross-probe
tests added after the initial (P2) classification was found to be imprecise; see "Statistical tightening" below.

**Terminology note:** SD1.5 and SDXL are both UNet latent-diffusion systems — a second, independently trained
SDXL diffusion probe from the same broad model family, not an architecturally independent probe. This experiment
establishes robustness to a substantial probe/checkpoint swap within that family, not to a fundamentally
different probe architecture.

| generator | SD1.5 probe (d, 95% CI) | SDXL probe (d, 95% CI) |
|---|---:|---:|
| sd15 | −0.333 [−0.551, −0.111] | −0.296 [−0.542, −0.063] |
| sdxl | −0.358 [−0.607, −0.105] | **−0.720 [−0.999, −0.495]** |
| pixart_dit (secondary) | +0.380 [0.155, 0.674] | +0.333 [0.101, 0.654] |
| amused (secondary) | +1.390 [1.144, 1.793] | +0.649 [0.414, 0.940] |

Figure: `results/reviewer_validation/plots/probe_swap_matrix.png`. Full table (including univariate AUROC and
both incremental comparisons): `results/reviewer_validation/probe_swap_summary.csv`.

### Direct paired cross-probe tests

The table above invites inferring probe-dependence from whether two *separately*-resampled confidence intervals
overlap — a well-known statistical trap (overlapping CIs do not imply a non-significant difference, and
non-overlapping CIs are not required for one). Both quantities below instead resample the same 60 content ids
once per bootstrap draw and apply both probes' data to that identical resample, directly testing the paired
difference (`synthimage.analysis.cv.paired_probe_delta_effect_ci` /
`paired_probe_delta_auroc_ci`; 1000 draws; committed in `results/reviewer_validation/probe_swap_summary.csv`).

**Δd — raw effect-size difference** (`d_SDXL_probe − d_SD1.5_probe`):

| generator | Δd | 95% CI | excludes zero? |
|---|---:|---:|:---:|
| sd15 | +0.036 | [−0.114, 0.176] | no |
| sdxl | **−0.370** | **[−0.572, −0.187]** | **yes** |
| pixart_dit (secondary) | −0.045 | [−0.238, 0.142] | no |
| amused (secondary) | **−0.750** | **[−1.026, −0.512]** | **yes** |

**ΔΔAUROC — incremental-value difference** (`ΔAUROC_SDXLprobe − ΔAUROC_SD1.5probe`, where each
`ΔAUROC = AUROC(VAE+score+path_length) − AUROC(VAE+score)`):

| generator | ΔΔAUROC | 95% CI | excludes zero? |
|---|---:|---:|:---:|
| sd15 | −0.022 | [−0.076, 0.031] | no |
| sdxl | +0.005 | [−0.038, 0.046] | no |
| pixart_dit (secondary) | −0.002 | [−0.014, 0.008] | no |
| amused (secondary) | −0.001 | [−0.014, 0.016] | no |

**Reading the two tests together is the key result of this experiment.** Raw effect-size magnitude is
significantly probe-dependent for exactly two of the four generators (sdxl, amused) and not the other two
(sd15, pixart_dit) — a pattern with no clean symmetric explanation: SDXL-generated images separate much better
under the SDXL probe (self-affinity), but aMUSEd — a masked-token model sharing no architecture lineage with
either probe — is affected just as strongly, and the SD1.5 probe shows no reciprocal preference for its own
family (Δd for sd15 is not significant). Meanwhile, **every ΔΔAUROC CI includes zero** — the incremental
predictive value `path_length` contributes on top of VAE+score is statistically indistinguishable between the
two probes for all four generators, including the two (sdxl, amused) whose raw effect size did shift.

### Interpretation

`path_length`'s raw effect magnitude is not probe-independent for at least sdxl- and amused-generated images —
real, directly-tested evidence responding to reviewer concern 2. But the decision-relevant claim — does adding
`path_length` to VAE+score measurably improve discrimination, and for which generators — is unchanged under a
probe/checkpoint swap within the UNet latent-diffusion family: positive for SD1.5- and SDXL-generated images,
negative (harmful) for PixArt-Sigma, absent for aMUSEd, and the *difference in that incremental value* between
probes is not statistically resolvable for any generator. This is reported as mixed (P4), not as a symmetric
probe-family-affinity reframing (P2): the affinity pattern that would justify P2 is present for one generator
pairing (SDXL probe / SDXL generator) but absent or reversed elsewhere, so no simple probe-family model explains
all four generators.

### Statistical tightening (post-hoc note)

An earlier version of this document classified this result as "P2 — probe-family affinity (partial)" based on
inspecting the two probes' separately-computed CIs. A follow-up pass added the direct paired tests above,
which showed (a) the incremental-value story is not merely "similar" but statistically indistinguishable between
probes for every generator, and (b) the raw-magnitude pattern is not the symmetric affinity signature a P2
label implies. The classification was corrected to P4 accordingly; this is a statistical clarification, not a
change in any underlying number.

## Combined interpretation and wording changes

Both stress tests leave the v1.0 result's *qualitative* claim — `path_length` carries incremental information
beyond VAE+score for SD1.5- and SDXL-generated images but not for PixArt-Sigma — intact under caption removal,
caption shuffling, and a second, independently trained SDXL probe from the same broad UNet latent-diffusion
family. Neither test found grounds to retract or substantially narrow that claim, and the direct paired
cross-probe test makes this precise rather than impressionistic: every `ΔΔAUROC` CI includes zero, i.e. the
incremental value the `path_length` result depends on is not statistically distinguishable between the two
probes for any generator. Experiment 2 does add one genuine, directly-tested refinement: `path_length`'s raw
effect *magnitude* (not its incremental value, and not its sign) is probe-dependent for the sdxl and amused
generators specifically (`Δd` CI excludes zero), in a pattern with no clean symmetric probe-family explanation
(P4, not P2) — so language implying the magnitude of the effect is an intrinsic, probe-independent property of
the image should be softened, and "architecturally independent probe" should not be used for an SDXL-vs-SD1.5
comparison. Docs updated accordingly: `docs/FINAL_RESULTS.md` §10 (Limitations) and §11 (Conclusion), and the
root `README.md` headline findings.

## Relation to recent literature

"Diffusion-trajectory forensic signal exists" is not itself a novel claim as of 2024–2026: recent work
independently confirms it using different methodology — **Denoising Trajectory Biases for Zero-Shot AI-Generated
Image Detection** (NeurIPS 2025) finds generated images converge faster than real ones under diffusion inversion
and generalizes zero-shot across 21 generators ([paper](https://openreview.net/pdf?id=2h8vXbEufN)); **LATTE**
(arXiv 2507.03054, 2025) learns an embedding over the latent trajectory across denoising steps for cross-generator
detection ([paper](https://arxiv.org/abs/2507.03054)); **DRCT** (ICML 2024) extends DIRE-style reconstruction with
contrastive training on hard reconstructed samples
([paper](https://proceedings.mlr.press/v235/chen24ay.html)); **FakeInversion** (CVPR 2024) uses inversion into a
text-to-image model's noise space as a detection feature for unseen generators
([paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Cazenavette_FakeInversion_Learning_to_Detect_Images_from_Unseen_Text-to-Image_Models_by_CVPR_2024_paper.pdf)).

None of these decompose which *stage* of a frozen, off-the-shelf probe adds incremental value over a simpler
reconstruction baseline, and none stress-test that specific incremental claim against a caption-conditioning
ablation or a cross-probe swap. **SynthImage's contribution is therefore not "diffusion trajectories carry
real/fake information"** (independently established by the above) **but a confound-controlled, stage-wise
decomposition of which frozen-probe signal survives after simpler reconstruction features are accounted for,
across four generator architectures, together with the conditioning and cross-probe stress tests reported in
this document.**

