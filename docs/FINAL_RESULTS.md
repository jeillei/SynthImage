# FINAL_RESULTS — SynthImage

This document is the final scientific synthesis of the SynthImage project. It replaces the need to read the
full chronological research ledger (`docs/research_history/RESEARCH_REPORT.md`, 100+ entries) to understand what
was found. Methodology detail: `docs/METHODS.md`. Reproduction: `docs/REPRODUCIBILITY.md`. Two reviewer-motivated
stress tests of §7's `path_length` result (a caption-conditioning ablation and a cross-probe swap) were run after
this document was first written; both are folded into §10/§11 below, with full detail in
`docs/research_history/reviewer_validation/REVIEWER_VALIDATION_RESULTS.md`.

## 1. Research question

> Where does forensic information about an image's origin (real photograph vs. AI-generated) live inside a
> pretrained diffusion pipeline, how much does each computational stage add beyond the earlier ones, and which
> of those signals survive realistic image transformations, partial AI editing, and — critically — changes in
> generator architecture?

This is a narrower and more defensible question than SynthImage started with. The original hypothesis was that
running a pretrained diffusion model "in reverse" on an image would reveal a generator-general provenance
signal. That specific hypothesis was tested directly and falsified (§2) before the project moved to the
question above.

## 2. Why naive AI-image benchmarks are confounded

SynthImage's first large-scale benchmark produced strong-looking results (AUROC well above 0.8) that did not
survive scrutiny. A confounding audit found:

- **Four simple file-geometry numbers** (native resolution, file size, aspect ratio, and one compression
  statistic) beat the full 652-feature trajectory representation on the original protocol.
- **Caption text alone**, with no image information at all, separated real from synthetic images at AUROC 0.79.
- The trajectory representation could predict *which real-image dataset* an image came from — a dataset-source
  signature, not a provenance signature.

None of this means diffusion probes carry no forensic signal. It means a benchmark that lets real and generated
images differ in acquisition history, resolution, compression, or caption style cannot distinguish a genuine
provenance mechanism from these shortcuts. Full detail: `docs/research_history/SCIENTIFIC_AUDIT.md`.

## 3. Caption-matched experimental design

Every result after this point uses **60 real COCO photographs**, each paired with a counterpart from every
generator under test that was generated *from that photograph's human caption* — same caption, same
canonicalization (256×256, no crop), computed by the same frozen probe. **The generated image does not
reproduce the real photograph's actual scene**; it is an independent generation from the shared caption, so
"matched" here means caption-matched, not content-matched. Every classifier, cross-validation fold, and
bootstrap resample is grouped by **content id** (the shared caption/pairing key) — a real image and its
generated counterpart are never split across a fold. This single design change is what turned an apparently
strong but confounded benchmark into an honest, falsifiable measurement (`docs/METHODS.md` §4-5).

## 4. Literature-grounded stage decomposition

Rather than continue engineering new features, the project reset to a **10-feature panel**, each feature either
a direct reproduction of a published forensic method or a small, explicitly disclosed adaptation
(`docs/research_history/LITERATURE_FEATURE_PANEL.md`), organized into four pipeline stages:

`image → VAE reconstruction → score response → inverse trajectory → round-trip reconstruction`

The central question at every stage: **does this stage add information beyond every earlier stage**, tested with
paired out-of-fold AUROC differences and content-bootstrap confidence intervals, not just standalone accuracy.

## 5. Generator comparison

The same frozen protocol was applied, unmodified, to four generators against the same 60 real photographs:

| generator | architecture | VAE-alone AUROC | does trajectory add info beyond VAE+score? |
|---|---|---:|---|
| SD1.5 | UNet latent diffusion | 0.639 | **yes** (+0.125 AUROC, CI [0.056, 0.204]) |
| SDXL | UNet latent diffusion (larger, separate VAE) | 0.855 | **yes** (+0.053, CI [0.016, 0.092]) |
| PixArt-Sigma | Diffusion Transformer (no UNet) | 0.876 | **no** (−0.005, CI [−0.029, 0.018]) |
| aMUSEd | masked-token model (no diffusion process) | 0.981 | no (every stage beyond VAE has a CI crossing zero) |

Two findings stand out:
- **VAE reconstruction alone is already a strong, generator-dependent forensic signal** — it ranges from
  moderate (SD1.5) to near-ceiling (aMUSEd). This reconstruction is always performed by the **frozen SD1.5
  probe's own VAE** (encode → decode), applied identically regardless of which generator produced the image —
  never each generator's own decoder. The AUROC gradient therefore reflects how compatible each generator's
  output distribution is with *that one frozen VAE* specifically, which is itself evidence for a
  probe-compatibility effect rather than a decoder-quality property intrinsic to each generator in isolation.
- **The trajectory stage's contribution is architecture-dependent, not simply diffusion-vs-not-diffusion**: it
  helps both UNet-based diffusion generators tested (SD1.5, SDXL) but not the one Diffusion Transformer tested
  (PixArt-Sigma), despite PixArt-Sigma also being a genuine diffusion model. This ruled out the simplest version
  of the original hypothesis ("diffusion models generically show this") and motivated the mechanism analyses
  below. Two UNets and one DiT is enough to rule out the simplest hypothesis; it is not enough to support a
  class-general claim about UNets vs. DiTs in either direction.

Full detail: `docs/research_history/STAGE_DECOMPOSITION_RESULTS.md`,
`docs/research_history/PIXART_STAGE_DECOMPOSITION.md` (the SDXL run — see the naming note in `scripts/README.md`),
`docs/research_history/DIT_STAGE_DECOMPOSITION.md` (the genuine PixArt-Sigma DiT run).

## 6. VAE / trajectory redundancy analysis

Why does the trajectory stage help SD1.5/SDXL but not PixArt-Sigma, when the trajectory's headline feature
(`diffpath_curvature`) shows an almost identical raw effect size across all three diffusion generators
(d ≈ −0.53 to −0.67)? Cross-fitted, label-blind ridge residualization (regressing each trajectory feature on the
three VAE features, fit on training folds only, tested on held-out folds) answered this directly:

- `diffpath_curvature`'s real/fake signal is **substantially redundant with VAE-level information for both
  SDXL and PixArt-Sigma** (only 10–11% of its raw effect survives residualization, CI crossing zero for both) —
  but **not for SD1.5** (62% of its effect survives, CI still excluding zero). This ruled out the working
  hypothesis that curvature redundancy specifically tracked PixArt vs. the two UNet models.

Full detail: `docs/research_history/VAE_CURVATURE_REDUNDANCY.md`.

## 7. Path-length mechanism result

The trajectory stage's *other* frozen feature, `path_length`, resolved the SDXL-vs-PixArt discrepancy cleanly:

- **`path_length`'s VAE-redundancy is the inverse of curvature's.** For SD1.5 and SDXL, its real/fake effect
  *strengthens* after VAE-residualization (173%/186% retention, CI excluding zero) — for SDXL alone, its
  VAE-residualized value reaches AUROC 0.695. For PixArt-Sigma, its raw positive effect (a sign reversal
  relative to SD1.5/SDXL to begin with) collapses to non-significance under residualization.
- Decomposing SDXL's original trajectory-stage gain feature-by-feature shows **`path_length` alone reproduces
  essentially all of it** (+0.052 of the original +0.053 total AUROC gain); `diffpath_curvature`'s own marginal
  contribution is not distinguishable from zero anywhere in this decomposition, for either generator.
- A preregistered `curvature × path_length` interaction term was tested explicitly and added nothing for any
  generator, on any metric — ruling out a "joint trajectory geometry" or "score-conditioned interaction"
  explanation.

**Headline mechanistic finding**: *`path_length` provides predictive information beyond the three frozen VAE
features for SD1.5/SDXL specifically — not for the genuine Diffusion Transformer generator tested (PixArt-Sigma), and not attributable
to `diffpath_curvature` or an interaction between the two trajectory features.* This is the most specific,
decisive result in the project and the point at which feature-level mechanism exploration was deliberately
stopped (per its own preregistered decision rule). Full detail:
`docs/research_history/PATH_LENGTH_MECHANISM.md`.

## 8. Robustness (realistic transformations)

*Protocol frozen in `docs/research_history/FINAL_VALIDATION_PLAN.md` before this section was written. Full tables:
`results/final_validation/q1_feature_survival.csv`, `q2_incremental_survival.csv`,
`q3_clean_trained_transfer.csv`. Figure: `results/final_validation/plots/01_path_length_robustness.png`.*

2,520 transformed images (real/SD1.5/SDXL × 60 content ids × 14 conditions: JPEG 90/70/50/30, Gaussian blur
0.5/1.0/2.0, resize round-trip 0.5×/0.25×, Gaussian noise 0.02/0.05/0.10, one color-jitter condition, center crop)
were extracted through the frozen SD1.5-probe pipeline, plus aMUSEd/PixArt-Sigma as secondary comparisons.

**`path_length`'s predictive information beyond the VAE features survives realistic transformation almost universally.** Across the 14
transform conditions × 2 primary generators (28 condition/generator combinations, clean excluded from this
count since it trivially matches itself), `path_length`'s effect **survived** (CI excludes zero, same sign as
clean) in **27/28** (SDXL: 14/14; SD1.5: 13/14, only `center_crop_0.8` weakened to a CI crossing zero). In
several conditions — notably blur and noise — the effect **strengthens** relative to clean (e.g. SDXL blur
σ=2.0: d=−1.01 vs. clean d=−0.36; SD1.5 resize 0.25×: d=−0.39 vs. clean −0.33). `lpips_ae` and
`diffpath_curvature` show similarly strong retention for the two primary generators, with occasional
degradation at the most aggressive settings (e.g. SDXL `lpips_ae` under blur σ=2.0 and resize 0.25× degrades to
30–57% of its clean effect, still same-signed).

**The incremental-information result survives too, universally.** Repeating the primary VAE+score vs.
VAE+score+`path_length` AUROC comparison inside every transform condition: the CI-excludes-zero, positive
result held in **all 28/28 condition/generator combinations** (14 conditions × SD1.5/SDXL; ΔAUROC ranging
+0.024 to +0.162 for SDXL, i.e. it sometimes exceeds the clean-condition gain).

**Clean-trained deployment-style transfer is stable.** A model fit once on clean images only, frozen, and
applied unchanged to every transformed condition (no recalibration) stays close to its clean-condition AUROC
throughout: SD1.5 ranges 0.71–0.81 (clean: 0.76), SDXL ranges 0.89–0.95 (clean: 0.93) — log loss/Brier degrade
more noticeably at the most aggressive noise condition (σ=0.10) for both generators, consistent with calibration
drift under heavy distribution shift even where ranking (AUROC) holds up.

**Read together: `path_length`'s predictive information beyond the VAE features, for SD1.5/SDXL, is not a clean-data artifact.**
It survives realistic JPEG compression, blur, resizing, noise, color jitter, and cropping — both as a standalone
effect and as incremental classifier information — and a classifier trained only on clean images transfers with
only modest, expected degradation to every transformed condition tested.

## 9. AI-edit continuum

*Protocol frozen in `docs/research_history/FINAL_VALIDATION_PLAN.md` before this section was written. Full tables:
`results/final_validation/track_b_population_curves.csv`, `track_b_trend_test.csv`. Figure:
`results/final_validation/plots/02_ai_edit_response_comparison.png`.*

The 8-content img2img-strength pilot (`docs/research_history/STAGE_DECOMPOSITION_RESULTS.md` §Phase 7) was
scaled to all 60 content ids, same mechanism, strengths (0.0/0.3/0.6/0.9), steps, guidance, and seed policy.

| feature | pooled Spearman r (vs. strength) | fraction of contents monotonic | interpretation |
|---|---:|---:|---|
| `lpips_ae` (VAE) | **−0.62** (p<0.001) | 27% | strongest, most consistent population-level trend |
| `diffpath_curvature` | −0.60 (p<0.001) | 8% | strong pooled trend, but far less consistent per-content |
| `path_length` | +0.33 (p<0.001) | 5% | weakest, least consistent trend, opposite sign to curvature |

**The static/dynamical comparison is genuinely informative, not forced.** `lpips_ae` changes almost immediately
between strength 0.0 and 0.3 and then stays flat (saturates) — consistent with the "VAE changes immediately and
saturates" pattern the protocol asked about. `diffpath_curvature` is *non-monotonic* at the population level
(rises from strength 0 to 0.3, then falls); `path_length` is also non-monotonic (flat through 0.6, then rises
sharply at 0.9). Neither trajectory feature tracks edit strength as cleanly or consistently as the VAE signal
does. **This does not undermine §7's mechanism result** — that result is about real-vs-fake separation at fixed
generation, not about a monotonic response to a continuously-varying global img2img edit, which is a
qualitatively different intervention (whole-image re-diffusion, not the trajectory dynamics comparison the
mechanism analysis targeted). No claim is made that any feature's magnitude represents a calibrated "percent
AI" score — the fraction-monotonic figures above are reported plainly, including their weakness, not smoothed
into a stronger claim than the data supports.

## 10. Limitations

- All generator comparisons use **one frozen probe (SD1.5)** measuring the candidate generator's *output pixels*
  — the candidate generator itself never needs to be a diffusion model, but every measurement is filtered
  through SD1.5's own VAE/UNet, which is itself one specific, dated checkpoint. A reviewer-motivated cross-probe
  stress test (SD1.5 vs. a second, independently trained SDXL probe — both are UNet latent-diffusion systems,
  not architecturally independent probes) found `path_length`'s *raw effect magnitude* is probe-dependent for
  the sdxl and amused generators specifically (a direct paired bootstrap gives a 95% CI excluding zero for the
  probe-to-probe difference in effect size), in a pattern with no clean symmetric probe-family explanation. The
  *incremental*-value question — does `path_length` add predictive value beyond VAE+score, and for which
  generators — is not statistically distinguishable between the two probes for any generator (every direct
  ΔΔAUROC 95% CI includes zero). Full detail:
  `docs/research_history/reviewer_validation/REVIEWER_VALIDATION_RESULTS.md` (Experiment 2, classified P4 —
  mixed).
- Every generated image shares its exact generation caption with the caption later supplied to the probe during
  measurement, raising a caption-conditioning confound. A reviewer-motivated ablation (null and shuffled-caption
  conditioning) found the effect and its incremental value are essentially unchanged (retention 0.97–1.05 of the
  correct-caption effect) under both alternative conditioning modes for SD1.5 and SDXL — evidence against this
  being primarily a prompt-compatibility artifact. Full detail: same document (Experiment 1).
- The dataset is **60 caption-matched content ids** (real and generated images share a COCO caption, not a
  depicted scene) — enough for the paired-bootstrap statistics used throughout to be meaningful, not enough to
  support population-level detector performance claims.
- **`path_length`'s architecture-dependence is established for exactly one Diffusion Transformer (PixArt-Sigma)
  and two UNet models (SD1.5, SDXL)** — it is a real feature-level decomposition (§7 identifies which measured
  feature carries the incremental signal, and for which of these three generators), not a causal account of why
  those generators' outputs produce that path geometry, and not yet shown to generalize to DiT architectures in
  general.
- PixArt-Sigma's text encoder is a disclosed, community GGUF-quantized adaptation of the official T5-XXL weights
  (used only because the original fp32 weights exceeded available disk space during development) — the
  denoiser and VAE under test are official, unmodified weights; see
  `docs/research_history/PREREGISTRATION_DIT_GENERATOR_V1.md` for the full disclosure.
- No claim in this project should be read as a deployable, generator-independent, or adversarially robust AI-
  image detector. See §11.

## 11. Conclusion

Pretrained diffusion probes expose **multiple, architecture-dependent forensic signals at different
computational stages** of a latent-diffusion pipeline — not one universal signature. VAE-level reconstruction
carries real, generator-dependent signal on its own. The diffusion trajectory adds further information for the
two UNet-based diffusion generators tested but not for the one Diffusion Transformer tested. Within the trajectory stage
itself, that added information is attributable specifically to path geometry (`path_length`), not curvature,
for the generators where it appears at all. **This specific signal is realistic-transformation-robust**:
`path_length`'s effect and its incremental AUROC contribution beyond the VAE features both survived essentially every
tested JPEG/blur/resize/noise/color-jitter/crop condition for SD1.5 and SDXL (§8), and a classifier trained only
on clean images transferred with only modest degradation to every transformed condition. The AI-edit continuum
(§9) shows the VAE signal tracks increasing edit strength far more monotonically than either trajectory feature
— a genuine, unforced difference between the static and dynamic mechanisms, not evidence against the §7 result
(the two measure different things: fixed-generation separation vs. continuous-intervention response).

**This project does not claim**: a universal AI-image detector, generator-independent deployment performance,
that raw diffusion-path curvature alone proves diffusion provenance, or that any AI-edit-strength response
represents a calibrated "percent AI" score.

**Post-v1.0 reviewer validation.** Two alternative explanations for the `path_length` result — that it reflects
caption–image compatibility rather than provenance, and that it reflects SD1.5-probe-specific affinity rather
than an image-intrinsic property — were tested directly after this document was first written (§10, and in full
in `docs/research_history/reviewer_validation/REVIEWER_VALIDATION_RESULTS.md`). Neither test found grounds to
retract the qualitative claim above: it survives removing or scrambling the conditioning caption, and it survives
substituting a second, independently trained SDXL probe from the same broad UNet latent-diffusion family (not an
architecturally independent probe — that stress test remains to be done). A direct paired bootstrap shows
`path_length`'s raw effect *magnitude* — not its sign, and not its incremental value, which is statistically
indistinguishable between the two probes for every generator tested — is probe-dependent for the sdxl and
amused generators specifically, in a pattern with no clean symmetric probe-family explanation (classified P4 —
mixed, not a probe-family-affinity reframing). It is therefore described here as a **generator-dependent
trajectory signal under a frozen diffusion probe**, not an instrument-independent image property.

## Future work

Not part of active SynthImage experimentation, which is closed. Recorded here as possible directions for a
future, separate project:
- Test whether `path_length`'s VAE-independence generalizes across a broader sample of Diffusion Transformer
  checkpoints (not just PixArt-Sigma) and UNet checkpoints (not just SD1.5/SDXL).
- A calibrated, held-out-generator study of whether any frozen feature's response to AI-edit strength could
  support a genuine "degree of intervention" estimate, with proper calibration — not attempted here.
- Extending the frozen probe itself (e.g. a more modern base checkpoint) — deliberately out of scope for this
  project, which measured what one fixed, well-understood instrument reveals.
