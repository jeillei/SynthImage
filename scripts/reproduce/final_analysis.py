"""The single entrypoint for reproducing SynthImage's headline results.

    uv run python scripts/reproduce/final_analysis.py

Runs entirely from committed, compact feature tables under results/ -- no GPU, no model download, no image
dataset. Reproduces: the four-generator stage decomposition, the VAE-curvature redundancy mechanism analysis,
the path-length mechanism analysis, the realistic-transformation/AI-edit final validation, and the v1.1
reviewer-validation extension (conditioning ablation + cross-probe swap, including the direct paired
cross-probe bootstrap tests) -- each with its matching figures. See docs/FINAL_RESULTS.md for the narrative
these numbers support, and docs/REPRODUCIBILITY.md for the (separate, GPU-requiring) path that regenerates the
underlying feature tables themselves from raw images.

Each step below is a thin wrapper around an existing, independently runnable analysis script -- this file adds
no new statistics, only sequences the already-frozen pipeline so a stranger has one obvious command to run."""
from __future__ import annotations
import subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

STEPS = [
    ("Stage decomposition across four generators", "scripts/dit_stage_decomposition_analysis.py", []),
    ("Stage decomposition figures", "scripts/dit_stage_decomposition_plots.py", []),
    ("VAE / curvature redundancy mechanism analysis", "scripts/vae_curvature_redundancy_analysis.py", []),
    ("VAE / curvature redundancy figures", "scripts/vae_curvature_redundancy_plots.py", []),
    ("path_length mechanism analysis", "scripts/path_length_mechanism_analysis.py", []),
    ("path_length mechanism figures", "scripts/path_length_mechanism_plots.py", []),
    ("Final validation: robustness + AI-edit continuum", "scripts/final_validation_analysis.py", []),
    ("Final validation figures", "scripts/final_validation_plots.py", []),
    ("Reviewer validation: conditioning ablation", "scripts/reviewer_validation/conditioning_ablation.py", ["--from-features"]),
    ("Reviewer validation: cross-probe swap + direct tests", "scripts/reviewer_validation/probe_swap.py", ["--from-features"]),
]


def main() -> None:
    print("SynthImage -- reproducing headline results from committed feature tables (no GPU, no model download)\n")
    t0 = time.time()
    for i, (label, rel_path, extra_args) in enumerate(STEPS, 1):
        print(f"[{i}/{len(STEPS)}] {label} ({rel_path})")
        result = subprocess.run([sys.executable, rel_path, *extra_args], cwd=ROOT)
        if result.returncode != 0:
            print(f"\nFAILED at step {i}: {rel_path} (exit code {result.returncode})", file=sys.stderr)
            sys.exit(result.returncode)
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. Headline tables and figures are under:")
    print("  results/stage_decomposition/dit_generator/   (generator comparison, incremental information)")
    print("  results/vae_curvature_redundancy/            (VAE/curvature mechanism)")
    print("  results/path_length_mechanism/               (path_length mechanism -- the decisive result)")
    print("  results/final_validation/                    (robustness + AI-edit continuum)")
    print("  results/reviewer_validation/                 (v1.1: conditioning ablation + cross-probe swap)")
    print("  results/summary/                              (the headline figures used in README.md)")
    print("\nSee docs/FINAL_RESULTS.md for how these numbers support the project's conclusions.")


if __name__ == "__main__":
    main()
