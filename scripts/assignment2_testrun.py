from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_step(module: str, extra_args: list[str] | None = None) -> None:
    cmd = [sys.executable, "-m", module]
    if extra_args:
        cmd.extend(extra_args)
    print(f"\n>>> Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def summarize_outputs() -> None:
    data_dir = PROJECT_ROOT / "data"
    figures_dir = PROJECT_ROOT / "figures"
    results_dir = PROJECT_ROOT / "results"

    expected_paths = [
        data_dir / "random_trajectories.csv",
        data_dir / "random_trajectories_processed.csv",
        results_dir / "dataset_profile.csv",
        figures_dir / "eda_insights.txt",
        figures_dir / "correlation_heatmap.png",
    ]

    print("\n=== OUTPUT CHECK ===")
    for path in expected_paths:
        status = "OK" if path.exists() else "MISSING"
        print(f"[{status}] {path.relative_to(PROJECT_ROOT)}")

    figure_count = len(list(figures_dir.glob("*"))) if figures_dir.exists() else 0
    print(f"Figures generated: {figure_count}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Assignment 2 reproducibility pipeline.")
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip scripts.preprocess and generate figures from raw random rollouts instead.",
    )
    parser.add_argument(
        "--preprocess-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra arguments passed through to scripts.preprocess.",
    )
    args = parser.parse_args()

    run_step("scripts.collect_random_data")

    if not args.skip_preprocess:
        run_step("scripts.preprocess", args.preprocess_args)

    run_step("scripts.data_profile")
    run_step("scripts.generate_figures")
    summarize_outputs()
    print("\nAssignment 2 pipeline finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
