"""
Preprocess trajectory CSV data: observation normalization and optional reward scaling/clipping.
Uses only the Python standard library (no numpy/pandas required).

Outputs processed CSVs for analysis (e.g. offline RL or behavior cloning).
"""
import argparse
import csv
import math
import os
import sys


# Observation column names (state and next-state)
OBS_COLS = ["s_indoor", "s_outdoor", "s_tod", "s_occ", "s_price"]
NEXT_OBS_COLS = ["s2_indoor", "s2_outdoor", "s2_tod", "s2_occ", "s2_price"]
REWARD_COL = "reward"


def _col_index(header: list[str], name: str) -> int | None:
    try:
        return header.index(name)
    except ValueError:
        return None


def _mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 1.0
    m = sum(values) / n
    if n == 1:
        return m, 1.0
    var = sum((x - m) ** 2 for x in values) / (n - 1)  # sample std
    s = math.sqrt(var)
    if s == 0.0:
        s = 1.0
    return m, s


def _min_max(values: list[float]) -> tuple[list[float], list[float]]:
    if not values:
        return [], []
    return [min(values)], [max(values)]


def _collect_obs_columns(rows: list[list[str]], header: list[str], obs_cols: list[str]) -> list[list[float]]:
    """Extract float columns for obs_cols from rows."""
    idxs = [_col_index(header, c) for c in obs_cols]
    if any(i is None for i in idxs):
        return []
    cols: list[list[float]] = [[] for _ in obs_cols]
    for row in rows:
        for j, i in enumerate(idxs):
            cols[j].append(float(row[i]))
    return cols


def normalize_rows(
    rows: list[list[str]],
    header: list[str],
    method: str,
    obs_cols: list[str],
    next_obs_cols: list[str],
) -> None:
    """Normalize observation columns in place in `rows` (mutates row lists)."""
    obs_idx = [_col_index(header, c) for c in obs_cols]
    next_idx = [_col_index(header, c) for c in next_obs_cols]
    if any(i is None for i in obs_idx):
        return

    # Fit on current-state columns only
    obs_matrix = _collect_obs_columns(rows, header, obs_cols)
    if not obs_matrix:
        return

    if method == "zscore":
        stats = [_mean_std(col) for col in obs_matrix]
        for row in rows:
            for j, i in enumerate(obs_idx):
                m, s = stats[j]
                row[i] = str((float(row[i]) - m) / s)
            if all(i is not None for i in next_idx):
                for j, i in enumerate(next_idx):
                    m, s = stats[j]
                    row[i] = str((float(row[i]) - m) / s)
    elif method == "minmax":
        mins = [min(col) for col in obs_matrix]
        maxs = [max(col) for col in obs_matrix]
        ranges = [maxs[j] - mins[j] if maxs[j] != mins[j] else 1.0 for j in range(len(obs_cols))]
        for row in rows:
            for j, i in enumerate(obs_idx):
                row[i] = str((float(row[i]) - mins[j]) / ranges[j])
            if all(i is not None for i in next_idx):
                for j, i in enumerate(next_idx):
                    row[i] = str((float(row[i]) - mins[j]) / ranges[j])
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def scale_rewards(rows: list[list[str]], header: list[str], clip_low, clip_high, scale) -> None:
    ri = _col_index(header, REWARD_COL)
    if ri is None:
        return
    for row in rows:
        r = float(row[ri])
        if clip_low is not None:
            r = max(r, clip_low)
        if clip_high is not None:
            r = min(r, clip_high)
        if scale is not None:
            r = r * scale
        row[ri] = str(r)


def preprocess_file(
    input_path: str,
    output_path: str,
    norm_method: str = "zscore",
    reward_clip_low: float | None = None,
    reward_clip_high: float | None = None,
    reward_scale: float | None = None,
) -> bool:
    """Load CSV, normalize observations, optionally scale/clip rewards, save processed CSV."""
    if not os.path.isfile(input_path):
        return False
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return False
        rows = [list(r) for r in reader]

    normalize_rows(rows, header, norm_method, OBS_COLS, NEXT_OBS_COLS)
    if reward_clip_low is not None or reward_clip_high is not None or reward_scale is not None:
        scale_rewards(rows, header, reward_clip_low, reward_clip_high, reward_scale)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess trajectory CSVs: normalize observations, optional reward scaling/clipping."
    )
    parser.add_argument(
        "--norm",
        choices=["zscore", "minmax"],
        default="zscore",
        help="Observation normalization: zscore or minmax (default: zscore)",
    )
    parser.add_argument(
        "--reward-clip-low",
        type=float,
        default=None,
        help="Optional: clip rewards below this value",
    )
    parser.add_argument(
        "--reward-clip-high",
        type=float,
        default=None,
        help="Optional: clip rewards above this value",
    )
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=None,
        help="Optional: scale rewards by this factor",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing input CSVs and where processed files are written (default: data)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    required_input = os.path.join(data_dir, "random_trajectories.csv")
    required_output = os.path.join(data_dir, "random_trajectories_processed.csv")
    optional_input = os.path.join(data_dir, "ppo_trajectories.csv")
    optional_output = os.path.join(data_dir, "ppo_trajectories_processed.csv")

    if preprocess_file(
        required_input,
        required_output,
        norm_method=args.norm,
        reward_clip_low=args.reward_clip_low,
        reward_clip_high=args.reward_clip_high,
        reward_scale=args.reward_scale,
    ):
        print(f"Wrote {required_output}")
    else:
        print(f"Input not found or empty: {required_input}", file=sys.stderr)
        return 1

    if preprocess_file(
        optional_input,
        optional_output,
        norm_method=args.norm,
        reward_clip_low=args.reward_clip_low,
        reward_clip_high=args.reward_clip_high,
        reward_scale=args.reward_scale,
    ):
        print(f"Wrote {optional_output}")
    else:
        print(f"(Skipped optional: {optional_input} not found)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
