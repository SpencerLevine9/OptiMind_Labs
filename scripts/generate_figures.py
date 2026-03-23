from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = PROJECT_ROOT / "data" / "random_trajectories_processed.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find data file: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print("Loaded:", DATA_PATH)
    print("Columns:", df.columns.tolist())
    return df


def get_numeric_columns(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def save_distribution_plots(df):
    numeric_cols = get_numeric_columns(df)

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        plt.figure(figsize=(8, 5))
        plt.hist(series, bins=30)
        plt.title(f"Histogram - {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"hist_{col}.png", dpi=200)
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.boxplot(series, vert=True)
        plt.title(f"Box Plot - {col}")
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"box_{col}.png", dpi=200)
        plt.close()


def save_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        print("No numeric columns found for correlation heatmap.")
        return

    corr = numeric_df.corr()

    plt.figure(figsize=(11, 8))
    plt.imshow(corr, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=200)
    plt.close()


def save_sample_episode_plots(df):
    if "episode_id" not in df.columns or "t" not in df.columns:
        print("Skipping sample episode plots: missing episode_id or t.")
        return

    episode_ids = sorted(df["episode_id"].dropna().unique())
    if len(episode_ids) == 0:
        print("No episodes found.")
        return

    plot_cols = [c for c in ["s_indoor", "action", "reward"] if c in df.columns]

    for ep in episode_ids[:2]:
        ep_df = df[df["episode_id"] == ep].sort_values("t")

        for col in plot_cols:
            plt.figure(figsize=(9, 4))
            plt.plot(ep_df["t"], ep_df[col])
            plt.title(f"Episode {ep} - {col} over time")
            plt.xlabel("Time step")
            plt.ylabel(col)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"episode_{ep}_{col}_timeseries.png", dpi=200)
            plt.close()


def compute_pca_2d(X):
    X = np.asarray(X, dtype=float)
    X_mean = X.mean(axis=0, keepdims=True)
    X_centered = X - X_mean

    cov = np.cov(X_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)

    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    components = eigvecs[:, :2]

    return X_centered @ components


def save_dimensionality_reduction(df):
    state_cols = ["s_indoor", "s_outdoor", "s_tod", "s_occ", "s_price"]
    state_cols = [c for c in state_cols if c in df.columns]

    if len(state_cols) < 2:
        print("Skipping PCA plot: not enough state columns.")
        return

    X = df[state_cols].dropna().values
    if len(X) < 2:
        print("Skipping PCA plot: not enough rows.")
        return

    X_pca = compute_pca_2d(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], s=10, alpha=0.6)
    plt.title("PCA of State Space")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pca_state_space.png", dpi=200)
    plt.close()


def save_rl_specific_plots(df):
    if "reward" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.hist(df["reward"].dropna(), bins=30)
        plt.title("Reward Distribution")
        plt.xlabel("Reward")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "reward_distribution.png", dpi=200)
        plt.close()

    if "episode_id" in df.columns and "t" in df.columns:
        episode_lengths = df.groupby("episode_id")["t"].max() + 1

        plt.figure(figsize=(8, 5))
        plt.hist(episode_lengths.dropna(), bins=20)
        plt.title("Episode Length Distribution")
        plt.xlabel("Episode Length")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "episode_length_distribution.png", dpi=200)
        plt.close()

    if "s_indoor" in df.columns and "s_outdoor" in df.columns:
        plt.figure(figsize=(8, 6))
        plt.scatter(df["s_outdoor"], df["s_indoor"], s=10, alpha=0.5)
        plt.title("State-Space Coverage: Indoor vs Outdoor")
        plt.xlabel("Outdoor Temperature")
        plt.ylabel("Indoor Temperature")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "state_space_coverage_indoor_vs_outdoor.png", dpi=200)
        plt.close()


def save_eda_insights(df):
    insights = []

    numeric_cols = get_numeric_columns(df)
    insights.append(f"- Dataset contains {len(df)} rows and {len(df.columns)} columns.")
    insights.append(f"- Numeric columns analyzed: {', '.join(numeric_cols)}.")

    if "reward" in df.columns:
        insights.append(
            f"- Reward mean = {df['reward'].mean():.4f}, min = {df['reward'].min():.4f}, max = {df['reward'].max():.4f}."
        )

    if "s_indoor" in df.columns:
        insights.append(
            f"- Indoor temperature ranged from {df['s_indoor'].min():.2f} to {df['s_indoor'].max():.2f}."
        )

    if "s_outdoor" in df.columns:
        insights.append(
            f"- Outdoor temperature ranged from {df['s_outdoor'].min():.2f} to {df['s_outdoor'].max():.2f}."
        )

    if "action" in df.columns:
        action_counts = df["action"].value_counts().sort_index().to_dict()
        insights.append(f"- Action usage counts: {action_counts}.")

    if "episode_id" in df.columns and "t" in df.columns:
        episode_lengths = df.groupby("episode_id")["t"].max() + 1
        insights.append(f"- Average episode length = {episode_lengths.mean():.2f} steps.")

    output_path = FIGURES_DIR / "eda_insights.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("EDA Insights\n")
        f.write("===========\n\n")
        for line in insights:
            f.write(line + "\n")

    print(f"Saved insights: {output_path}")


def main():
    df = load_data()

    save_distribution_plots(df)
    save_correlation_heatmap(df)
    save_sample_episode_plots(df)
    save_dimensionality_reduction(df)
    save_rl_specific_plots(df)
    save_eda_insights(df)

    print(f"All figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()