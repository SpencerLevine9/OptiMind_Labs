import os
import sys
import csv
from pathlib import Path

import numpy as np
import pandas as pd

# Stable-Baselines3 PPO
from stable_baselines3 import PPO

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "scripts"))

from src.envs.hvac_env import HVACEnv
from baseline_controller import RuleBasedBaseline


RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def safe_reset(env):
    """
    Works with both Gym and Gymnasium reset styles.
    """
    out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out


def safe_step(env, action):
    """
    Works with both Gym and Gymnasium step styles.
    Returns: obs, reward, done, info
    """
    out = env.step(action)

    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = terminated or truncated
        return obs, reward, done, info

    elif len(out) == 4:
        obs, reward, done, info = out
        return obs, reward, done, info

    else:
        raise ValueError("Unexpected env.step() return format.")


def run_policy(env, policy, n_episodes=10, is_sb3_model=False):
    """
    Runs either:
    - a Stable-Baselines3 PPO model, or
    - a baseline policy object with .predict(obs)
    """

    episode_rewards = []
    episode_energy_costs = []
    episode_violation_rates = []

    for ep in range(n_episodes):
        obs = safe_reset(env)
        done = False

        total_reward = 0.0
        total_energy_cost = 0.0
        total_violations = 0
        total_steps = 0

        while not done:
            if is_sb3_model:
                action, _ = policy.predict(obs, deterministic=True)
            else:
                action = policy.predict(obs)

            obs, reward, done, info = safe_step(env, action)

            total_reward += float(reward)
            total_energy_cost += float(info.get("energy_cost", 0.0))
            total_violations += int(info.get("comfort_violation", 0))
            total_steps += 1

        violation_rate = (total_violations / total_steps) if total_steps > 0 else 0.0

        episode_rewards.append(total_reward)
        episode_energy_costs.append(total_energy_cost)
        episode_violation_rates.append(violation_rate)

    metrics = {
        "avg_total_reward": float(np.mean(episode_rewards)),
        "std_total_reward": float(np.std(episode_rewards)),
        "avg_energy_cost": float(np.mean(episode_energy_costs)),
        "std_energy_cost": float(np.std(episode_energy_costs)),
        "comfort_violation_rate": float(np.mean(episode_violation_rates)),
        "episodes": int(n_episodes),
    }

    return metrics


def format_percent(x):
    return f"{100.0 * x:.2f}%"


def save_results(results_dict):
    rows = []
    for method_name, metrics in results_dict.items():
        rows.append({
            "method": method_name,
            "avg_total_reward": round(metrics["avg_total_reward"], 4),
            "std_total_reward": round(metrics["std_total_reward"], 4),
            "avg_energy_cost": round(metrics["avg_energy_cost"], 4),
            "std_energy_cost": round(metrics["std_energy_cost"], 4),
            "comfort_violation_rate": round(metrics["comfort_violation_rate"], 4),
            "episodes": metrics["episodes"],
        })

    df = pd.DataFrame(rows)

    csv_path = RESULTS_DIR / "final_metrics.csv"
    df.to_csv(csv_path, index=False)

    md_path = RESULTS_DIR / "final_metrics.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Final Evaluation Metrics\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n")
        f.write("## Summary\n\n")

        if "PPO" in results_dict and "Baseline" in results_dict:
            ppo = results_dict["PPO"]
            base = results_dict["Baseline"]

            f.write(f"- PPO average total reward: **{ppo['avg_total_reward']:.4f}**\n")
            f.write(f"- Baseline average total reward: **{base['avg_total_reward']:.4f}**\n")
            f.write(f"- PPO average energy cost: **{ppo['avg_energy_cost']:.4f}**\n")
            f.write(f"- Baseline average energy cost: **{base['avg_energy_cost']:.4f}**\n")
            f.write(f"- PPO comfort violation rate: **{format_percent(ppo['comfort_violation_rate'])}**\n")
            f.write(f"- Baseline comfort violation rate: **{format_percent(base['comfort_violation_rate'])}**\n")

    return csv_path, md_path, df


def main():
    model_path = PROJECT_ROOT / "models" / "ppo_hvac.zip"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find PPO model at: {model_path}\n"
            "Update model_path in scripts/evaluate.py to match your saved model."
        )

    # Create fresh evaluation environments
    env_ppo = HVACEnv()
    env_baseline = HVACEnv()

    print("Loading PPO model...")
    ppo_model = PPO.load(str(model_path))

    print("Creating baseline controller...")
    baseline = RuleBasedBaseline(comfort_band=1.0)

    print("Running PPO evaluation...")
    ppo_metrics = run_policy(env_ppo, ppo_model, n_episodes=10, is_sb3_model=True)

    print("Running baseline evaluation...")
    baseline_metrics = run_policy(env_baseline, baseline, n_episodes=10, is_sb3_model=False)

    results = {
        "PPO": ppo_metrics,
        "Baseline": baseline_metrics,
    }

    csv_path, md_path, df = save_results(results)

    print("\n=== FINAL RESULTS ===")
    print(df.to_string(index=False))
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved Markdown: {md_path}")


if __name__ == "__main__":
    main()