# Generate the “before model” dataset (random policy)
# Because the instructions explicitly say:
# “For RL projects: plot … from initial random-policy rollouts.”
# Runs the HVAC env with a random action policy
# Saves data/random_trajectories.csv
# Same column format as your current data/trajectories.csv
# Output file: data/random_trajectories.csv

import csv
import os
from src.envs.hvac_env import HVACEnv


def main():
    env = HVACEnv(seed=123)

    os.makedirs("data", exist_ok=True)

    steps_to_collect = 10000
    output_file = "data/random_trajectories.csv"

    obs, _ = env.reset()
    episode_id = 0
    t_in_episode = 0

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode_id", "t",
            "s_indoor", "s_outdoor", "s_tod", "s_occ", "s_price",
            "action", "reward",
            "s2_indoor", "s2_outdoor", "s2_tod", "s2_occ", "s2_price",
            "done"
        ])

        for _ in range(steps_to_collect):
            action = env.action_space.sample()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            writer.writerow([
                episode_id, t_in_episode,
                obs[0], obs[1], obs[2], obs[3], obs[4],
                int(action), float(reward),
                next_obs[0], next_obs[1], next_obs[2], next_obs[3], next_obs[4],
                int(done)
            ])

            obs = next_obs
            t_in_episode += 1

            if done:
                obs, _ = env.reset()
                episode_id += 1
                t_in_episode = 0

    print(f"Wrote {steps_to_collect} random-policy steps to {output_file}")


if __name__ == "__main__":
    main()