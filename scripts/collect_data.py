import csv
from stable_baselines3 import PPO
from src.envs.hvac_env import HVACEnv

def main():
    env = HVACEnv(seed=123)

    try:
        model = PPO.load("ppo_hvac_model", env=env)
    except:
        # If no saved model, quickly train one
        model = PPO("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=20000)

    steps_to_collect = 10000
    output_file = "data/trajectories.csv"

    obs, _ = env.reset()

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "s_indoor", "s_outdoor", "s_tod", "s_occ", "s_price",
            "action", "reward",
            "s2_indoor", "s2_outdoor", "s2_tod", "s2_occ", "s2_price",
            "done"
        ])

        for _ in range(steps_to_collect):
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated

            writer.writerow([
                obs[0], obs[1], obs[2], obs[3], obs[4],
                int(action), float(reward),
                next_obs[0], next_obs[1], next_obs[2], next_obs[3], next_obs[4],
                int(done)
            ])

            obs = next_obs
            if done:
                obs, _ = env.reset()

    print(f"Wrote {steps_to_collect} steps to {output_file}")

if __name__ == "__main__":
    main()
