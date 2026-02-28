from stable_baselines3 import PPO
from src.envs.hvac_env import HVACEnv

def main():
    env = HVACEnv(seed=42)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50_000)
    model.save("ppo_hvac_model")

if __name__ == "__main__":
    main()