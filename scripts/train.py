from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.envs.hvac_env import HVACEnv

def make_env():
    return HVACEnv(seed=42)

def main():

    env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs"
    )

    model.learn(total_timesteps=100_000)

    model.save("ppo_hvac_model")

if __name__ == "__main__":
    main()