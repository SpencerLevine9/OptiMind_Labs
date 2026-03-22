from pathlib import Path
import sys

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

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
        verbose=1
    )

    model.learn(total_timesteps=100_000)

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model.save(models_dir / "ppo_hvac")

if __name__ == "__main__":
    main()