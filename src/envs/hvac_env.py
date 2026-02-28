import gymnasium as gym
from gymnasium import spaces
import numpy as np

class HVACEnv(gym.Env):
    """
    Custom Gymnasium-compatible environment modeling simplified HVAC dynamics.
    State (5D):
      [indoor_temp, outdoor_temp, time_of_day, occupancy, price_signal]
    Actions (Discrete 3):
      0 = decrease output
      1 = maintain
      2 = increase output
    """

    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None):
        super().__init__()
        self.rng = np.random.default_rng(seed)

        # Observation bounds for each state variable
        low = np.array([10.0, -10.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([35.0, 50.0, 23.0, 100.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.action_space = spaces.Discrete(3)

        # Comfort range target for indoor temperature
        self.comfort_low = 21.0
        self.comfort_high = 24.0

        # Internal state 
        self.state = None
        self.step_count = 0
        self.max_steps = 96  #  96 steps would = 15-min increments in a day

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        indoor_temp = self.rng.uniform(20.5, 24.5)
        outdoor_temp = self.rng.uniform(10.0, 35.0)
        time_of_day = float(self.rng.integers(0, 24))
        occupancy = float(self.rng.integers(0, 101))
        price = self.rng.uniform(0.0, 1.0)

        self.state = np.array([indoor_temp, outdoor_temp, time_of_day, occupancy, price], dtype=np.float32)
        self.step_count = 0
        return self.state, {}

    def step(self, action: int):
        indoor, outdoor, tod, occ, price = self.state
        self.step_count += 1

        # Simple HVAC effect model:
        # action -> delta that nudges indoor temp
        if action == 0:       # decrease output (less cooling/heating)
            hvac_delta = 0.15
        elif action == 1:     # maintain
            hvac_delta = 0.0
        else:                 # increase output (more cooling/heating)
            hvac_delta = -0.15

        # Drift toward outdoor temp plus noise
        drift = 0.05 * (outdoor - indoor)
        noise = self.rng.normal(0.0, 0.05)
        indoor_next = indoor + drift + hvac_delta + noise

        # Update time/occupancy/price (toy dynamics)
        tod_next = (tod + 0.25) % 24.0  # 15-min increments
        occ_next = float(np.clip(occ + self.rng.integers(-5, 6), 0, 100))
        price_next = float(np.clip(price + self.rng.normal(0.0, 0.05), 0.0, 1.0))

        next_state = np.array([indoor_next, outdoor, tod_next, occ_next, price_next], dtype=np.float32)

        # Reward shaping: energy cost + comfort penalty
        # Energy proxy: more "increase output" costs more, scaled by price signal
        energy_cost = (1.0 if action == 2 else 0.3 if action == 1 else 0.1) * (0.5 + price_next)

        comfort_penalty = 0.0
        if indoor_next < self.comfort_low:
            comfort_penalty = self.comfort_low - indoor_next
        elif indoor_next > self.comfort_high:
            comfort_penalty = indoor_next - self.comfort_high

        reward = -(energy_cost + 2.0 * comfort_penalty)

        terminated = False
        truncated = self.step_count >= self.max_steps

        self.state = next_state
        info = {"energy_cost": float(energy_cost), "comfort_penalty": float(comfort_penalty)}
        return next_state, float(reward), terminated, truncated, info