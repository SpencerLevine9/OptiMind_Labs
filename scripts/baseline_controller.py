import numpy as np


class RuleBasedBaseline:
    """
    Simple rule-based HVAC baseline.

    Assumes:
        0 = low effort toward comfort
        1 = maintain
        2 = high effort toward comfort
    """

    def __init__(self, comfort_band=1.0):
        self.comfort_band = comfort_band
        self.setpoint = 22.5  # matches HVACEnv comfort_center

    def predict(self, obs):
        indoor_temp = float(obs[0])

        if indoor_temp > self.setpoint + self.comfort_band:
            action = 2  # strong correction
        elif indoor_temp < self.setpoint - self.comfort_band:
            action = 2  # strong correction
        else:
            action = 1  # maintain

        return action