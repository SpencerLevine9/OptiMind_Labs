# OptiMind Labs

Modern commercial buildings consume large amounts of energy, with HVAC systems accounting for a significant portion of total electricity usage. Inefficient energy control leads to high operational costs and unnecessary environmental impact, while overly aggressive cost reduction can compromise occupant comfort.

This project proposes a deep reinforcement learning–based intelligent agent that learns to optimally control energy usage in a simulated commercial building environment. The agent dynamically adjusts HVAC power levels by observing environmental conditions such as indoor temperature, time of day, and energy pricing. The objective is to minimize energy cost while maintaining acceptable thermal comfort.

By modeling energy management as a sequential decision-making problem, this project demonstrates how deep reinforcement learning can outperform traditional rule-based control strategies. The expected outcome is a trained RL agent that achieves lower energy costs while maintaining comfort, supported by quantitative evaluation and visual analysis. The project follows an industry-standard deep learning workflow including data generation, preprocessing, model training, hyperparameter tuning, and evaluation.

## Preprocessing

Run from the project root:

```bash
python scripts/preprocess.py
```

This writes **`data/random_trajectories_processed.csv`** (required). If **`data/ppo_trajectories.csv`** exists, it also writes **`data/ppo_trajectories_processed.csv`**. Use `--norm minmax` for min–max scaling instead of the default z-score, and optional `--reward-clip-low`, `--reward-clip-high`, or `--reward-scale` for reward handling.