"""Comparing pink action noise with the default noise on SAC."""

from dm_control import suite
import numpy as np
import torch
from pink import PinkNoiseDist
from pink import ColoredNoiseDist
from stable_baselines3 import SAC
import time

# Reproducibility
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
rng = np.random.default_rng(seed)

# Initialize environment
env = suite.load(domain_name="cartpole", task_name="swingup")
action_dim = env.action_spec().shape[-1]
seq_len = 1000
rng = np.random.default_rng(0)

# Initialize agents
model_default = SAC("MlpPolicy", env, seed=seed)
model_pink = SAC("MlpPolicy", env, seed=seed)
model_colored = SAC("MlpPolicy", env, seed=seed)

# Set action noise
model_pink.actor.action_dist = PinkNoiseDist(seq_len, action_dim, rng=rng)
model_colored.actor.action_dist = ColoredNoiseDist(1.75, seq_len, action_dim, rng=rng)

# Train agents
t1 = time.time()
model_default.learn(total_timesteps=10_000)
t2 = time.time()
print(f"Time taken (Default Model): {t2-t1 :.2f} seconds")
t1 = time.time()
model_pink.learn(total_timesteps=10_000)
t2 = time.time()
print(f"Time taken (Pink Noise Model): {t2-t1 :.2f} seconds")
t1 = time.time()
model_colored.learn(total_timesteps=10_000)
t2 = time.time()
print(f"Time taken (Colored Noise Model): {t2-t1 :.2f} seconds")
# Evaluate learned policies
N = 100
for name, model in zip(["Default noise\n-------------", "Pink noise\n----------", "Colored noise\n--------------"], [
        model_default, 
                                                                        model_pink, model_colored]):
    solved = 0
    for i in range(N):
        time_step = env.reset()
        stp_typ, r, dis, obs = time_step
        # print(obs)
        done = False
        steps = 0
        while not time_step.last() and steps < 10_00:
            time_step = env.step(model.predict(obs, deterministic=True)[0])
            stp_typ, r, dis, obs = time_step
            steps += 1
            if r > 0.5:
                solved += 1
                break

    print(name)
    print(f"Solved: {solved/N * 100:.0f}%\n")


# - Output of this program -
# Default noise
# -------------
# Solved: 0%
#
# Pink noise
# ----------
# Solved: 100%
