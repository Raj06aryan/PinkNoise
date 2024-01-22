import gymnasium as gym
import numpy as np
import torch
from pink import PinkNoiseDist
from pink import ColoredNoiseDist
from stable_baselines3 import SAC
import time

# Define a function to evaluate an episode
def evaluate_episode(model, env):
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    steps=0
    while steps<1000 and not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        steps+=1
    return total_reward

# Reproducibility
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
rng = np.random.default_rng(seed)

# Initialize environment
env = gym.make("MountainCarContinuous-v0")
action_dim = env.action_space.shape[-1]
seq_len = env._max_episode_steps
rng = np.random.default_rng(0)

# Initialize agents
model_default = SAC("MlpPolicy", env, seed=seed)
model_pink = SAC("MlpPolicy", env, seed=seed)
model_OU = SAC("MlpPolicy", env, seed=seed)

# Set action noise
model_pink.actor.action_dist = PinkNoiseDist(seq_len, action_dim, rng=rng)
model_OU.actor.action_dist = ColoredNoiseDist(beta=2, seq_len=seq_len, action_dim=action_dim, rng=rng)

# Training parameters
total_timesteps = 1000000
eval_frequency = 10000 # Evaluate every 104 interactions
eval_rollouts = 5

#Final average performances
avg_default=0.0
avg_pink=0.0
avg_OU=0.0
final_default=0.0
final_pink=0.0
final_OU=0.0

# Train agents with evaluation
timesteps_so_far = 0
while timesteps_so_far < total_timesteps:
    t1 = time.time()
    # Train the default noise model
    model_default.learn(total_timesteps=eval_frequency)
    t2 = time.time()

    # Evaluate the default noise model
    mean_return_default = 0.0
    for _ in range(eval_rollouts):
        mean_return_default += evaluate_episode(model_default, env)
    mean_return_default /= eval_rollouts
    avg_default+=mean_return_default
    if(timesteps_so_far>=0.95*total_timesteps):
        final_default+=mean_return_default

    print(f"Return (Default): {mean_return_default}")
    print(f"Time taken (Default Model): {t2 - t1:.2f} seconds")
    print(f"Timesteps: {timesteps_so_far}, Mean Return: {mean_return_default}")

    t1=time.time()
    # Train the pink noise model
    model_pink.learn(total_timesteps=eval_frequency)
    # timesteps_so_far += eval_frequency
    t2 = time.time()

    # Evaluate the pink noise model
    mean_return_pink = 0.0
    for _ in range(eval_rollouts):
        mean_return_pink += evaluate_episode(model_pink, env)
    mean_return_pink /= eval_rollouts
    avg_pink+=mean_return_pink
    if(timesteps_so_far>=0.95*total_timesteps):
        final_pink+=mean_return_pink

    print(f"Return (Pink): {mean_return_pink}")
    print(f"Time taken (Pink Noise Model): {t2 - t1:.2f} seconds")
    print(f"Timesteps: {timesteps_so_far}, Mean Return: {mean_return_pink}")

    t1=time.time()
    # Train the pink noise model
    model_OU.learn(total_timesteps=eval_frequency)
    # timesteps_so_far += eval_frequency
    t2 = time.time()
    
    # Evaluate the pink noise model
    mean_return_OU = 0.0
    for _ in range(eval_rollouts):
        mean_return_OU += evaluate_episode(model_OU, env)
    mean_return_OU/= eval_rollouts
    avg_OU+=mean_return_OU
    if(timesteps_so_far>=0.95*total_timesteps):
        final_OU+=mean_return_OU

    print(f"Return (OU): {mean_return_OU}")
    print(f"Time taken (OU Noise Model): {t2 - t1:.2f} seconds")
    print(f"Timesteps: {timesteps_so_far}, Mean Return: {mean_return_OU}")

    timesteps_so_far += eval_frequency

avg_default/=(total_timesteps/eval_frequency)
avg_pink/=(total_timesteps/eval_frequency)
avg_OU/=(total_timesteps/eval_frequency)

final_default/=(0.05*total_timesteps/eval_frequency)
final_pink/=(0.05*total_timesteps/eval_frequency)
final_OU/=(0.05*total_timesteps/eval_frequency)

print("Mean:")
print(f"White:{avg_default}           Pink:{avg_pink}             OU:{avg_OU}")
print("Final:")
print(f"White:{final_default}           Pink:{final_pink}             OU:{final_OU}")
