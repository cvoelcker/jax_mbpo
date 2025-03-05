from typing import Dict

import gymnasium as gym
import numpy as np

from mbpo.data.dataset import Dataset


def evaluate(agent, env: gym.Env, num_episodes: int) -> Dict[str, float]:
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=num_episodes)
    for _ in range(num_episodes):
        (observation, _), done = env.reset(), False
        while not done:
            action = agent.eval_actions(observation)
            observation, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    return {"return": np.mean(env.return_queue), "length": np.mean(env.length_queue)}


def evaluate_log_prob(agent, dataset: Dataset, batch_size: int = 2048) -> float:
    num_iters = len(dataset) // batch_size
    total_log_prob = 0.0
    for j in range(num_iters):
        indx = np.arange(j * batch_size, (j + 1) * batch_size)
        batch = dataset.sample(batch_size, keys=("observations", "actions"), indx=indx)
        log_prob = agent.eval_log_probs(batch)
        total_log_prob += log_prob

    return total_log_prob / num_iters
