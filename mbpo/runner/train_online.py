#! /usr/bin/env pythonA
import gymnasium as gym
import tqdm
import wandb
import hydra
from omegaconf import OmegaConf
from mbpo.algos.model_learning.model_learner import ModelLearner

from mbpo.algos.sac import SACLearner
from mbpo.data import ReplayBuffer
from mbpo.evaluation import evaluate

from mbpo.env_utils.termination_fns import lookup_termination_fn


@hydra.main(config_path="../../config", config_name="main")
def main(cfg):
    print(cfg)
    wandb.init(project="mbpo_online")
    wandb.config.update(OmegaConf.to_container(cfg=cfg))

    env = gym.make(cfg.env_name)
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=1)

    eval_env = gym.make(cfg.env_name)

    kwargs = OmegaConf.to_container(cfg.algo_kwargs)
    agent = SACLearner(cfg.seed, env.observation_space, env.action_space, **kwargs)
    model = ModelLearner(cfg.seed, env.observation_space, env.action_space, **kwargs)

    replay_buffer = ReplayBuffer(env.observation_space, env.action_space, cfg.max_steps)
    replay_buffer.seed(cfg.seed)

    (observation, _), done = env.reset(), False
    for i in tqdm.tqdm(
        range(1, int(cfg.max_steps) + 1), smoothing=0.1, disable=not cfg.tqdm
    ):
        if i < cfg.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = done or terminated

        if not terminated:
            mask = 1.0
        else:
            mask = 0.0
        replay_buffer.insert(
            dict(
                observations=observation,
                actions=action,
                rewards=reward,
                masks=mask,
                dones=done,
                next_observations=next_observation,
            )
        )
        observation = next_observation

        if truncated or terminated:
            (observation, _), done = env.reset(), False
            for k, v in info["episode"].items():
                decode = {"r": "return", "l": "length", "t": "time"}
                wandb.log({f"training/{decode[k]}": v}, step=i)

        if i >= cfg.start_training:
            if (i - cfg.start_training) % cfg.model_update_interval == 0:
                info = model.update_model(replay_buffer, cfg.batch_size)
                for k, v in info.items():
                    wandb.log({f"training/model/{k}": v}, step=i)

            model_dataset = model.yield_data(
                replay_buffer,
                agent,
                cfg.batch_size,
                cfg.policy_steps,
                termination_fn=lookup_termination_fn(cfg.env_name),
                depth=compute_schedule(*cfg.algo_kwargs.depth_schedule, i//1000)
            )
            for batch in model_dataset:
                batch = replay_buffer.sample(cfg.batch_size)
                update_info = agent.update(batch)

            if i % cfg.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f"training/critic/{k}": v}, step=i)

        if i % cfg.eval_interval == 0:
            eval_info = evaluate(agent, eval_env, num_episodes=cfg.eval_episodes)
            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=i)


def compute_schedule(init_epoch, end_epoch, init_value, end_value, epoch):
    if epoch < init_epoch:
        return init_value
    if epoch > end_epoch:
        return end_value
    return int(init_value + (end_value - init_value) * (epoch - init_epoch) / (end_epoch - init_epoch))


if __name__ == "__main__":

    main()
