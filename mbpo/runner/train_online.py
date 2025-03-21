#! /usr/bin/env pythonA
import copy
import os

import gymnasium as gym
import jax
import tqdm
import wandb
import hydra
from omegaconf import OmegaConf

import jax.numpy as jnp
from flax.training import orbax_utils
import orbax
import orbax.checkpoint as ocp

from mbpo.algos.model_learning.model_trainer import ModelTrainer
from mbpo.algos.sac.sac_trainer import SACTrainer
from mbpo.data import ReplayBuffer
from mbpo.evaluation import evaluate
from mbpo.env_utils.termination_fns import lookup_termination_fn
from mbpo.utils.checkpoint import CheckpointGroup


@hydra.main(config_path="../../config", config_name="main")
def main(cfg):
    if cfg.checkpoint_setup == "cluster":
	os.chdir(f"/cehckpoint/voelcker/{os.getenv(SLURM_JOB_ID)}")
        os.environ["TQDM_DISABLE"] = "True"

    cfg = cfg.algo
    if os.path.exists("wandb_id"):
        with open("wandb_id", "r") as f:
            run_id = f.read().strip()
        wandb.init(**cfg.wandb_kwargs, resume="allow", id=run_id)
    else:
        wandb.init(**cfg.wandb_kwargs, resume="allow")
        with open("wandb_id", "w") as f:
            f.write(wandb.run.id)

    wandb.config.update(OmegaConf.to_container(cfg=cfg))

    env = gym.make(cfg.env_name)
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=1)
    term_fn = lookup_termination_fn(cfg.env_name, env)

    eval_env = gym.make(cfg.env_name)

    sac_kwargs = OmegaConf.to_container(cfg.sac_kwargs)
    model_kwargs = OmegaConf.to_container(cfg.model_kwargs)
    agent = SACTrainer(cfg.seed, env.observation_space, env.action_space, **sac_kwargs)
    model = ModelTrainer(
        cfg.seed, env.observation_space, env.action_space, **model_kwargs
    )

    replay_buffer = ReplayBuffer(env.observation_space, env.action_space, cfg.max_steps)
    replay_buffer.seed(cfg.seed)

    # setup checkpoint handling
    options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
    checkpoint_manager = ocp.CheckpointManager(
        os.path.join(os.getcwd(), 'checkpoint'), options=options)
    
    step = checkpoint_manager.latest_step()

    if step is not None:
        print("Restoring from step", step)
        state = CheckpointGroup(
            agent=agent.get_checkpoint(),
            model= model.get_checkpoint(),
            buffer= replay_buffer.get_checkpoint(),
        )
        checkpoint = checkpoint_manager.restore(step, args=ocp.args.StandardRestore(state))
        agent.load_checkpoint(checkpoint.agent)
        model.load_checkpoint(checkpoint.model)
        replay_buffer.load_checkpoint(checkpoint.buffer)
    else:
        step = 0

    (observation, _), done = env.reset(), False
    for i in tqdm.tqdm(range(step, int(cfg.max_steps) + 1), disable=not cfg.tqdm):
        if (i % cfg.save_freq) == 0:
            state = CheckpointGroup(
                agent=agent.get_checkpoint(),
                model= model.get_checkpoint(),
                buffer= replay_buffer.get_checkpoint(),
            )
            checkpoint_manager.save(i, args=ocp.args.StandardSave(state))
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
            epoch = (i - cfg.start_training) // 1000
            if (i - cfg.start_training) % cfg.model_update_interval == 0:
                info = model.update_model(
                    replay_buffer, agent, cfg.batch_size
                )
                depth = compute_schedule(*cfg.model_kwargs.depth_schedule, epoch)
                prop_real = compute_schedule(
                    *cfg.model_kwargs.prop_real_schedule, epoch
                )
                model_dataset = model.yield_data(
                    copy.deepcopy(replay_buffer),
                    copy.deepcopy(agent),
                    cfg.batch_size,
                    cfg.policy_steps * cfg.model_update_interval,
                    termination_fn=term_fn,
                    depth=depth,
                    prop_real=prop_real
                )
                wandb.log({"training/model/depth": depth}, step=i)
                wandb.log({"training/model/prop_real": prop_real}, step=i)
                for k, v in info.items():
                    assert jnp.isfinite(
                        v
                    ).all(), f"{k} is not finite in iteration {i}, got {v}, with full dict {info}"
                    wandb.log({f"training/model/{k}": v}, step=i)

            for j in range(cfg.policy_steps):
                batch = next(model_dataset)
                if j == 0:
                    real_batch = replay_buffer.sample(cfg.batch_size)
                    # compute correctness of termination function
                    terminated_real = 1.0 - real_batch["masks"].squeeze()
                    terminated_pred = term_fn(real_batch["next_observations"]).squeeze()
                    correct = (terminated_real == terminated_pred).mean()
                    batch_info = {
                        "batch/rewards": batch["rewards"].mean(),
                        "batch/masks": batch["masks"].mean(),
                        "batch/real_rewards": real_batch["rewards"].mean(),
                        "batch/real_masks": real_batch["masks"].mean(),
                        "batch/actions": batch["actions"].mean(),
                        "batch/real_actions": real_batch["actions"].mean(),
                        "batch/actions_std": batch["actions"].std(),
                        "batch/real_actions_std": real_batch["actions"].std(),
                        "batch/termination_correct": correct,
                    }
                update_info = agent.update(batch, i * cfg.policy_steps + j)

            if i % cfg.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f"training/critic/{k}": v}, step=i)
                for k, v in batch_info.items():
                    wandb.log({f"training/batch/{k}": v}, step=i)

        if i % cfg.eval_interval == 0:
            eval_info = evaluate(agent, eval_env, num_episodes=cfg.eval_episodes)
            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=i)

            


def compute_schedule(init_epoch, end_epoch, init_value, end_value, increment, epoch):
    """
    Compute a schedule that linearly interpolates between init_value and end_value.
    The schedule is incremented discretely by increment to allow for integer values
    and to be used with jax.jit static argnames.

    Args:
        init_epoch (int): The epoch at which the schedule starts
        end_epoch (int): The epoch at which the schedule ends
        init_value (float): The value at init_epoch
        end_value (float): The value at end_epoch
        increment (int): The increment of the schedule
        epoch (int): The current epoch
    """
    dtype = jnp.array([increment]).dtype

    if epoch < init_epoch:
        schedule_value = init_value
    elif epoch > end_epoch:
        schedule_value = end_value
    else:
        schedule_value = (
            jnp.round(
                (end_value - init_value)
                / (end_epoch - init_epoch)
                * (epoch - init_epoch)
                / increment
            )
            * increment
            + init_value
        ).astype(dtype).item()
    return schedule_value


if __name__ == "__main__":

    main()
