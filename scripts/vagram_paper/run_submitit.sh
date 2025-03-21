# baseline

uv run mbpo/runner/train_online.py \
    --config-name=main_submitit \
    --multirun \
    hydra/launcher=submitit_slurm \
    algo=ant,humanoid,cartpole,cheetah,hopper,walker \
    algo.model_kwargs.loss_mode=vagram \
    algo.seed=$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM \
    algo.model_kwargs.loss_mode=nll \
    algo.model_kwargs.deterministic=True \
    algo.wandb_kwargs.entity=viper_svg \
    algo.wandb_kwargs.project=mbpo_online &
uv run mbpo/runner/train_online.py \
    --config-name=main_submitit \
    --multirun \
    hydra/launcher=submitit_slurm \
    algo=ant,humanoid,cartpole,cheetah,hopper,walker \
    algo.model_kwargs.loss_mode=vagram \
    algo.seed=$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM \
    algo.model_kwargs.loss_mode=nll \
    algo.model_kwargs.deterministic=False \
    algo.wandb_kwargs.entity=viper_svg \
    algo.wandb_kwargs.project=mbpo_online &
uv run mbpo/runner/train_online.py \
    --config-name=main_submitit \
    --multirun \
    hydra/launcher=submitit_slurm \
    algo=hopper \
    algo.model_kwargs.loss_mode=vagram \
    algo.seed=$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM \
    algo.model_kwargs.loss_mode=vagram \
    algo.model_kwargs.deterministic=True \
    algo.wandb_kwargs.entity=viper_svg \
    algo.wandb_kwargs.project=mbpo_online &

# small model

uv run mbpo/runner/train_online.py \
    --config-name=main_submitit \
    --multirun \
    hydra/launcher=submitit_slurm \
    algo=hopper \
    algo.model_kwargs.loss_mode=vagram \
    algo.seed=$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM \
    algo.model_kwargs.model_hidden_dims=[200,200],[200,200,200] \
    algo.model_kwargs.loss_mode=vagram \
    algo.model_kwargs.deterministic=True \
    algo.wandb_kwargs.entity=viper_svg \
    algo.wandb_kwargs.project=mbpo_online &
uv run mbpo/runner/train_online.py \
    --config-name=main_submitit \
    --multirun \
    hydra/launcher=submitit_slurm \
    algo=hopper \
    algo.model_kwargs.loss_mode=vagram \
    algo.seed=$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM \
    algo.model_kwargs.model_hidden_dims=[200,200],[200,200,200] \
    algo.model_kwargs.loss_mode=nll \
    algo.model_kwargs.deterministic=False \
    algo.wandb_kwargs.entity=viper_svg \
    algo.wandb_kwargs.project=mbpo_online &

# distracting dimensions

uv run mbpo/runner/train_online.py \
    --config-name=main_submitit \
    --multirun \
    hydra/launcher=submitit_slurm \
    algo=hopper \
    algo.model_kwargs.loss_mode=vagram \
    algo.seed=$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM \
    algo.env_distractions=True \
    algo.distraction_kwargs.num_distractions=5,10,15 \
    algo.model_kwargs.loss_mode=vagram \
    algo.model_kwargs.deterministic=True \
    algo.wandb_kwargs.entity=viper_svg \
    algo.wandb_kwargs.project=mbpo_online &
uv run mbpo/runner/train_online.py \
    --config-name=main_submitit \
    --multirun \
    hydra/launcher=submitit_slurm \
    algo=hopper \
    algo.model_kwargs.loss_mode=vagram \
    algo.seed=$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM \
    algo.env_distractions=True \
    algo.distraction_kwargs.num_distractions=5,10,15 \
    algo.model_kwargs.loss_mode=nll \
    algo.model_kwargs.deterministic=False \
    algo.wandb_kwargs.entity=viper_svg \
    algo.wandb_kwargs.project=mbpo_online &