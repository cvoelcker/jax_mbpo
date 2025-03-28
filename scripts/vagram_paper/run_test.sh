uv run mbpo/runner/train_online.py \
    --multirun \
    algo=hopper \
    algo.model_kwargs.loss_mode=vagram \
    algo.seed=$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM \
    algo.model_kwargs.model_hidden_dims=[200,200],[200,200,200] \
    algo.model_kwargs.loss_mode=nll \
    algo.model_kwargs.deterministic=false \
    algo.wandb_kwargs.entity=viper_svg \
    algo.wandb_kwargs.project=mbpo_online 
uv run mbpo/runner/train_online.py \
    --multirun \
    algo=hopper \
    algo.model_kwargs.loss_mode=vagram \
    algo.seed=$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM,$RANDOM \
    algo.env_distractions=True \
    algo.distraction_kwargs.num_distractions=5,10,15 \
    algo.model_kwargs.loss_mode=nll \
    algo.model_kwargs.deterministic=false \
    algo.wandb_kwargs.entity=viper_svg \
    algo.wandb_kwargs.project=mbpo_online &