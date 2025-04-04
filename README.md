# Jax - Model-based Policy Optimization

This repository implements the papers [When to Trust Your Model: Model-Based Policy Optimization](https://github.com/jannerm/mbpo?tab=readme-ov-file) and [Value Gradient weighted Model-Based Reinforcement Learning](https://github.com/pairlab/vagram/tree/main) in jax.

The underlying SAC code is built on [jaxrl2](https://github.com/ikostrikov/jaxrl2).

This is a work in progress, and it builds on newer environment version. Therefore I cannot guarantee exactly equivalent results to the original papers.

## Installation

The repository uses [`uv`](https://github.com/astral-sh/uv) to make running simple. Make sure you have a GPU with CUDA 12 installed.
If you want to run this on CPU, TPU, or an ARM machine, you will have to change the relevant jax package in the `pyrpoject.toml` file.
Necessary python packages will be installed automatically if you execute the run script.
In case you need a separate installation you can also create a virtualenv and run `pip install -e .`.
Necessary dependencies will be installed.

Logging is handled via weights and biases.
Please make sure you have a wandb account.
In case your system is not set up for wandb but you have an account, you will be prompted to generate an API key automatically.
Simply follow the instructions.

## Running

To run the experiment, you can simply execute `uv run mbpo/runner/train_online.py`.
The config is handled via hydra.
The default config can be found in `config/main.yaml`.

## Roadmap

- [x] hydra submitit integration
- [x] set default configs to paper values for each env
- [x] saving and loading of models and interrupt training
- [x] random distractions from paper
- [ ] run script for paper experiments
- [ ] modern SAC architectures
- [ ] parallel multi-seed training
  - difficult due to variable length training times in MBPO
