# Jax - Model-based Policy optimization

This repository implements the papers [When to Trust Your Model: Model-Based Policy Optimization](https://github.com/jannerm/mbpo?tab=readme-ov-file) and [Value Gradient weighted Model-Based Reinforcement Learning](https://github.com/pairlab/vagram/tree/main) in jax.

The underlying SAC code is built on [jaxrl2](https://github.com/ikostrikov/jaxrl2).

This is a work in progress, and it builds on newer environment version. Therefore I cannot guarantee exactly equivalent results to the original papers.

## Installation

The repository uses `uv` to make running simple. Make sure you have a GPU with CUDA 12 installed.
Necessary python packages will be installed automatically if you execute the run script.
In case you need a separate installation you can also create a virtualenv and run `pip install -e .`.
Necessary dependencies will be installed.

Logging is handled via weights and biases.
Please make sure you have a wandb account.
In case your system is not set up for wandb but you have an account, you will be prompted to generate an API key automatically.
Simply follow the instructions.

## Running

To run the experiment, you can simply execute `uv run mbp/runner/train_online.py`.

## Roadmap

- [ ] hydra submitit integration
- [ ] saving and loading of models and interrupt training
- [ ] modern SAC architectures
- [ ] parallel multi-seed training