# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math

import jax
import jax.numpy as jnp


def get_hopper(env):
    z_range = env.unwrapped._healthy_z_range
    angle_range = env.unwrapped._healthy_angle_range
    state_range = env.unwrapped._healthy_state_range

    @jax.jit
    def term_fn(next_obs: jax.Array) -> jax.Array:
        z = next_obs[:, 0]
        angle = next_obs[:, 1]
        state = next_obs[:, 2:]

        healthy_z = (z > z_range[0]) * (z < z_range[1])
        healthy_angle = (angle > angle_range[0]) * (angle < angle_range[1])
        healthy_state = jax.lax.reduce_and(
            (state > state_range[0]) * (state < state_range[1]), axes=[1]
        )

        is_healthy = jnp.logical_and(
            jnp.logical_and(healthy_z, healthy_angle), healthy_state
        )

        return jnp.logical_not(is_healthy)

    return term_fn


def no_termination(env):
    def term_fn(next_obs: jax.Array):
        return jnp.zeros_like(next_obs[:, 0])

    return term_fn


def get_walker2d(env):
    z_range = env.unwrapped._healthy_z_range
    angle_range = env.unwrapped._healthy_angle_range

    @jax.jit
    def term_fn(next_obs: jax.Array) -> jax.Array:
        z = next_obs[:, 0]
        angle = next_obs[:, 1]

        healthy_z = (z > z_range[0]) * (z < z_range[1])
        healthy_angle = (angle > angle_range[0]) * (angle < angle_range[1])
        is_healthy = jnp.logical_and(healthy_z, healthy_angle)

        return jnp.logical_not(is_healthy)

    return term_fn


def get_ant(env):
    z_range = env.unwrapped._healthy_z_range

    @jax.jit
    def term_fn(next_obs: jax.Array) -> jax.Array:
        z = next_obs[:, 0]

        healthy_z = (z > z_range[0]) * (z < z_range[1])

        return jnp.logical_not(healthy_z)


def get_humanoid(env):
    z_range = env.unwrapped._healthy_z_range

    @jax.jit
    def term_fn(next_obs: jax.Array) -> jax.Array:
        z = next_obs[:, 0]

        healthy_z = (z > z_range[0]) * (z < z_range[1])

        return jnp.logical_not(healthy_z)


def lookup_termination_fn(env_name, env):
    return {
        "Hopper-v5": get_hopper,
        "Walker2d-v5": get_walker2d,
        "Ant-v5": get_ant,
        "Humanoid-v5": get_humanoid,
    }.get(env_name, no_termination)(env)
