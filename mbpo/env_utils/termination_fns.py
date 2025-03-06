# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math

import jax
import jax.numpy as jnp


def hopper(next_obs: jax.Array) -> jax.Array:
    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done = (
        jnp.isfinite(next_obs).all(-1)
        * (jnp.abs(next_obs[:, 1:]) < 100).all(-1)
        * (height > 0.7)
        * (jnp.abs(angle) < 0.2)
    )

    done = ~not_done
    done = done[:, None]
    return done


def cartpole(next_obs: jax.Array) -> jax.Array:
    x, theta = next_obs[:, 0], next_obs[:, 2]

    x_threshold = 2.4
    theta_threshold_radians = 12 * 2 * math.pi / 360
    not_done = (
        (x > -x_threshold)
        * (x < x_threshold)
        * (theta > -theta_threshold_radians)
        * (theta < theta_threshold_radians)
    )
    done = ~not_done
    done = done[:, None]
    return done


def inverted_pendulum(next_obs: jax.Array) -> jax.Array:
    not_done = jnp.isfinite(next_obs).all(-1) * (jnp.abs(next_obs[:, 1]) <= 0.2)
    done = ~not_done

    done = done[:, None]

    return done


def no_termination(next_obs: jax.Array) -> jax.Array:
    return jnp.zeros_like(next_obs[:, :1])


def walker2d(next_obs: jax.Array) -> jax.Array:
    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done = (height > 0.8) * (height < 2.0) * (angle > -1.0) * (angle < 1.0)
    done = ~not_done
    done = done[:, None]
    return done


def ant(next_obs: jax.Array):
    x = next_obs[:, 0]
    not_done = jnp.isfinite(next_obs).all(-1) * (x >= 0.2) * (x <= 1.0)

    done = ~not_done
    done = done[:, None]
    return done


def humanoid(next_obs: jax.Array):
    z = next_obs[:, 0]
    done = (z < 1.0) + (z > 2.0)

    done = done[:, None]
    return done


def lookup_termination_fn(env_name):
    return {
        "Hopper-v4": hopper,
        "CartPole-v4": cartpole,
        "InvertedPendulum-v4": inverted_pendulum,
        "Walker2d-v4": walker2d,
        "Ant-v4": ant,
        "Humanoid-v4": humanoid,
    }.get(env_name, no_termination)
