import jax

from mbpo.types import Params


def soft_target_update(
    critic_params: Params, critic_target_params: Params, tau: float
) -> Params:
    new_target_params = jax.tree_util.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic_params, critic_target_params
    )

    return new_target_params
