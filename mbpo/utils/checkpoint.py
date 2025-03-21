import flax


@flax.struct.dataclass
class CheckpointGroup:
    agent: flax.core.frozen_dict.FrozenDict
    model: flax.core.frozen_dict.FrozenDict
    buffer: flax.core.frozen_dict.FrozenDict
