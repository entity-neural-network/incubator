import numpy as np
import torch
from ragged_buffer import RaggedBufferI64

from entity_gym.env import VecCategoricalActionMask
from rogue_net.categorical_action_head import CategoricalActionHead
from rogue_net.ragged_tensor import RaggedTensor


def test_empty_actors() -> None:
    head = CategoricalActionHead(d_model=4, n_choice=2)
    x = RaggedTensor(
        data=torch.zeros(12, 4),
        batch_index=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3]),
        lengths=torch.tensor([4, 4, 2, 2]),
    )

    action, lengths, logprob, entropy, logits = head.forward(
        x,
        index_offsets=RaggedBufferI64.from_array(
            np.array([[[0]], [[4]], [[6]], [[8]]])
        ),
        mask=VecCategoricalActionMask(
            actors=RaggedBufferI64.from_flattened(
                np.zeros((0, 1), dtype=np.int64),
                lengths=np.array([0, 0, 0, 0], dtype=np.int64),
            ),
            mask=None,
        ),
        prev_actions=None,
    )
    assert action.shape == (0,)
    assert np.array_equal(lengths, np.array([0, 0, 0, 0]))
    assert logprob.shape == (0,)
    assert entropy.shape == (0,)
    assert logits.shape == (0, 2)
