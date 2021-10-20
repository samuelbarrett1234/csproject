import numpy as np
from bnb import padded_batch, solve_mask


def _model(seqs):
    batch_size = 32
    assert(len(seqs) == batch_size)
    ps = np.random.random((batch_size, seqs.shape[1], 256))
    return ps / np.sum(ps, axis=-1, keepdims=True)


def test_solve_mask():
    seqs = np.random.randint(0, 7, size=(4, 48))
    solve_mask(_model, 14.3, seqs, np.ones_like(seqs), 8, 32)
