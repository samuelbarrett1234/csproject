import numpy as np
from bnb import padded_batch, solve_mask


def _model(seqs):
    batch_size = 32
    assert(len(seqs) == batch_size)
    ps = np.random.random((batch_size, seqs.shape[1], 256))
    return ps / np.sum(ps, axis=-1, keepdims=True)


def test_padded_batch():
    seqs = [np.array([0, 1, 2], dtype=np.int32),
            np.array([3, 4], dtype=np.int32)]
    seqs, init_keep = padded_batch(seqs, 5)
    assert(np.all(seqs == np.array(
        [[0, 1, 2], [3, 4, 5]],
        dtype=np.int32
    )))
    assert(np.all(init_keep == np.array(
        [[1, 1, 1], [1, 1, 0]],
        dtype=np.int32
    )))


def test_solve_mask():
    seqs = np.random.randint(0, 7, size=(4, 48))
    solve_mask(_model, 14.3, seqs, np.ones_like(seqs), 8, 32)


def test_solve_mask_early_stopping():
    seqs = np.random.randint(0, 7, size=(4, 48))
    solve_mask(_model, 14.3, seqs, np.ones_like(seqs), 8, 32,
               early_stopping_cond=lambda p, d: (d - p <= 3))
