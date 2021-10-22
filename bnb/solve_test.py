import numpy as np
from bnb import padded_batch, solve, solve_mask


def _model(seqs):
    batch_size = 32
    assert(len(seqs) == batch_size)
    ps = np.ones((batch_size, seqs.shape[1], 16))
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


def _complex_model(seqs):
    assert(seqs.shape[1] == 5)
    result = 0.5 * np.ones((seqs.shape[0], seqs.shape[1], 2))
    know1 = np.any(seqs[:, :3] != 3, axis=-1)
    know2 = np.any(seqs[:, 3:] != 3, axis=-1)
    result[know1, :3, 0] = 1.0
    result[know1, :3, 1] = 0.0
    result[know2, 3:, 1] = 1.0
    result[know2, 3:, 0] = 0.0
    return result


def test_solve_complex():
    seqs = np.array([0, 0, 0, 1, 1], dtype=np.int32)[np.newaxis, :]
    result = solve_mask(_complex_model, np.log(2), seqs, np.ones_like(seqs), 3, 8)
    assert(np.sum(np.where(result == 3, 1, 0)) == 4)
