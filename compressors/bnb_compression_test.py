import numpy as np
from compressors.bnb_compression import (
    serialise_l2r, serialise_bnb, compress_serialisation
)


def _model(seqs):
    assert(seqs.shape == (2, 4))
    p_pad = np.array([1.0 if i == 0 else 0.0 for i in range(257)])
    p_other = (1.0 - p_pad) / 256.0
    ps = np.where((seqs == 0)[..., np.newaxis], p_pad, p_other)
    return ps


def test_l2r():
    seqs = [
        np.array([1, 2, 3], dtype=np.int32),
        np.array([4, 5, 6, 7], dtype=np.int32)
    ]
    seqs, mask_arrays = serialise_l2r(seqs, 0)
    assert(np.all(mask_arrays == np.array([
        [[1, 1, 1, 0],
        [1, 1, 1, 1]],

       [[0, 1, 1, 0],
        [0, 1, 1, 1]],

       [[0, 0, 1, 0],
        [0, 0, 1, 1]],

       [[0, 0, 0, 0],
        [0, 0, 0, 1]],

       [[0, 0, 0, 0],
        [0, 0, 0, 0]]
        ], dtype=np.int32)))
    codes = compress_serialisation(_model, seqs, mask_arrays, 257, 2)
    assert(len(codes) == len(seqs))
    assert(len(codes[0]) == 3 * 8)
    assert(len(codes[1]) == 4 * 8)


def test_bnb():
    seqs = [
        np.array([1, 2, 3], dtype=np.int32),
        np.array([4, 5, 6, 7], dtype=np.int32)
    ]
    seqs, mask_arrays = serialise_bnb(_model, seqs, 257, 0, 10.0)
    codes = compress_serialisation(_model, seqs, mask_arrays, 257, 2)
    assert(len(codes) == len(seqs))
    assert(len(codes[0]) == 3 * 8)
    assert(len(codes[1]) == 4 * 8)
