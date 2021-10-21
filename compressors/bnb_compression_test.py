import numpy as np
from compressors.bnb_compression import (
    serialise_l2r, serialise_bnb, compress_serialisation
)


def _model(seqs):
    return np.ones((seqs.shape[0], seqs.shape[1], 256)) / 256.0


def test_l2r():
    seqs = [
        np.array([1, 2, 3], dtype=np.int32),
        np.array([4, 5, 6, 7], dtype=np.int32)
    ]
    seqs, mask_arrays = serialise_l2r(seqs, 0)
    codes = compress_serialisation(_model, seqs, mask_arrays, 256, 2)
    assert(len(codes) == len(seqs))


def test_bnb():
    seqs = [
        np.array([1, 2, 3], dtype=np.int32),
        np.array([4, 5, 6, 7], dtype=np.int32)
    ]
    seqs, mask_arrays = serialise_bnb(_model, seqs, 256, 0, 2.0)
    codes = compress_serialisation(_model, seqs, mask_arrays, 256, 2)
    assert(len(codes) == len(seqs))
