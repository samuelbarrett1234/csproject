"""
Implementation of "cutting sort", an alternative algorithm to the branch-and-bound
procedure.
"""


import numpy as np


def cut_sort(model, seqs, mask_value):
    original_shape = seqs.shape
    # repeat each sequence several times, keeping exactly one
    # token masked in turn
    I = np.eye(seqs.shape[1], dtype=np.int32)
    I = np.repeat(I[np.newaxis, :, :], seqs.shape[0], axis=0)
    seqs = np.repeat(seqs[:, np.newaxis, :], seqs.shape[1], axis=1)
    seqs = np.where(I == 1, mask_value, seqs)
    seqs = np.reshape(seqs, (seqs.shape[0] * seqs.shape[1], seqs.shape[2]))

    assert(seqs.shape == (original_shape[0] * original_shape[1], original_shape[1]))

    # compute model output entropies
    ps = np.reshape(
        model(seqs),
        (original_shape[0], original_shape[1], original_shape[1]) + (-1,))
    hs = np.sum((-ps * np.ma.log(ps)).filled(0.0), axis=-1)

    # pick out the entropy of the position we're interested in
    # (by producing an array whose i,j th element is the i,j,j th
    # element of `hs` above)
    r1 = np.arange(0, original_shape[0], dtype=np.int32)
    r2 = np.arange(0, original_shape[1], dtype=np.int32)
    r1 = np.repeat(r1[:, np.newaxis], original_shape[1], axis=1)
    r2 = np.repeat(r2[np.newaxis, :], original_shape[0], axis=0)
    hs = hs[r1, r2, r2]

    return np.argsort(hs, axis=-1)
