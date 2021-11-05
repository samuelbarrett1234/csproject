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

    # compute model output entropies
    hs = np.zeros(original_shape, dtype=np.float32)
    for i in range(seqs.shape[1]):
        ps_i = model(seqs[:, i, :])[:, i, :]
        hs[:, i] = np.sum((-ps_i * np.ma.log(ps_i)).filled(0.0), axis=-1)

    return np.argsort(hs, axis=-1)
