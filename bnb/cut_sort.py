"""
Implementation of "cutting sort", an alternative algorithm to the branch-and-bound
procedure.
"""


import numpy as np


def cut_sort(model, seqs, mask_value, blocking=None):
    if blocking is None:
        blocking = 1

    n_steps = seqs.shape[1] // blocking
    if n_steps % blocking > 0:
        n_steps += 1

    # compute model output entropies
    hs = np.zeros(seqs.shape, dtype=np.float32)
    for i in range(n_steps):
        # mask out the given block [i * blocking, i * (blocking + 1)),
        # taking into account that the sequence length may not be divisible
        # by the block size
        block_idxs = np.arange(
            i * blocking,
            min(i * (blocking + 1), seqs.shape[1]),
            dtype=np.int32)

        block_masked_seqs = np.copy(seqs)
        block_masked_seqs[:, block_idxs] = mask_value
        ps_i = model(block_masked_seqs)
        hs[:, block_idxs] = np.sum((-ps_i * np.ma.log(ps_i)).filled(0.0), axis=-1)

    return np.argsort(hs, axis=-1)
