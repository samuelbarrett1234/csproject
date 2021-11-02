"""
Implements the greedy ordering: at each timestep, transmit the
token with the highest conditional entropy.
"""


import numpy as np


def greedy_order(model, seqs, mask_value):
    idxs = np.zeros_like(seqs)
    mask_array = np.ones_like(seqs)
    for i in range(0, seqs.shape[1]):
        # run the model, then reveal the token with highest entropy
        ps = model(np.where(mask_array == 0, seqs, mask_value))
        Hs = np.sum((-ps * np.ma.log(ps)).filled(0.0), axis=-1)
        Hs = np.where(mask_array == 0, 0.0, Hs)
        maxs = np.argmax(Hs, axis=-1)
        # reveal the new elements
        idxs[:, i] = maxs
        mask_array[np.arange(0, seqs.shape[0], dtype=np.int32), maxs] = 0
    return idxs
