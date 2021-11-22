"""
Implements the greedy ordering: at each timestep, transmit the
token with the highest conditional entropy.
"""


import numpy as np


def greedy_order(model, seqs, mask_value, blocking=None):
    if blocking is None:
        blocking = 1

    idxs = np.zeros_like(seqs)
    mask_array = np.ones_like(seqs)

    n_steps = seqs.shape[1] // blocking
    if n_steps % blocking > 0:
        n_steps += 1

    for i in range(0, n_steps):
        # run the model, then reveal the token with highest entropy
        ps = model(np.where(mask_array == 0, seqs, mask_value))
        Hs = np.sum((-ps * np.ma.log(ps)).filled(0.0), axis=-1)
        Hs = np.where(mask_array == 0, 0.0, Hs)

        for k in range(min(blocking, idxs.shape[1] - i * blocking)):
            maxs = np.argmax(Hs, axis=-1)
            Hs[np.arange(Hs.shape[0]), maxs] = 0.0
            # reveal the new elements
            idxs[:, i * blocking + k] = maxs
            mask_array[np.arange(0, seqs.shape[0], dtype=np.int32), maxs] = 0

    return idxs
