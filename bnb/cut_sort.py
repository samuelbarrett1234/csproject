"""
Implementation of "cutting sort", an alternative algorithm to the branch-and-bound
procedure.
"""


import numpy as np


def _batched_merge_sort(xs, batched_cmp, start=0, end=-1):
    end %= xs.shape[1]  # turn -1 into actual end

    if start == end:  # base case - nothing to do
        pass

    else:
        mid = (start + end) // 2  # middle
        assert(mid + 1 <= end)

        # sort halves
        _batched_merge_sort(xs, batched_cmp, start, mid)
        _batched_merge_sort(xs, batched_cmp, mid + 1, end)

        # now merge in parallel:

        idx1 = np.ones((xs.shape[0],), dtype=np.int32) * start
        idx2 = np.ones_like(idx1) * (mid + 1)
        output = np.zeros_like(xs[:, start:end + 1])
        i_out = 0

        while np.any(idx1 <= mid) or np.any(idx2 <= end):
            # check that every seq in the batch isn't finished yet:
            assert(np.all(np.logical_or(idx1 <= mid, idx2 <= end)))
            i_in = np.arange(0, idx1.shape[0], dtype=np.int32)

            b = batched_cmp(
                xs[i_in, idx1],
                xs[i_in, idx2 % xs.shape[1]]  # safety
            )
            # pick idx1 if idx2 is finished or b >= 0, and idx1 is not finished
            # (when b == 0 we can put it in either, but put it in 1 for stability)
            update = np.logical_and(np.logical_or(b >= 0, idx2 > end), idx1 <= mid)
            output[:, i_out] = np.where(update, xs[i_in, idx1], xs[i_in, idx2 % (end + 1)])
            # advance the relevant indices
            idx1[np.argwhere(update)] += 1
            idx2[np.argwhere(np.logical_not(update))] += 1
            # output next value
            i_out += 1

        assert(i_out == output.shape[1])

        # store output back into original array
        xs[:, start:end + 1] = output


def cut_sort(model, seqs, mask_value):
    def H(idxs1, idxs2):
        xs = np.copy(seqs)
        rng = np.arange(0, idxs1.shape[0], dtype=np.int32)
        xs[:, idxs1] = mask_value
        xs[:, idxs2] = mask_value
        hs = model(xs)
        hs = np.sum(-hs * np.log(hs), axis=-1)
        hs = np.sign(hs[rng, idxs1] - hs[rng, idxs2])
        assert(hs.shape == idxs1.shape)
        return hs

    idxs = np.repeat(np.arange(0, seqs.shape[1])[np.newaxis, :], seqs.shape[0], axis=0)
    _batched_merge_sort(idxs, H)
    return idxs
