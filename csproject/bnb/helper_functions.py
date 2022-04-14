import numpy as np


def padded_batch(seqs, pad_token, min_length=None):
    """Take a collection of sequences of possibly different lengths, and
    put them together in a matrix, with padding at the ends. In the
    meantime, also produce an initial `Keep` state, to pass to `solve_mask`,
    because we do not want to mask the padding tokens!

    Args:
        seqs (List of np.ndarray): Collection of integral sequences. Must not
                                   contain `pad_token`.
        pad_token (int): The padding token. (Important note: the model should
                         ideally not be able to see these. We do not want
                         it to think it can use these as information about
                         the sequence itself!)
        min_length (int, optional): If not None, force the output sequences to
                                    be at least this long, by adding padding.
                                    If provided, cannot be shorter than any
                                    sequence.
    """
    assert(not np.any(seqs == pad_token))
    N = max(map(len, seqs))
    seqs = np.stack(
        [np.concatenate((
            s,
            pad_token * np.ones((N - len(s)), dtype=s.dtype)
            )) for s in seqs]
    )
    if min_length is not None:
        assert(min_length >= seqs.shape[1])
        seqs = np.concatenate(
            (seqs,
            pad_token *
            np.ones((seqs.shape[0],
                    min_length - seqs.shape[1]),
                    dtype=np.int32)), axis=1)
    init_keep = np.where(
        seqs == pad_token,
        0, 1
    )
    return seqs, init_keep
