"""
Implements masking procedures based on the original
BERT paper, SpanBERT, and my branch and bound
procedure.
"""


import numpy as np
from bnb import padded_batch, solve_mask


def _apply_mask(seqs, mask, alphabet_size, mask_value, pad_value):
    """Once we know *where* the masks will be applied, this
    function does so, in a way consistent with the BERT paper.
    """
    assert(mask.dtype == np.bool)
    assert(np.all(seqs != mask_value))
    # uniformly random tokens (but NOT masking or padding tokens!)
    toks = [i for i in range(alphabet_size) if i not in (mask_value, pad_value)]
    rand_tokens = np.random.choice(toks, size=seqs.shape)

    # constant matrix containing the `mask_value` everywhere:
    mask_tokens = mask_value * np.ones_like(seqs, dtype=np.int32)

    # choosing which slice to select from:
    mask_result = np.random.choice(a=[0, 1, 2], size=seqs.shape,
                                   p=[0.8, 0.1, 0.1])

    # obviously, any position which is not masked must be
    # kept the same:
    mask_result = np.where(mask, mask_result, 1)

    # now look it up in the following array:
    results = np.stack((mask_tokens, seqs, rand_tokens), axis=0)
    return np.choose(mask_result, results)


def bert_masking(seqs, pad_value, mask_value, alphabet_size):
    """Mask the given list of sequences according to
    the rules given in the BERT paper. Returns the sequences
    padded-batched into a matrix, with masking tokens inserted.
    """
    seqs, _ = padded_batch(seqs, pad_value)
    mask = np.random.choice(a=[True, False], size=seqs.shape,
                            p=[0.15, 0.85])

    # cannot mask padded positions
    mask = np.where(seqs == pad_value, False, mask)

    return _apply_mask(seqs, mask, alphabet_size, mask_value, pad_value)


def span_bert_masking(seqs, pad_value, mask_value, alphabet_size):
    """Mask the given list of sequences according to
    the rules given in the SpanBERT paper. Returns the sequences
    padded-batched into a matrix, with masking tokens inserted.
    """
    seqs, init_keep = padded_batch(seqs, pad_value)
    seq_lens = np.sum(init_keep, axis=-1)

    # generate random, geometric, masking lengths, which are
    # clipped at 10 and also no more than the sequence length
    L = np.random.geometric(p=0.2, size=(seqs.shape[0],))
    L = np.where(L > 10, 10, L)
    L = np.where(L > seq_lens, seq_lens, L)
    assert(np.all(L > 0))

    # generate start points, such that extending by L does not
    # go into the padding tokens at the end
    starts = np.random.random(size=(seqs.shape[0],))
    starts = np.floor(starts * (seq_lens - L)).astype(np.int32)
    assert(np.all(starts >= 0))
    assert(np.all(starts <= seq_lens - L))

    # now generate the masking matrix
    A = np.arange(0, seqs.shape[1], dtype=np.int32)
    mask = np.logical_and(
        A[np.newaxis, :] >= starts[:, np.newaxis],
        A[np.newaxis, :] < (starts + L)[:, np.newaxis]
    )
    return _apply_mask(seqs, mask, alphabet_size, mask_value, pad_value)


def bnb_masking(model, seqs, pad_value, mask_value, alphabet_size):
    """Mask the given list of sequences according to
    the branch-and-bound algorithm. Returns the sequences
    padded-batched into a matrix, with masking tokens inserted.
    """    
    seqs, init_keep = padded_batch(seqs, pad_value)

    # heuristically, mask at most 15% of the tokens
    # (of course, by independence, we may end up masking
    # more.)
    n_mask_aim = max(int(0.15 * seqs.shape[1]), 1)
    ent_bud = np.log(alphabet_size) * n_mask_aim
    assert(ent_bud > 0.0)
    stop_cond = lambda p, d: (p >= n_mask_aim or p == d)
    # warning: I'm using an entropy budget which is
    # determined using the length of the longest sequence.
    # this is probably not a problem, but if some sequences
    # are much shorter than others, they might end up getting
    # completely masked.

    new_seqs = solve_mask(model, ent_bud, seqs, init_keep,
                          mask_value, seqs.shape[0],
                          early_stopping_cond=stop_cond)
    mask = (new_seqs == mask_value)

    return _apply_mask(seqs, mask, alphabet_size, mask_value, pad_value)
