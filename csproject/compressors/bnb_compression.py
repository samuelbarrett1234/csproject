"""
This script provides helper functions to implement
compression based on probabilistic models, possibly
using the B&B techniques.

The methods starting with `serialise_` will return
a `mask_arrays` numpy array, which can then be passed
directly to `compress_serialisation` which turns it
into a sequence of codes.
"""


import numpy as np
from bnb import padded_batch, cut_sort, greedy_order
from compressors.coding import huffman_codebook


def _block_mask_arrays(mask_arrays, blocking):
    if blocking == 1 or blocking is None:
        return mask_arrays

    # select masking arrays in steps of size `blocking`
    rng = list(range(0, mask_arrays.shape[0], blocking))
    # but it is crucial that we sample the first and the last:
    if rng[0] != 0:
        rng = [0] + rng
    if rng[-1] != mask_arrays.shape[0] - 1:
        rng = rng + [mask_arrays.shape[0] - 1]
    return mask_arrays[rng]


def _keep_start_end(seqs, init_keep):
    """Add the start/end/separation symbols to the initial
    keep set. This is important, as models are allowed to
    condition on such information for free.
    """
    seq_lens = np.array(list(map(len, seqs)), dtype=np.int32)
    sep_symbol = seqs[0, seq_lens[0] - 1]
    start_symbol = seqs[0, 0]
    # condition on ALL occurrences of the special symbols!
    # typically they will only appear at the start and end
    # of each sequence, but for paired sequences they also
    # appear in the middle
    init_keep = np.where(np.logical_or(seqs == start_symbol, seqs == sep_symbol), 0, init_keep)
    return init_keep


def serialise_l2r(seqs, pad_value,
                  keep_start_end=False,
                  min_length=None,
                  blocking=None):
    """Serialise a list of sequences according to their left-to-right
    ordering.

    Args:
        seqs (list of np.ndarray): A list of 1D integral vectors
        pad_value (int): The integer representing padding, so that the
                         sequences can be padded to the same length and
                         put into a batch.
        keep_start_end (boolean): If true, do not mask the first and last token
                                  of any sequence.
        min_length (int, optional): If not None, force the output sequences to
                                    be at least this long, by adding padding.
                                    If provided, cannot be shorter than any
                                    sequence.
        blocking (int, optional): If not None, mask tokens in groups of this
                                  size. This will make for a slightly worse
                                  compressor, but the speedup will be on the
                                  order of the blocking size. Passing None
                                  here is equivalent to passing 1.

    Returns:
        A 2-tuple containing the padded batch of sequences to run the
        model on, and the list of masking matrices. These parameters
        should be passed directly to `compress_serialisation`.
    """
    seqs, init_keep = padded_batch(seqs, pad_value, min_length=min_length)
    if keep_start_end:
        init_keep = _keep_start_end(seqs, init_keep)

    mask_arrays = np.zeros((seqs.shape[1] + 1, seqs.shape[0], seqs.shape[1]), dtype=np.int32)
    mask_arrays[0] = init_keep
    for i in range(1, mask_arrays.shape[0]):
        # at timestep i, copy the state from the last timestep
        # but reveal token i-1
        mask_arrays[i] = mask_arrays[i - 1]
        mask_arrays[i, :, i - 1] = 0
    return seqs, _block_mask_arrays(mask_arrays, blocking)


def serialise_cutting_sort(model, seqs, mask_value, pad_value,
                           keep_start_end=False, min_length=None,
                           blocking=None):
    """Serialise a list of sequences according to the 'cutting sort order'.

    Args:
        model (callable): The model function, which takes as input a
                          numpy array of shape (batch-size, seq-len)
                          and outputs a probability distribution at
                          each of those positions.
        seqs (list of np.ndarray): A list of 1D integral vectors of
                                   length equal to the batch size!!!
        mask_value (int): The integer representing a masked token.
        pad_value (int): The integer representing padding, so that the
                         sequences can be padded to the same length and
                         put into a batch.
        ent_bud (float): The entropy budget - aka the minimum amount
                         of entropy to be contained in each "new message".
                         The smaller this is, the more messages you will
                         send, and potentially the more compressive,
                         however it will be slower.
        keep_start_end (boolean): If true, do not mask the first and last token
                                  of any sequence.
        min_length (int, optional): If not None, force the output sequences to
                                    be at least this long, by adding padding.
                                    If provided, cannot be shorter than any
                                    sequence.
        blocking (int, optional): If not None, mask tokens in groups of this
                                  size. This will make for a slightly worse
                                  compressor, but the speedup will be on the
                                  order of the blocking size. Passing None
                                  here is equivalent to passing 1.

    Returns:
        A 2-tuple containing the padded batch of sequences to run the
        model on, and the list of masking matrices. These parameters
        should be passed directly to `compress_serialisation`.
    """
    seqs, init_keep = padded_batch(seqs, pad_value, min_length=min_length)
    if keep_start_end:
        init_keep = _keep_start_end(seqs, init_keep)

    n_mask_arrays = seqs.shape[1] + 1
    if keep_start_end:
        n_mask_arrays -= 2

    mask_arrays = np.repeat(init_keep[np.newaxis, :, :], n_mask_arrays, axis=0)
    mask_arrays[-1, :, :] = 0  # final array must be all-zeroes

    idxs = cut_sort(model, seqs, mask_value, blocking=blocking)

    for i in range(seqs.shape[0]):
        # remove start/end/padding
        # (need to do this separately for each element in the batch,
        # since they may be of different lengths)
        i_idxs = idxs[i, init_keep[i, idxs[i, :]] == 1]
        for j in range(i_idxs.shape[0]):
            mask_arrays[j, i, i_idxs[:j]] = 0
        # any leftover padding
        # (aka accounting for any discrepancy between `n_mask_arrays`
        # and len(i_idxs))
        mask_arrays[i_idxs.shape[0]:, i, :] = 0

    return seqs, _block_mask_arrays(mask_arrays, blocking)


def serialise_greedy(model, seqs, mask_value, pad_value,
                     keep_start_end=False, min_length=None,
                     blocking=None):
    """Serialise a list of sequences according to the 'greedy order'.

    Args:
        model (callable): The model function, which takes as input a
                          numpy array of shape (batch-size, seq-len)
                          and outputs a probability distribution at
                          each of those positions.
        seqs (list of np.ndarray): A list of 1D integral vectors of
                                   length equal to the batch size!!!
        mask_value (int): The integer representing a masked token.
        pad_value (int): The integer representing padding, so that the
                         sequences can be padded to the same length and
                         put into a batch.
        ent_bud (float): The entropy budget - aka the minimum amount
                         of entropy to be contained in each "new message".
                         The smaller this is, the more messages you will
                         send, and potentially the more compressive,
                         however it will be slower.
        keep_start_end (boolean): If true, do not mask the first and last token
                                  of any sequence.
        min_length (int, optional): If not None, force the output sequences to
                                    be at least this long, by adding padding.
                                    If provided, cannot be shorter than any
                                    sequence.
        blocking (int, optional): If not None, mask tokens in groups of this
                                  size. This will make for a slightly worse
                                  compressor, but the speedup will be on the
                                  order of the blocking size. Passing None
                                  here is equivalent to passing 1.

    Returns:
        A 2-tuple containing the padded batch of sequences to run the
        model on, and the list of masking matrices. These parameters
        should be passed directly to `compress_serialisation`.
    """
    seqs, init_keep = padded_batch(seqs, pad_value, min_length=min_length)
    if keep_start_end:
        init_keep = _keep_start_end(seqs, init_keep)

    n_mask_arrays = seqs.shape[1] + 1
    if keep_start_end:
        n_mask_arrays -= 2

    mask_arrays = np.repeat(init_keep[np.newaxis, :, :], n_mask_arrays, axis=0)
    mask_arrays[-1, :, :] = 0  # final array must be all-zeroes

    idxs = greedy_order(model, seqs, mask_value, blocking=blocking)

    for i in range(seqs.shape[0]):
        # remove start/end/padding
        # (need to do this separately for each element in the batch,
        # since they may be of different lengths)
        i_idxs = idxs[i, init_keep[i, idxs[i, :]] == 1]
        for j in range(i_idxs.shape[0]):
            mask_arrays[j, i, i_idxs[:j]] = 0
        # any leftover padding
        # (aka accounting for any discrepancy between `n_mask_arrays`
        # and len(i_idxs))
        mask_arrays[i_idxs.shape[0]:, i, :] = 0

    return seqs, _block_mask_arrays(mask_arrays, blocking)


def compress_serialisation(model, seqs, mask_arrays, mask_value, d,
                           chunking=1):
    """Given a batch of sequences, and an order in which to compress them,
    perform that compression.

    The key ingredient here is `mask_arrays`. The outer dimension corresponds
    to "time" (revealing new tokens at each timestep). The inner dimension
    corresponds to the batch and then the sequence length. The matrix is a
    binary matrix where 1 denotes masking, 0 denotes no masking. The sequence
    of matrices over the outer dimension must be well-ordered: once a position
    is unmasked it can never be masked again. The first mask matrix should be
    all ones, except possibly if there are any padding characters, which must
    be unmasked from the beginning. The last mask matrix must be all-zero.

    Args:
        model (callable): A function which is run on integer matrices of shape
                          (batch-size, seq-len), and outputs the probability
                          distribution at each cell.
        seqs (np.ndarray): An integral numpy array of shape (batch-size, seq-len)
                           which does not contain the value `mask_value`.
        mask_arrays (np.ndarray): A binary numpy array of shape
                                  (num-timesteps, batch-size, seq-len), which
                                  satisfies rules as explained above.
        mask_value (int): The index corresponding to the masking value.
        d (int): The arity of the output.
        chunking (int): Call the model once every `chunking`th token. Higher
                        values will result in less efficient codes, but the
                        speedup is of the order of the chunking value.
                        Warning: this operates ON TOP OF any chunking already
                        done in the masks. In other words, it chunks the
                        masks, rather than the tokens. Ignore this warning if
                        your masks are unmodified return values from
                        any serialise method that returns an actual ordering.

    Returns:
        List of Lists of Ints: Returns the d-ary Huffman codes for each sequence
                               in the batch.
    """
    assert(np.all(mask_arrays[-1] == 0))
    assert(not np.all(mask_arrays[0] == 0))  # SOME may be 0, for padding/start/end tokens
    codes = [[] for i in range(seqs.shape[0])]
    last_seqs = None
    last_masks = None
    chunk = 0
    for masks in mask_arrays:
        if last_seqs is not None:
            # compute which new tokens are being revealed at this
            # timestep (possibly none):
            new = last_masks - masks
            assert(np.all(new >= 0))
            # (note that the number of revealed tokens may be different
            # for different sequences)

            # only update the model every `chunking`th
            # token (but obviously it is crucial to call
            # this the first time `_compute_joint_code`
            # is called, because this sets up `ps`.)
            if chunk % chunking == 0:
                # run the model to get its current belief over masked
                # sequences:
                ps = model(last_seqs)
            chunk += 1

            # now, for each sequence, compute the Huffman code of the
            # joint distribution of the revealed tokens, and add it
            # to the code for sequence `i`.
            # (we do not need a comma character because Huffman codes
            # form prefix codes.)
            for i in range(len(codes)):
                codes[i] += _compute_joint_code(
                    ps[i, new[i] == 1, :],
                    seqs[i][new[i] == 1],
                    d
                )
        last_seqs = np.where(masks == 1, mask_value, seqs)
        last_masks = masks

    return codes


def _compute_joint_code(indep_dists, seq, d):
    """Given a collection of integral variables which
    are independent and drawn from given distributions,
    what d-ary Huffman code would/could they produce
    when encoded?

    This function is useful because we do not want to
    construct the entire Huffman codebook to just
    encode one sequence!

    Postcondition: returns empty list when given empty
    input.

    Args:
        indep_dists (np.ndarray): Array of shape (N, M)
                                  where the last axis is
                                  a probability distribution
                                  for each N.
        seq (np.ndarray): Array of shape (N,) with values in
                          the range 0..M-1.
        d (int): The arity of the Huffman code.

    Returns:
        list: List of length K with int values in the
              range 0..d-1.
    """
    # we know the *length* of the Huffman code, K, is given
    # by `ceil(-log_d(probability))`.
    # by swapping siblings around in the Huffman tree, any
    # specific event can be represented by a code of arbitrary
    # 1s and 0s for this fixed length.
    # hence for the purposes of this function it suffices to
    # compute K and randomly generate the code!
    # (we can only make this simplification because we don't
    # actually care about implementing a decoder, only the
    # fact that there exists a decoding algorithm.)
    idxs = np.arange(0, len(seq), dtype=np.int32)
    log_p = np.sum(np.log(indep_dists[idxs, seq]))
    K = np.ceil(-(log_p / np.log(d))).astype(np.int32)
    return list(np.random.randint(0, d, size=(K,)))
