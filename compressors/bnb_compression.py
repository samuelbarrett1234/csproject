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
from bnb import padded_batch, solve_mask


def serialise_l2r(seqs, pad_value):
    """Serialise a list of sequences according to their left-to-right
    ordering.

    Args:
        seqs (list of np.ndarray): A list of 1D integral vectors
        pad_value (int): The integer representing padding, so that the
                         sequences can be padded to the same length and
                         put into a batch.

    Returns:
        A 2-tuple containing the padded batch of sequences to run the
        model on, and the list of masking matrices. These parameters
        should be passed directly to `compress_serialisation`.
    """
    seqs, init_keep = padded_batch(seqs, pad_value)
    mask_arrays = np.zeros((seqs.shape[1] + 1, seqs.shape[0], seqs.shape[1]), dtype=np.int32)
    mask_arrays[0] = init_keep
    for i in range(1, mask_arrays.shape[0]):
        # at timestep i, copy the state from the last timestep
        # but reveal token i-1
        mask_arrays[i] = mask_arrays[i - 1]
        mask_arrays[i, :, i - 1] = 0
    return seqs, mask_arrays


def serialise_bnb(model, seqs, mask_value, pad_value, ent_bud,
                  keep_start_end=False):
    """Serialise a list of sequences according to the output of solving
    the branch-and-bound procedure.

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

    Returns:
        A 2-tuple containing the padded batch of sequences to run the
        model on, and the list of masking matrices. These parameters
        should be passed directly to `compress_serialisation`.
    """
    lens = list(map(len, seqs))
    # pad the sequences, which simultaneously gives us
    # the initial `keep` state:
    seqs, keep = padded_batch(seqs, pad_value)

    # special start/end tokens must be kept if the caller indicates:
    if keep_start_end:
        keep[:, 0] = 0
        keep[[i for i in range(len(lens))], [L - 1 for L in lens]] = 0

    mask_arrays = [keep]

    # repeat until everything is unmasked:
    while not np.all(mask_arrays[-1] == 0):
        # only kept around for assertion checking
        old_keep = keep

        # run next B&B operation, then compute the
        # resulting `keep` state:
        keep = np.where(solve_mask(
            model, ent_bud, seqs, keep,
            mask_value, seqs.shape[0]
        ) == mask_value, 1, 0)

        # if we did not manage to keep any more than last
        # iteration, perhaps because the entropy budget is
        # too high, we MUST change `keep` to prevent a fixed
        # point (hence infinite loop).
        # rather than error, setting `keep` to all-zeroes
        # will terminate the function, which is better than
        # asking the user to exit the program and re-run
        # with a smaller entropy budget!
        if not np.any(keep < old_keep):
            keep = np.zeros_like(keep)

        # the progress must be monotonic:
        assert(np.all(keep <= old_keep))

        # store the new `keep` state:
        mask_arrays.append(keep)

    return seqs, np.stack(mask_arrays)


def compress_serialisation(model, seqs, mask_arrays, mask_value, d):
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

    Returns:
        List of Lists of Ints: Returns the d-ary Huffman codes for each sequence
                               in the batch.
    """
    assert(np.all(mask_arrays[-1] == 0))
    codes = [[] for i in range(seqs.shape[0])]
    last_seqs = None
    last_masks = None
    for masks in mask_arrays:
        if last_seqs is not None:
            # run the model to get its current belief over masked
            # sequences:
            ps = model(last_seqs)

            # compute which new tokens are being revealed at this
            # timestep (possibly none):
            new = last_masks - masks
            assert(np.all(new >= 0))
            # (note that the number of revealed tokens may be different
            # for different sequences)

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
