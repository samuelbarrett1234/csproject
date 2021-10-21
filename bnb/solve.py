"""
This module collects everything from this package together
as a single function.
"""


import numpy as np
from bnb.node import Node
from bnb.stoppable_node import StoppableNode
from bnb.frontier import NodeFrontier
from bnb.frontier_batch import NodeFrontierBatch


def solve_mask(model, entropy_budget, seqs, init_keeps, mask_value,
               batch_size, early_stopping_cond=None):
    """Solve a masking problem for the given collection of sentences.
    For efficiency, this function allows you to specify several sequences.
    Important note: the number of sequences is, in general, distinct from
    the batch size. This is because the B&B optimisation problem *for
    each sequence* will have many nodes, and will in turn, be parallelised.
    Specifying initial state `init_keeps` allows you to call this repeatedly,
    to incrementally reveal tokens. Use `padded_batch` as a convenience
    function to transform to this state.
    Important note: all sequences must have the same length, denoted N here
    in this documentation.

    Args:
        model (function): The model function must take as input an integral
                          matrix of shape (batch-size, N) and output the
                          probability distributions at each position of
                          the tokens.
        seqs (np.ndarray): Integral array of shape (M, N) for *any* value
                           of M, which does not contain `mask_value`.
        init_keeps (np.ndarray): Binary array of shape (M, N) where `0`
                                 denotes the requirement that the token at
                                 that position must be kept. By default you
                                 will want to pick this to be an all-1 matrix
                                 however for repeated calls to this function
                                 you may want to set elements to 0 which were
                                 previously elected to be kept.
        mask_value (int): The value which, when provided as input to the model,
                          will be understood as a masked token. This must NOT
                          form part of the model's output predictions! This must
                          also NOT be present in `seqs`.
        batch_size (int): The batch size the model is expected to receive, which
                          is distinct from `M`, the number of sequences provided
                          as input.
        early_stopping_cond (optional function): Either `None` if you want to solve
                                                 to optimality, or otherwise, it
                                                 should be a function operating on
                                                 two integers (primal, dual), and
                                                 should return true for stopping
                                                 early.
    """
    assert(not np.any(seqs == mask_value))
    assert(seqs.shape == init_keeps.shape)
    assert(entropy_budget >= 0.0)

    # a function for creating the root node of a tree
    # (the type of which depends on whether we are implementing
    # early stopping)
    def _create_root(init_keep):
        init_mask = np.zeros_like(init_keep)
        if early_stopping_cond is not None:
            return StoppableNode(
                init_mask, init_keep, entropy_budget,
                early_stopping_cond)
        else:
            return Node(
                init_mask, init_keep, entropy_budget)

    nfs = NodeFrontierBatch(
        [
            NodeFrontier(_create_root(init_keep))
            for init_keep in init_keeps
        ],
        seqs, mask_value, batch_size
    )

    def _ent(ps):  # entropy
        return -np.sum(np.where(ps > 0.0, ps * np.log(ps), 0.0), axis=-1)

    while not nfs.done():
        nfs.update(_ent(model(nfs.get_updates())))

    # TODO: return entropy budget remaining

    return nfs.solutions()


def padded_batch(seqs, pad_token):
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
    """
    assert(not np.any(seqs == pad_token))
    N = max(map(len, seqs))
    seqs = np.stack(
        [np.concatenate((
            s,
            pad_token * np.ones((N - len(s)), dtype=s.dtype)
            )) for s in seqs]
    )
    init_keep = np.where(
        seqs == pad_token,
        0, 1
    )
    return seqs, init_keep
