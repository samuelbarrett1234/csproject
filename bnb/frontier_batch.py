"""
This module contains the branch and bound algorithm, used
for masking and compression.
"""


import numpy as np
from bnb.buffered_frontier import BufferedNodeFrontier


class NodeFrontierBatch:
    """This allows you to work on several independent node frontiers in
    parallel, such that the calls to the model are always yielded in
    multiples of the batch size. This class will also encapsulate sequence
    masking, so you must provide it with the sequences that the frontiers
    are modelling.
    """
    def __init__(self, init_frontiers, sequences, mask_value, batch_size):
        """Create a batch of frontiers from the given list of initial
        states.
        """
        # check that the sequences map to the frontiers in 1:1 correspondence
        # and that all sequences are of the same length
        assert(len(sequences) == len(init_frontiers))
        assert(all(map(lambda s: len(s) == len(sequences[0]), sequences[1:])))

        # note: all frontiers must be buffered!
        def _buffer_frontier(f):
            if not isinstance(f, BufferedNodeFrontier):
                return BufferedNodeFrontier(f)
            else:
                return f

        self.frontiers = list(map(_buffer_frontier, init_frontiers))

        self.seqs = np.stack(sequences)
        self.seq_len = len(sequences[0])
        self.mask_value = mask_value
        self.batch_size = batch_size
        self.cur_sources = []  # list of indices


    def get_updates(self):
        """Compute the next set of sequences and masks to run the model on.

        Returns:
            np.ndarray of shape (self.batch_size, self.seq_len).
        """
        assert(not self.done())

        masks = np.zeros((self.batch_size, self.seq_len))

        # extract up to `self.batch_size` new sequences to run on
        self.cur_sources = []
        for i, m in zip(range(self.batch_size), _zigzag_frontiers(self.frontiers)):
            masks[i, :] = m[0]
            self.cur_sources.append(m[1])

        result_seqs = self.mask_value * np.ones((self.batch_size, self.seq_len),
                                                dtype=np.int32)
        result_seqs[:len(self.cur_sources), :] = self._apply_mask(
            np.stack(masks, axis=0)[:len(self.cur_sources), :],
            np.array(self.cur_sources, dtype=np.int32)
        )

        # if we couldn't find enough, then we must extend the batch
        # arbitrarily to be the right size, but we need to make a note
        # of this fact
        self.cur_sources += [None] * (self.batch_size - len(self.cur_sources))

        return result_seqs


    def update(self, entropies):
        """Report the results of executing the model on the returned sequences
        from `get_updates`. The input should be the entropies of the predicted
        distributions at each sequence position in each batch.
        """
        assert(entropies.shape == (self.batch_size, self.seq_len))
        for i, entropy in zip(self.cur_sources, entropies):
            if i is not None:
                self.frontiers[i].push_update(entropy)


    def _apply_mask(self, mask, idxs):
        seqs = np.take(self.seqs, idxs, axis=0)
        return np.where(mask == 1, self.mask_value, seqs)


    def done(self):
        return all(map(lambda f: f.done(), self.frontiers))


    def solutions(self):
        assert(self.done())
        return self._apply_mask(
            np.stack([f.primal_value() for f in self.frontiers], axis=0),
            np.array(range(len(self.frontiers)), dtype=np.int32)
        )


def _zigzag_frontiers(fs):
    """Given a list of *BUFFERED* frontiers, yield updates from them one-by-one,
    until they are all exhausted.

    Yields: 2-tuples of the form (<masking-vector>, <original-index-of-frontier-in-`fs`>)
    """
    fs = list(zip(fs, range(len(fs))))  # keep original locations around
    fs2 = []  # double-buffering for `fs`
    # preprocessing step: return any frontiers which are finished
    # before we even do anything
    fs = [f for f in fs if not f[0].done()]
    while len(fs) > 0:
        for f in fs:
            m = f[0].pop_update()
            if m is not None:
                yield m, f[1]
                fs2.append(f)
            # else drop `f`, as it is exhausted
        fs = fs2
        fs2 = []
