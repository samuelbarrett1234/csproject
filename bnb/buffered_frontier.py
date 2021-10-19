"""
This module contains a wrapper for the NodeFrontier class
which simplifies the interface. Recall that the NodeFrontier
gives the caller several options regarding which data to run
the model on before reporting back. This is tricky to parallelise
across many independent problem instances, whereas the API
below is easier. Namely, with a bit of buffering, this class
makes a node frontier "feel like" a queue of work, executed
one-at-a-time.
"""


class BufferedNodeFrontier:
    """This is a version of the node frontier with simpler "handling".
    By this, it means you don't need to worry about grouping together
    the inputs to pass to `update`. In other words, `get_updates` and
    `update` are rephrased to be push/pop-like, operating one-seq-at-a-time.
    """
    def __init__(self, frontier):
        self.frontier = frontier
        self.out_q, self.in_q = [], []
        self.num_popped = 0
        self.num_popped_required = 0
        # initialise with the first updates
        self._reset_out_q()


    def _reset_out_q(self):
        u, o = self.frontier.get_updates()
        self.out_q = u + o  # (`o` is optional, thus must go second)
        self.num_popped_required = len(u)  # must pop all required updates


    def done(self):
        # if we are in the middle of receiving results (num_popped > 0)
        # then we are certainly not done; else delegate to `self.frontier`.
        if self.num_popped == 0:
            return self.frontier.done()
        else:
            return False


    def pop_update(self):
        if len(self.out_q) == 0:
            return None  # no update available
        self.num_popped += 1
        return self.out_q.pop(0)


    def push_update(self, entropies):
        self.in_q.append(entropies)
        # we should update the frontier if and only if:
        # (i) all of the vectors we expect to see (=`num_popped`) have been seen,
        # (ii) sufficiently many vectors have been extracted
        if len(self.in_q) == self.num_popped and self.num_popped >= self.num_popped_required:
            self.frontier.update(self.in_q)
            self.in_q = []
            self.num_popped = 0
            self._reset_out_q()


    def dual(self):
        return self.frontier.dual()


    def primal(self):
        return self.frontier.primal()


    def primal_value(self):
        return self.frontier.primal_value()
