"""
This module contains the branch and bound algorithm, used
for masking and compression.
"""


import numpy as np


class Node:
    """A single node in the branch and bound optimisation procedure.
    This class works by (i) constructing it, (ii) repeatedly tightening
    the bounds until you are told otherwise by its return value that you
    may stop (each call to `tighten` is preceded by one call to the model)
    and (iii) either the node is terminal or you must branch, in the
    latter case requiring a further call to the model.
    """
    def __init__(self, mask, keep, entropy_budget):
        """Create a new node (in a partially-constructed state.)
        The primal/dual bounds produced *straight away* might not be
        great, as they are just directly inherited from the construction
        of this node. Call `_tighten` at least once to improve them.

        Args:
            mask (np.ndarray): A vector where entries of 1 correspond to
                               the decision to mask, and 0 elsewhere.
            keep (np.ndarray): A vector where entries of 0 correspond to
                               the decision to keep, and 1 elsewhere.
            entropy_budget (float): The entropy budget, > 0.
        """
        assert(entropy_budget > 0.0)
        assert(len(mask) == len(keep))
        assert(np.all(np.logical_or(
            mask == 0, keep == 1
        )))
        self.mask = mask
        self.keep = keep
        self.budget = entropy_budget
        self._primal = sum(mask)
        self._dual = sum(keep)
        assert(self._primal <= self._dual)
        self._update_remaining()


    def _update_remaining(self):
        self.remaining = np.where(np.logical_and(self.mask == 0, self.keep == 1), 1, 0)


    def tighten(self, entropies):
        """Try to improve the primal/dual bounds of this node after running
        the model. Note that you may be able to tighten further by calling
        this function many times. The input, `entropies`, should be the
        output of running the model *CONDITIONING PRECISELY ON THE INDICES
        WHICH ARE 0 IN THE VECTOR RETURNED BY `self.masking()`!*
        This function MUST return False at some point before you stop calling
        it, because future calculations on remaining entropy budget etc rely
        on the most up-to-date information.

        Obviously there is no point running this if the node is terminal.

        Args:
            entropies (np.ndarray): A vector of model prediction entropies
                                    after conditioning on the tokens dictated
                                    by this node. (In theory, the entropies
                                    of those in the keep set should be 0 since
                                    we have conditioned on them, but this
                                    function will not check!)

        Returns:
            bool: True if the bounds were improved (hence it may be worth
                  running again), false if no improvement was made (hence
                  a fixed point reached.)
        """
        assert(not self.terminal())

        # store the old dual bound to check for improvements:
        old_dual = self._dual

        # also store this for later:
        self.entropies = entropies

        # automatically improve dual by putting any singleton
        # which does not fit into the budget into the keep set
        self.keep = np.where(
            np.logical_or(
                self.keep == 0,
                np.logical_and(entropies > self.budget, self.remaining == 1)
            ),
            1, 0
        )
        self._update_remaining()

        # also compute greedy primal (by stuffing as many singletons
        # into the mask as the entropy budget will allow - obviously
        # this is achieved by sorting the singleton entropies and
        # then greedily going up the list until the cumsum doesn't fit)
        idxs = np.argsort(np.where(self.remaining == 1, entropies, np.inf))
        cum_sums = np.cumsum(entropies[idxs])
        i = np.searchsorted(cum_sums, self.budget, side='right')
        idxs = idxs[:i]
        greedy_mask = np.copy(self.mask)
        greedy_mask[idxs] = 1

        # then also update the masking vectors
        self._primal = sum(greedy_mask)
        self._dual = sum(self.keep)
        assert(self._primal <= self._dual)

        # finally, detect change in the dual (we may get improvement by
        # re-running if and only if the dual improves)
        return (self._dual < old_dual)


    def masking(self):
        """Return a vector which is 0 at all entries which this node will
        definitely keep (hence condition on), and 1 otherwise (i.e. you
        should mask). If this node is terminal, then a 1 indicates you
        *definitely* should mask (i.e. in a committal way) rather than
        just asking you to mask for the next calculation.
        """
        return self.keep


    def _branching_index(self, conditioned_entropies):
        # TODO: consider other strategies; there is no reason why
        # the greedily-pick-the-smallest-entropy strategy below
        # is optimal.
        return np.argmin(
            np.where(
                self.remaining == 1,
                conditioned_entropies,
                np.inf
            )
        )


    def branch(self, conditioned_entropies):
        """Branch the node on the results of an extra call to the model to compute
        conditioned entropies. The input, `conditioned_entropies`, should be the
        output of running the model *CONDITIONING PRECISELY ON THE INDICES
        WHICH ARE 0 IN THE VECTOR RETURNED BY `self.conditioned_masking()`!*

        Obviously there is no point running this if the node is terminal.

        Args:
            conditioned_entropies (np.ndarray): Resulting entropies computed from the
                                                call to the model with input from
                                                `conditioned_masking()`.

        Returns:
            List of Node: Returns the list of nodes to branch on.
        """
        assert(not self.terminal())
        i = self._branching_index(conditioned_entropies)
        assert(conditioned_entropies[i] <= self.budget)  # otherwise would've been added to `keep`
        mask1, keep2 = np.copy(self.mask), np.copy(self.keep)
        mask1[i] = 1
        keep2[i] = 0
        return [
            Node(mask1, self.keep, self.budget - conditioned_entropies[i]),
            Node(self.mask, keep2, self.budget - conditioned_entropies[i] + self.entropies[i])
        ]


    def conditioned_masking(self):
        """Return a vector which is 0 for all entries for which this node
        has decided what to do (mask or keep) and 1 for all entries which
        are yet undecided.
        """
        return np.remaining


    def terminal(self):
        """Is this node terminal (i.e. solved)?
        """
        return (self._primal == self._dual)


    def primal(self):
        """Get the primal bound of this node.
        """
        return self._primal


    def dual(self):
        """Get the dual bound of this node.
        """
        return self._dual


    def primal_value(self):
        """Return a vector which masks `primal()` tokens and satisfies the
        entropy budget constraints.
        """
        return self.mask


class NodeFrontier:
    """Represents a collection of the above nodes, all solving the same problem
    instance. Note that this might look slightly different to normal B&B algorithms
    for two reasons: (i) we need to tighten each node an arbitrary number of times
    before we can branch on it or class it as terminated, or (ii) in the wake of
    graph recompilation, it may be optimal to look ahead slightly when branching
    rather than just branching one node at a time.
    """
    def __init__(self, mask, keep, entropy_budget):
        self.tightening = [Node(mask, keep, entropy_budget)]
        self.branching = []
        self.terminated = []


    def get_updates(self):
        """Get the masking schemes to run the model on next. Note that these are
        separated into two: those which you MUST run, and those which you MAY
        run if it would help fill the batch.

        Returns:
            A 2-tuple of lists of 0-1 masking vectors.
        """
        t, b = [n.masking() for n in self.tightening] + [n.conditioned_masking() for n in self.branching]
        return t + [b[0]], b[1:]


    def update(self, results):
        """Update this branch and bound instance with the result of running the model
        on the information we provided in the return of `get_updates()`.

        IMPORTANT: the input `results` should be the output of running the model on
        every element of the *first tuple* returned from `get_updates()`, and any
        number of elements of the *second tuple* returned from `get_updates()`, with
        order maintained (implying that all elements from the first tuple occur before
        any elements of the second tuple).

        In terms of the precise model output: this function wants the *entropies* of the
        predicted label distributions at each position!

        Args:
            results (list of np.ndarray, or 2d np.ndarray): The entropies resulting from
                                                            running the model.
        """ 
        to_tighten, to_branch = [], []
        states = [True] * len(self.tightening) + [False] * len(self.branching)

        for entropies, n, is_tightening in zip(results, self.tightening + self.branching, states):
            # update the node, according to its current state
            if is_tightening:
                if n.tighten(entropies):
                    to_tighten.append(n)
                elif not n.terminal():
                    to_branch.append(n)
                else:
                    self.terminated.append(n)
            else:
                for b in n.branch(entropies):
                    if b.terminal():
                        self.terminated.append(b)
                    else:
                        to_tighten.append(b)

        # update our lists
        self.tightening = to_tighten
        self.branching = to_branch

        # now filter any node which provably is suboptimal:
        p = self.primal()
        self.tightening = [n for n in self.tightening if n.dual() >= p]
        self.branching = [n for n in self.branching if n.dual() >= p]
        self.terminated = [n for n in self.terminated if n.dual() >= p]

        # now sort the branching set based on priorities:
        self.branching = sorted(self.branching, key=lambda n: -n.dual())


    def dual(self):
        return max(map(lambda n: n.dual(), self.tightening + self.branching + self.terminated))


    def primal(self):
        return max(map(lambda n: n.primal(), self.tightening + self.branching + self.terminated))


    def primal_value(self):
        p = self.primal()
        for n in self.tightening + self.branching + self.terminated:
            if n.primal() == p:
                return n.primal_value()
