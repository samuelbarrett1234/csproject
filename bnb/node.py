"""
This module contains the class representing the individual
nodes in this branch and bound problem. This is where the
core of the optimisation problem lies (in particular, the
formulae for primal and dual bounds, and the variable
branching rules.)
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
        assert(entropy_budget >= 0.0)
        assert(len(mask) == len(keep))
        assert(np.all(np.logical_or(
            mask == 0, keep == 1
        )))
        assert(mask.dtype == np.int32)
        assert(keep.dtype == np.int32)
        self.mask = np.copy(mask)
        self.keep = np.copy(keep)
        self.budget = entropy_budget
        self._primal = sum(mask)
        self._primal_value = mask
        self._dual = sum(keep)
        assert(self._primal <= self._dual)
        self._update_remaining()


    def _update_remaining(self):
        self.remaining = np.where(np.logical_and(self.mask == 0, self.keep == 1), 1, 0)


    def tighten(self, conditioned_entropies):
        """Try to improve the primal/dual bounds of this node after running
        the model. Note that you may be able to tighten further by calling
        this function many times. The input, `entropies`, should be the
        output of running the model *CONDITIONING PRECISELY ON THE INDICES
        WHICH ARE 0 IN THE VECTOR RETURNED BY `self.conditioned_masking()`!*
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
        self.conditioned_entropies = conditioned_entropies

        # automatically improve dual by putting a singleton
        # which does not fit into the budget into the keep set
        # if one exists
        not_fit = np.argwhere(np.logical_and(
            conditioned_entropies > self.budget, self.remaining == 1)
        )
        if len(not_fit) > 0:
            self.keep[not_fit[0]] = 0
        self._update_remaining()
        self._dual = sum(self.keep)
        assert(self._primal <= self._dual)

        # if the dual has improved, but we are still not finished,
        # tell the caller to call `tighten` again (before bothering
        # to update the primal):
        if self._dual < old_dual and not self.terminal():
            return True

        # also compute greedy primal (by stuffing as many singletons
        # into the mask as the entropy budget will allow - obviously
        # this is achieved by sorting the singleton entropies and
        # then greedily going up the list until the cumsum doesn't fit)
        ent_if_possible = np.where(
            self.remaining == 1, conditioned_entropies, np.inf)
        idxs = np.argsort(ent_if_possible)
        cum_sums = np.cumsum(ent_if_possible[idxs])
        i = np.searchsorted(cum_sums, self.budget, side='right')
        idxs = idxs[:i]
        self._primal_value = np.copy(self.mask)
        self._primal_value[idxs] = 1

        # then also update the masking vectors
        self._primal = sum(self._primal_value)
        assert(self._primal <= self._dual)

        return False


    def _branching_index(self, entropies):
        # TODO: consider other strategies; there is no reason why
        # the greedily-pick-the-biggest-entropy strategy below
        # is optimal.
        return np.argmax(
            np.where(
                self.remaining == 1,
                entropies,
                -np.inf
            )
        )


    def branch(self, entropies):
        """Branch the node on the results of an extra call to the model to compute
        conditioned entropies. The input, `conditioned_entropies`, should be the
        output of running the model *CONDITIONING PRECISELY ON THE INDICES
        WHICH ARE 0 IN THE VECTOR RETURNED BY `self.masking()`!*

        Obviously there is no point running this if the node is terminal.

        Args:
            conditioned_entropies (np.ndarray): Resulting entropies computed from the
                                                call to the model with input from
                                                `conditioned_masking()`.

        Returns:
            List of Node: Returns the list of nodes to branch on.
        """
        assert(not self.terminal())
        i = self._branching_index(entropies)
        assert(self.conditioned_entropies[i] <= self.budget)  # otherwise would've been added to `keep`
        mask1, keep2 = np.copy(self.mask), np.copy(self.keep)
        mask1[i] = 1
        keep2[i] = 0
        return [
            Node(mask1, self.keep, self.budget - self.conditioned_entropies[i]),
            Node(self.mask, keep2, self.budget - self.conditioned_entropies[i] + entropies[i])
        ]


    def masking(self):
        """Return a vector which is 0 at all entries which this node will
        definitely keep (hence condition on), and 1 otherwise (i.e. you
        should mask). If this node is terminal, then a 1 indicates you
        *definitely* should mask (i.e. in a committal way) rather than
        just asking you to mask for the next calculation.
        """
        return self.keep


    def conditioned_masking(self):
        """Return a vector which is 0 for all entries for which this node
        has decided what to do (mask or keep) and 1 for all entries which
        are yet undecided.
        """
        return self.remaining


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
        return self._primal_value
