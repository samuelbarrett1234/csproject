"""
This file contains the notion of a "node frontier". This is
a fairly generic branch-and-bound concept, except I have made
minor modifications to the traditional workflow.
"""


from bnb.node import Node


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


    def done(self):
        """Returns true iff the optimisation has finished. This is the case precisely
        when all nodes are terminated and of maximal value.
        """
        d = self.dual()
        return (all(map(lambda n: n.dual() == d and n.primal() == d, self.terminated))
                and len(self.tightening) == 0 and len(self.branching) == 0)


    def get_updates(self):
        """Get the masking schemes to run the model on next. Note that these are
        separated into two: those which you MUST run, and those which you MAY
        run if it would help fill the batch.

        Returns:
            A 2-tuple of lists of 0-1 masking vectors.
        """
        assert(not self.done())
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
