"""
This class is a different type of node, which is capable of stopping
early, i.e. before optimality is reached. It satisfies the same
interface (except the stopping condition is passed to the constructor).
"""


from bnb.node import Node


class StoppableNode(Node):
    """This is a node augmented with an extra function `stop_cond` which
    takes two integral arguments `primal`, `dual`, and returns true if
    the node should stop. Note: in order to make sense, this condition
    should satisfy `stop_cond(x, x) == True` for all `x`, however this is
    not checked.
    """
    def __init__(self, mask, keep, entropy_budget, stop_cond):
        super(StoppableNode, self).__init__(mask, keep, entropy_budget)
        # bind the right parameters in:
        self.stop_cond = lambda: stop_cond(self._primal, self._dual)

    def dual(self):
        if self.stop_cond():
            # if we have stopped early, we have settled on our current primal
            return self._primal
        else:
            return super(StoppableNode, self).dual()

    def terminal(self):
        return (self.stop_cond() or super(StoppableNode, self).terminal())


def early_stopping_abs_gap(k):
    assert(k >= 0)
    return lambda p, d: (d - p <= k)


def early_stopping_rel_gap(prop):
    assert (prop >= 0.0 and prop <= 1.0)
    return lambda p, d: (1 - p / d <= prop)
