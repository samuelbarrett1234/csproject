import pytest
import numpy as np
from bnb import NodeFrontierBatch, NodeFrontier, Node
from bnb.frontier_batch import _zigzag_frontiers


class MockFrontier:
    def __init__(self, vals):
        self.vals = vals

    def done(self):
        return len(self.vals) == 0

    def pop_update(self):
        if len(self.vals) == 0:
            return None
        else:
            return self.vals.pop(0)


def test_zigzag_frontier():
    fs = [MockFrontier([0]), MockFrontier([1, 3, 4]), MockFrontier([]), MockFrontier([2])]
    assert(list(_zigzag_frontiers(fs)) == [
        (0, 0),
        (1, 1),
        (2, 3),
        (3, 1),
        (4, 1)
    ])


@pytest.mark.parametrize("N,ent_bud", [
    (10, 4.2),
    (5, 7.3),  # masking everything fits into the budget
    (100, 15.2),  # large
    (10, 4.0),  # integral budget
    (10, 1.1),  # small budget
    (10, 0.9),  # nothing can be masked
    (10, 0.0),  # nothing can be masked
    ])
def test_frontier_batch_simple(N, ent_bud):
    answer = min(N, int(ent_bud))

    nf = NodeFrontierBatch(
        [
            NodeFrontier(
                Node(np.zeros((N,), dtype=np.int32), np.ones((N,), dtype=np.int32), ent_bud)
            ),
            NodeFrontier(
                Node(np.zeros((N,), dtype=np.int32), np.ones((N,), dtype=np.int32), ent_bud)
            )
        ],
        [
            np.zeros((N,), dtype=np.int32),
            np.zeros((N,), dtype=np.int32)
        ],
        1, 4
    )
    while not nf.done():
        S = nf.get_updates()
        results = np.ones_like(S)  # every token is uniformly random on {0,1}!
        nf.update(results)

    sols = nf.solutions()
    assert(np.sum(sols[0]) == answer)
    assert(np.sum(sols[1]) == answer)
