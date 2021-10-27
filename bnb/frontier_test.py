import random
import pytest
import numpy as np
from bnb import (Node, StoppableNode, early_stopping_abs_gap,
                 NodeFrontier)


@pytest.mark.parametrize("N,ent_bud", [
    (10, 4.2),
    (5, 7.3),  # masking everything fits into the budget
    (100, 15.2),  # large
    (10, 4.0),  # integral budget
    (10, 1.1),  # small budget
    (10, 0.9),  # nothing can be masked
    (10, 0.0),  # nothing can be masked
    ])
def test_frontier_simple(N, ent_bud):
    answer = min(N, int(ent_bud))

    nf = NodeFrontier(
        Node(np.zeros((N,), dtype=np.int32), np.ones((N,), dtype=np.int32), ent_bud)
    )
    while not nf.done():
        u, o = nf.get_updates()
        # return an arbitrary number of updates between
        # [len(u), len(u) + len(o)]
        n = random.randint(len(u), len(u) + len(o))
        results = np.ones((n, N))  # every token is uniformly random on {0,1}!
        nf.update(results)

    assert(nf.primal() == answer)
    assert(nf.dual() == answer)


@pytest.mark.parametrize("N,ent_bud", [
    (10, 4.2),
    (5, 7.3),  # masking everything fits into the budget
    (100, 15.2),  # large
    (10, 4.0),  # integral budget
    (10, 1.1),  # small budget
    (10, 0.9),  # nothing can be masked
    (10, 0.0),  # nothing can be masked
    ])
def test_frontier_early_stopping(N, ent_bud):
    answer = min(N, int(ent_bud))
    k = 3

    nf = NodeFrontier(
        StoppableNode(np.zeros((N,), dtype=np.int32), np.ones((N,), dtype=np.int32),
                      ent_bud, early_stopping_abs_gap(k))
    )
    while not nf.done():
        u, o = nf.get_updates()
        # return an arbitrary number of updates between
        # [len(u), len(u) + len(o)]
        n = random.randint(len(u), len(u) + len(o))
        results = np.ones((n, N))  # every token is uniformly random on {0,1}!
        nf.update(results)

    assert(answer - nf.primal() <= k)
    assert(nf.dual() == nf.primal())


def test_frontier_nontrivial_initial_state():
    N = 10
    ent_bud = 4.2
    answer = min(N, int(ent_bud))

    init_mask, init_keep = np.zeros((N,), dtype=np.int32), np.ones((N,), dtype=np.int32)
    init_keep[0] = 0
    init_mask[1] = 1
    answer += 1  # accounting for the extra mask

    nf = NodeFrontier(Node(init_mask, init_keep, ent_bud))
    while not nf.done():
        u, o = nf.get_updates()
        # return an arbitrary number of updates between
        # [len(u), len(u) + len(o)]
        n = random.randint(len(u), len(u) + len(o))
        results = np.ones((n, N))  # every token is uniformly random on {0,1}!
        nf.update(results)

    assert(nf.primal() == answer)
    assert(nf.dual() == answer)


def test_frontier_with_terminal_root():
    nf = NodeFrontier(
        Node(np.zeros((1,), dtype=np.int32),
             np.zeros((1,), dtype=np.int32),
             0.0)
    )
    assert(nf.done())


def test_frontier_dual_bound_problems():
    N = 3
    ent_bud = 1.0
    answer = 2

    # test the frontier with a model which causes an older
    # version of the dual bound to fail
    nf = NodeFrontier(
        Node(np.zeros((N,), dtype=np.int32), np.ones((N,), dtype=np.int32), ent_bud)
    )
    while not nf.done():
        u, o = nf.get_updates()
        u += o
        u = np.stack(u)
        results = np.zeros_like(u, dtype=np.float32)

        # in this model, we have three variables X, Y and Z
        # Y and Z are conditionally independent given X
        # X, Y|X and Z|X have entropies/conditional entropies 0.51, 0.5, 0.5 respectively
        # X|Y and X|Z have zero entropy

        not_known = np.logical_not(np.logical_or(u[:, 0] == 0, np.logical_or(u[:, 1] == 0, u[:, 2] == 0)))

        results[:, 0] = np.where(not_known, 0.51, 0.0)
        results[:, 1] = np.where(u[:, 1] == 1, np.where(not_known, 1.01, 0.5), 0.0)
        results[:, 2] = np.where(u[:, 2] == 1, np.where(not_known, 1.01, 0.5), 0.0)

        nf.update(results)

    assert(nf.primal() == answer)
    assert(nf.dual() == answer)