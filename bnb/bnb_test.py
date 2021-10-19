"""
Test the whole B&B algorithm.
"""


import random
import numpy as np
from bnb import NodeFrontier


def test_simple():
    N = 10  # seq len
    ent_bud = 4.2  # entropy budget
    answer = min(N, int(ent_bud))

    nf = NodeFrontier(np.zeros((N,), dtype=np.int32), np.ones((N,), dtype=np.int32), ent_bud)
    while not nf.done():
        u, o = nf.get_updates()
        # return an arbitrary number of updates between
        # [len(u), len(u) + len(o)]
        n = random.randint(len(u), len(u) + len(o))
        results = np.ones((n, N))  # every token is uniformly random on {0,1}!
        nf.update(results)

    assert(nf.primal() == answer)
    assert(nf.dual() == answer)
