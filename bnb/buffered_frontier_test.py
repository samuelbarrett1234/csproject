import pytest
import numpy as np
from bnb import BufferedNodeFrontier


class MockNodeFrontier:
    def __init__(self):
        self.updated = False

    def done(self):
        return self.updated

    def get_updates(self):
        assert(not self.done())
        return [1, 2], [3, 4]

    def update(self, ls):
        assert(ls[:2] == [1, 2])
        if len(ls) >= 3:
            assert(ls[2] == 3)
        if len(ls) >= 4:
            assert(ls[3] == 4)
        assert(len(ls) <= 4)
        self.updated = True


@pytest.mark.parametrize("n,", [(2,), (3,), (4,)])
def test_buffered_frontier(n):
    n = n[0]
    mf = MockNodeFrontier()
    bf = BufferedNodeFrontier(mf)
    xs = []
    for i in range(n):
        xs.append(bf.pop_update())
        assert(not mf.updated)
    for x in xs:
        assert(not mf.updated)
        bf.push_update(x)
    assert(mf.updated)
