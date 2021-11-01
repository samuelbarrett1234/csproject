import pytest
import numpy as np
from bnb.cut_sort import _batched_merge_sort


@pytest.mark.parametrize("N", [
    64,
    51,
    1,
    10
])
def test_batched_merge_sort(N):
    seqs = np.random.randint(0, 100, size=(64, N))
    original = np.copy(seqs)
    cmp = lambda x, y: -np.sign(x - y)
    _batched_merge_sort(seqs, cmp)
    print(seqs)
    for i in range(64):
        assert(list(seqs[i]) == sorted(list(original[i])))
