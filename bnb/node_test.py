import numpy as np
from bnb.node import Node


def test_node():
    n = Node(
        np.array([0, 0, 0, 0], dtype=np.int32),
        np.array([1, 1, 1, 0], dtype=np.int32),
        4.0
    )
    n.tighten(np.array([1.0, 1.0, 1.0, 1.0]))
