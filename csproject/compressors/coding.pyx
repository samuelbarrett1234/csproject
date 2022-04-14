"""
Contains functions for coding (e.g. Huffman coding) to be compiled
into Cython. To do so, run
`python compressors/setup.py build_ext --inplace`
in the root of the repository directory.
"""


import heapq
import cython


def huffman_codebook(d: cython.int, tokens: dict):
    """Compute the Huffman codebook of output-alphabet-size `d`
    and with token probabilities/occurrences `tokens`.

    Args:
        d (int): The arity of the code. Use d=2 for binary.
        tokens (dict): A dictionary mapping tokens (as keys) to
                       probabilities or occurrences.

    Returns:
        dict: A dictionary mapping tokens to prefix-free lists of
              integers between 0 and d-1.
    """
    h = [(v, [(k, [])]) for k, v in tokens.items()]
    heapq.heapify(h)

    while len(h) > 1:
        # pop up to `d` least-likely elements
        xs = []
        for i in range(d):
            xs.append(heapq.heappop(h))
            if len(h) == 0:
                break

        # the new probability is the sum of the old ones
        new_p = sum(map(lambda t: t[0], xs))
        # union all of the codebooks
        new_code = [(k, [i] + code) for i, t in enumerate(xs) for k, code in t[1]]
        # now join their codes together
        heapq.heappush(h, (new_p, new_code))

    return dict(h[0][1])
