"""
This compressor creates a d-ary Huffman encoding.
Typically used to reduce the alphabet size in a useful way.
More efficient if the input alphabet is large.
"""


import heapq
import itertools
import progressbar as pgb
from compressors.base import Compressor


class Huffman(Compressor):
    def __init__(self, d):
        self.d = d


    def train(self, alphabet_size, iter_train, iter_val):
        # firstly, estimate the probability distribution of token
        # frequencies in the training dataset.
        # to do this we use the MAP-estimate of a Dirichlet(1)
        # prior on the occurrence frequencies.
        # but there's no need to normalise this probability distribution
        # because the priority queue involved in the Huffman construction
        # is monotonic.

        tok_occ = dict([(i, 1) for i in range(alphabet_size)])  # token occurrences
        for t in pgb.progressbar(itertools.chain(iter_train())):
            tok_occ[t] += 1

        # now compute the prefix-free codebook:
        h = [(v, [(k, [])]) for k, v in tok_occ.items()]
        heapq.heapify(h)

        while len(h) > 1:
            # pop up to `d` least-likely elements
            xs = []
            for i in range(self.d):
                xs.append(heapq.heappop(h))
                if len(h) == 0:
                    break
            
            # now join their codes together
            heapq.heappush(h, (
                # the new probability is the sum of the old ones
                sum(map(lambda t: t[0], xs)),
                # union all of the codebooks
                list(itertools.chain(
                    # but don't forget to prepend a new code element
                    map(
                        lambda i, v_codes: [(k, [i] + code) for k, code in v_codes[1]],
                    enumerate(xs))
                ))
            ))

        self.codebook = dict(h[0][1])

        return self.d


    def compress(self, seq):
        return list(itertools.chain(map(lambda t: list(self.codebook[t]), seq)))


    def compressmany(self, seqs):
        return super(self).compressmany(seqs)
