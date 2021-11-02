"""
This compressor creates a d-ary Huffman encoding.
Typically used to reduce the alphabet size in a useful way.
More efficient if the input alphabet is large.
"""


from compressors.coding import huffman_codebook
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
        self.codebook = huffman_codebook(self.d, tok_occ)

        return self.d


    def compress(self, seq):
        return list(itertools.chain(map(lambda t: list(self.codebook[t]), seq)))


    def compressmany(self, seqs):
        return super(self).compressmany(seqs)
