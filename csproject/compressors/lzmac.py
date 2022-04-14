"""
Performs LZMA compression.
"""


import lzma
from compressors.base import Compressor


class LZMA(Compressor):
    def train(self, alphabet_size, iter_train, iter_val):
        if alphabet_size != 256:
            raise ValueError("ZLib can only accept byte-sized input.")

        return 256  # TODO: check this is a byte


    def compress(self, seq):
        return list(map(int, lzma.compress(bytes(seq))))


    def compressmany(self, seqs):
        return super(LZMA, self).compressmany(seqs)
