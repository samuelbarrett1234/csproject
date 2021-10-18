"""
Performs zlib compression.
"""


import zlib
from compressors.base import Compressor


class ZLib(Compressor):
    def __init__(self, level=9):
        self.level = level


    def train(self, alphabet_size, iter_train, iter_val):
        if alphabet_size != 256:
            raise ValueError("ZLib can only accept byte-sized input.")

        return 256  # TODO: check this is a byte


    def compress(self, seq):
        return list(map(int, zlib.compress(seq, level=self.level)))


    def compressmany(self, seqs):
        return super(self).compressmany(seqs)
