"""
Performs gzip compression.
"""


import gzip
from compressors.base import Compressor


class GZip(Compressor):
    def __init__(self, compresslevel=9):
        self.compresslevel = compresslevel


    def train(self, alphabet_size, iter_train, iter_val):
        if alphabet_size != 256:
            raise ValueError("GZip can only accept byte-sized input.")

        return 256  # TODO: check this is a byte


    def compress(self, seq):
        return list(map(int, gzip.compress(seq, compresslevel=self.compresslevel)))


    def compressmany(self, seqs):
        return super(self).compressmany(seqs)
