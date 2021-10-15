"""
This compressor doesn't really do any compressing, and just
alters the alphabet to put the labels into blocks. For example
if the input is binary, and the block size is 8, the output
will be bytes!
"""


from compressors.base import Compressor


class Block(Compressor):
    def __init__(self, n):
        self.n = n


    def train(self, alphabet_size, iter_train, iter_val):
        self.input_alphabet_size = alphabet_size
        return alphabet_size ** self.n


    def compress(self, seq):
        return sum([tok * self.input_alphabet_size ** i for i, tok in enumerate(seq)])


    def compressmany(self, seqs):
        return super(self).compressmany(seqs)
