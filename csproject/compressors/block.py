"""
This compressor doesn't really do any compressing, and just
alters the alphabet to put the labels into blocks. For example
if the input is binary, and the block size is 8, the output
will be bytes!
"""


from compressors.base import Compressor


def _batch(iter, n):
    """Create a new iterator which batches the elements of `iter`
    into groups of at most `n`, with possibly the last batch
    being shorter.
    """
    buffer = []
    for i in iter:
        buffer.append(i)
        if len(buffer) == n:
            yield buffer
            buffer = []
    if len(buffer) > 0:
        yield buffer


class Block(Compressor):
    def __init__(self, n):
        self.n = n


    def train(self, alphabet_size, iter_train, iter_val):
        self.input_alphabet_size = alphabet_size

        # this is not just alphabet_size ** self.n, because we
        # have no guarantee (and thus must prepare for) data not
        # fitting entirely into blocks of size n, e.g. whenever
        # the sequence length is not a multiple of n.
        # (this is equivalent to extending the input alphabet with a
        # padding character, but excluding the single possibility
        # of an all-padding block of length n)
        self.output_alphabet_size = (alphabet_size + 1) ** self.n - 1

        return self.output_alphabet_size


    def compress(self, seq):
        # need to batch the input, taking into account the possibility
        # that the sequence length will may not be a multiple of n:
        for toks in _batch(seq, self.n):
            # extend with padding:
            toks = toks + [self.input_alphabet_size] * (self.n - len(toks))
            # get the output alphabet token corresponding to the entire
            # block
            block_tok = sum([tok * (self.input_alphabet_size + 1) ** i
                             for i, tok in enumerate(toks)])
            assert(block_tok < self.output_alphabet_size)
            yield block_tok


    def compressmany(self, seqs):
        return super(self).compressmany(seqs)
