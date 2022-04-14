import abc


class Compressor(abc.ABC):
    @abc.abstractmethod
    def train(self, alphabet_size, iter_train, iter_val):
        """This is called before `compress` is ever called, to allow
        the compressor to prepare. In particular, this tells the
        compressor about the alphabet to expect. These will all be
        integers in the range [0, alphabet_size).

        This function can print whatever it wants, but if it wants any of
        it to be persistent, this function should implement its own logging.

        Args:
            alphabet_size (int): The range of tokens to compress.
            iter_train (nullary function): Calling this creates an iterator
                                           which yields integer sequences
                                           corresponding to the training set
                                           in a newly shuffled order.
            iter_val (nullary function): Calling this creates an iterator
                                         which yields integer sequences
                                         corresponding to the validation set
                                         (not shuffled).

        Returns:
            The alphabet size of the output of this compressor.
        """
        pass


    @abc.abstractmethod
    def compress(self, seq):
        """Compress a single integer sequence of tokens.

        Args:
            seq (list of int): The sequence to compress.
        
        Returns:
            The compressed sequence, as a list of ints, but possibly with
            a different alphabet size, as dictated by the return value of
            `train`.
        """
        pass


    @abc.abstractmethod
    def compressmany(self, seqs):
        """Compress a collection of integer sequences. Base classes
        may be able to optimise this method, either on the GPU
        or with thread pooling for large amounts of sequences.
        However, obviously, the order of the output must equal the
        order of input, because otherwise there will be no way to
        associate the compressed sequences with the originals.

        Args:
            seqs (Iterator over integer sequences): A list, or generic iterator,
                                                    over a (potentially very large)
                                                    selection of 

        Returns:
            An iterator over (the compressed) integer sequences.
            As discussed above, the output alphabet size may be different.
        """
        return map(self.compress, seqs)


    def fine_tuning_method(self):
        return None


    def comp_method(self):
        return None
