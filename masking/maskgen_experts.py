"""
Some of the masking functions we are considering (indeed, all of the
novel ones) require relying on a model in order to generate masks
to fine tune a different model.

Here we refer to the relied-on model as the 'expert'. It is used for
generating masks for later epochs. There are many ways to pick an
expert, and the classes in this file are how.

Note that these classes also implement caching, to avoid calling the
expensive expert model more than necessary. The cache is only updated
when the expert model changes.

These classes should feel like "iterator factories", constructing
one-shot iterators on a dataset, but which have internal data (the
experts) which can be updated inbetween dataset iterations and which
will change the value of the iterator as a result.

NOTE: `update` must be called once before the class is ever called!
This is to allow experts to be generated before any iterating begins.
"""


import pickle
import tempfile
import numpy as np
import tensorflow as tf


class MaskGeneratorExpert:
    def __init__(self, iterator_constructor, masking_func, batch_size):
        """
        `iterator_constructor` must be a function which, when called,
        returns an iterator over integral sequences.
        `masking_func` must be a function accepting two arguments - a
        callable model and a batch of sequences - and must have the same
        return value as the public masking functions in `masking_functions.py`.
        """
        self.iter = iterator_constructor
        self.mask = masking_func
        self.batch_size = batch_size
        self.cache = None


    def __call__(self):
        # TODO: since all arrays are the same size / regular,
        # it should be possible to massively speed this up.
        if self.cache is not None:
            # if cache exists, just load the pickled values in the
            # file one-by-one
            self.cache.seek(0)
            while True:
                try:
                    yield pickle.load(self.cache)
                except EOFError:
                    break
        else:
            # if cache does not exist, create it as a temporary file
            self.cache = tempfile.TemporaryFile(mode='wb+')
            buffer = []
            for seq in self.iter():
                # collect the sequences into a buffer
                buffer.append(seq)
                # yield the buffer when it has reached the right size
                if len(buffer) == self.batch_size:
                    results = self.mask(self._call_model, buffer)
                    yield results
                    pickle.dump(results, self.cache)
                    buffer = []
            # personally I'm happy with dropping the last batch
            # if it is not the right shape


    def update(self, newmodel):
        raise NotImplementedError()


    def _call_model(self, xs):
        raise NotImplementedError()


class LastNMaskGeneratorExpert(MaskGeneratorExpert):
    """Group the epochs together into groups of size N, and use
    a uniform weighting across the *previous* group to generate
    the masks for all models in the next group. Larger N means more
    stable and more efficient.
    """
    def __init__(self, N, iterator_constructor, masking_func, batch_size):
        super(type(self), self).__init__(iterator_constructor, masking_func, batch_size)
        assert(N > 0)
        self.N = N
        self.cur_experts = []
        self.upcoming_experts = []


    def update(self, newmodel):
        if len(self.cur_experts) == 0:  # special case only occurring at the first epoch
            self.cur_experts = [newmodel]
        else:
            assert(len(self.upcoming_experts) < self.N)
            self.upcoming_experts.append(newmodel)
            if len(self.upcoming_experts) == self.N:
                # reset cache and update with new expert group
                self.cur_experts = self.upcoming_experts
                self.upcoming_experts = []
                self.cache = None


    def _call_model(self, xs):
        ps = np.stack([
            tf.nn.softmax(expert(xs).logits, axis=-1).numpy()
            for expert in self.cur_experts])
        return np.mean(ps, axis=0)
