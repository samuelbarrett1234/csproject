"""
This compressor trains and runs a chain of compressors.
It uses an intermediate temporary file to store compression results.
Because in order to train the 2nd compressor you need the 1st compressor
to have trained over all of the data.
"""


import os
import random
import tempfile
from compressors.base import Compressor


def _shuffle_file_lines(file, buffer_sz=1024*8):
    """(Approximately) shuffle the lines of a (potentially very large) file.

    Args:
        file (file handle): The file to shuffle.
        buffer_sz (int): The max number of lines we are allowed to read in at a time.
    """
    end = False
    while not end:
        buffer = []
        pos = file.tell()
        for i in range(buffer_sz):
            s = file.readline()
            if s != '':
                if s[-1] != '\n':
                    s = s + '\n'
                buffer.append(s)
            else:  # eof
                end = True
                break
        random.shuffle(buffer)
        file.seek(pos)
        file.write("".join(buffer))  # should be identical size in binary


def _lazy_iterate(compressor, old_iter, shuffle):
    """Create a new iterator from an old iterator by applying
    the given compressor to all elements. To save on computation
    the results are stored in a temporary file which is closed
    when the return result of this function is garbage collected.
    But also, since this iterator may never be needed, we do it
    lazily to prevent excessive computation.

    Args:
        compressor (Compressor): The compressor to apply.
        old_iter (Nullary function): Calling this creates an iterator
                                     over integer list sequences.
        shuffle (bool): If true, shuffle the data every time the
                        iterator is created, else preserve order of
                        the old iterator.

    Returns:
        A new iterator.
    """

    # unfortunately we must rely on the garbage collection of
    # `new_iter` to clean this temporary file up, because while
    # `new_iter` exists we could always receive another call
    # to it.
    storage = tempfile.TemporaryFile(mode='w+', newline='\n')

    def new_iter():
        storage.seek(0, os.SEEK_END)

        if storage.tell() != 0:
            # file is nonempty; don't need `old_iter`, just go
            # through the lines of `storage`
            storage.seek(0)

            if shuffle:
                # over time, this approximate shuffle will get better
                # and better, because it overwrites the file each time.
                _shuffle_file_lines(storage)

            for line in storage.readline():
                yield list(map(int, line.split()))

        else:
            # go through the old iterator only this once, writing
            # the sequences as we see them, after compressing
            for s in old_iter():
                s = compressor.compress(s)
                storage.write(" ".join(map(str, s)) + "\n")
                yield s

    return new_iter


class Chain(Compressor):
    def __init__(self, compressors):
        assert(all(map(lambda c: issubclass(type(c), Compressor), compressors)))
        self.compressors = compressors


    def train(self, alphabet_size, iter_train, iter_val):
        for c in self.compressors:
            alphabet_size = c.train(alphabet_size, iter_train, iter_val)

            # now we need to update `iter_train` and `iter_val`.
            # we could simply do this by mapping `c.compress` over them,
            # but this is inefficient.
            # however, another complication is the fact that not all
            # compressors even use `train`!
            # therefore we only want to put all of this data into a
            # file when it is called, not eagerly.
            iter_train = _lazy_iterate(c, iter_train, True)
            iter_val = _lazy_iterate(c, iter_val, False)

        return alphabet_size


    def compress(self, seq):
        for c in self.compressors:
            seq = c.compress(seq)
        return seq


    def compressmany(self, seqs):
        # cool fact: this function will often not perform any work,
        # and will simply return an iterator over many composed compressors
        for c in self.compressors:
            seqs = c.compressmany(seqs)
        return seqs


    def fine_tuning_method(self):
        # return first non-None return result from the children
        methods = [c.fine_tuning_method() for c in self.compressors]
        methods = [m for m in methods if m is not None]
        if len(methods) == 0:
            return None
        else:
            return methods[0]

    def comp_method(self):
        # return first non-None return result from the children
        methods = [c.comp_method() for c in self.compressors]
        methods = [m for m in methods if m is not None]
        if len(methods) == 0:
            return None
        else:
            return methods[0]
