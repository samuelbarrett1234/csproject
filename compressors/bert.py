"""
This compressor trains a BERT model and then uses it as a compressor.
There are three variables for this type of model:
- Which initial state do you want to use? (Train from scratch, or from a particular pretrained model)
- Which fine-tuning method do you want to use (None? Or ...?)
- Which compression method do you want to use?

WARNING: you must ensure that the input to this compressor has been correctly
tokenised! E.g., if you are using a pretrained model, the input data should've
been tokenised in exactly the same way, or the tokeniser won't work.
"""


import os
import numpy as np
import tensorflow as tf

# don't eat all GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from transformers import BertTokenizer, TFBertForMaskedLM
from compressors.base import Compressor
import compressors.bnb_compression as bnb_compression
import masking


BATCH_SIZE = 128
MAX_LENGTH = 32
INIT_STATE = [None, 'bert-base-uncased', 'bert-large-uncased']
FINE_TUNING = [None, 'bert', 'span-bert', 'cutting-sort', 'greedy']
COMPRESSION = ['L2R', 'cutting-sort', 'greedy']


def _chop(s):
    # chop a sequence (list of ints) into sublists
    # satisfying MAX_LENGTH
    buf = []
    for t in s:
        buf.append(t)
        if len(buf) == MAX_LENGTH:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


def _reverse_mask_arrays(mask_arrays):
    # swap order
    rev = np.copy(mask_arrays)[::-1, :, :]
    # swap mask and keep
    rev = 1 - rev
    # but: initial states needs to be preserved
    rev = np.where(mask_arrays[:1, :, :] == 0, 0, rev)
    # some checks:
    assert(np.all(rev[0, :, :] == mask_arrays[0, :, :]))
    assert(np.all(rev[1:] - rev[:-1] <= 0))
    assert(np.all(rev[-1, :, :] == 0))
    return rev


class BERT(Compressor):
    def __init__(self, model_dir,
                 init_state, fine_tuning, comp,
                 mask_value=None, pad_value=None,
                 train_repeat=0, out_alphabet_sz=2,
                 reverse_order=False):
        """Create a BERT compressor model.

        Args:
            model_dir (str): The directory to find/put any trained models.
            init_state (str, optional): Must be from `INIT_STATE`.
            fine_tuning (str, optional): Must be from `FINE_TUNING`.
            comp (str): Must be from `COMPRESSION`.
            mask_value (int, optional): The token representing masking. If left None,
                                        `init_state` must not be None, and the pretrained
                                        BERT's value will be inferred.
            pad_value (int, optional): The token representing padding. If left None,
                                       `init_state` must not be None, and the pretrained
                                       BERT's value will be inferred.
            train_repeat (int, optional): The index of the repeat of this training experiment.
            out_alphabet_sz (int, optional): The output alphabet size. Defaults to 2.
            reverse_order (bool, optional): If true, reverse the order of the compression method.
        """
        assert (init_state in INIT_STATE)
        assert (fine_tuning in FINE_TUNING)
        assert (comp in COMPRESSION)

        if tf.test.gpu_device_name(): 
            print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
        else:
            print("WARNING: no GPU found!")

        # the defaults for `mask_value` and `pad_value` are
        # only valid for BERT pretrained tokeniser
        if init_state is None:
            assert (mask_value is not None and pad_value is not None)

        self.init_state = init_state
        self.fine_tuning = fine_tuning
        self.comp = comp
        self.reverse = reverse_order
        self.repeat = train_repeat
        self._model_obj = None  # loaded in `train`

        self._out_alphabet_sz = out_alphabet_sz
        if self.init_state is not None:
            tokenizer = BertTokenizer.from_pretrained(self.init_state)
            # if we are starting from a pretrained model, we expect
            # a specific input alphabet size:
            self._in_alphabet_sz = tokenizer.vocab_size
            self.mask_value = tokenizer._convert_token_to_id(tokenizer.mask_token)
            self.pad_value = tokenizer._convert_token_to_id(tokenizer.pad_token)
        else:
            self._in_alphabet_sz = None
            self.mask_value = mask_value
            self.pad_value = pad_value

        self.model_fname = ''
        if self.init_state is not None:
            self.model_fname += self.init_state + '-'
        if self.fine_tuning is not None:
            self.model_fname += self.fine_tuning + '-'
        self.model_fname += str(self.repeat)
        self.model_fname = os.path.join(
            model_dir, self.model_fname
        )


    def _call_model(self, xs):
        assert (self._model_obj is not None)
        return tf.nn.softmax(self._model_obj(
            input_ids=xs
        ).logits, axis=-1).numpy()


    def train(self, alphabet_size, iter_train, iter_val):
        assert(self._in_alphabet_sz is None or self._in_alphabet_sz == alphabet_size)
        self._in_alphabet_sz = alphabet_size

        # TODO: check if *trained* model exists at `self.model_fname`
        # and if so, load it and DO NOT CONTINUE

        if self.init_state is not None:
            self._model_obj = TFBertForMaskedLM.from_pretrained(self.init_state)
        else:
            # TODO: construct untrained model!
            # is it as simple as TFBertForMaskedLM()?
            # (the key is that the input alphabet size, and values for special
            # parameters, may be different.)
            raise NotImplementedError()

        if self.fine_tuning is not None:
            # apply masking and batching to the iterators
            iter_train = self._make_masked_iter(iter_train)
            iter_val = self._make_masked_iter(iter_val)

            # TODO: train according to the given masking method!
            raise NotImplementedError()

        return self._out_alphabet_sz


    def compress(self, seq):
        return self.compressmany([seq])[0]


    def compressmany(self, seqs):
        chopped_seqs = map(list, map(_chop, seqs))

        csbuf = []  # chopped sequence buffer
        cslenq = []  # FIFO queue containing integers, corresponding to number of elements to read from `csbuf`
        codebuf = []  # buffer containing the resulting codes
        for cs in chopped_seqs:
            csbuf += cs
            cslenq.append(len(cs))

            # while we can run the model on new data
            while len(csbuf) >= BATCH_SIZE:
                # compress available data
                codebuf += self._compress_batch(csbuf[:BATCH_SIZE])
                # reduce buffer
                csbuf = csbuf[BATCH_SIZE:]

            # while we can extract data from the output of the model
            while len(cslenq) > 0 and len(codebuf) >= cslenq[0]:
                # concatenate the first `cslenq[0]` codes:
                result_code = [c for code in codebuf[:cslenq[0]] for c in code]
                yield result_code
                codebuf = codebuf[cslenq[0]:]
                cslenq = cslenq[1:]

        # there may still be data left over to yield
        if len(cslenq) > 0:
            assert(len(csbuf) > 0)
            # arbitrarily extend `csbuf` to fit the batch size
            csbuf += [csbuf[0]] * (BATCH_SIZE - len(csbuf))
            # compress it
            codebuf += self._compress_batch(csbuf)
            # empty it (not actually necessary)
            csbuf = []

            # yield all of the remaining results
            while len(cslenq) > 0:
                assert(len(codebuf) >= cslenq[0])
                # concatenate the first `cslenq[0]` codes:
                result_code = [c for code in codebuf[:cslenq[0]] for c in code]
                yield result_code
                codebuf = codebuf[cslenq[0]:]
                cslenq = cslenq[1:]


    def _compress_batch(self, seqs):
        seqs = list([np.array(xs, dtype=np.int32) for xs in seqs])

        # check dimensions of input
        assert(len(seqs) == BATCH_SIZE)
        largest_seq = max(map(len, seqs))
        assert(largest_seq <= MAX_LENGTH)

        if self.comp == 'L2R':
            seqs, mask_arrays = bnb_compression.serialise_l2r(
                seqs, self.pad_value, keep_start_end=True,
                min_length=MAX_LENGTH
            )
        elif self.comp == 'cutting-sort':
            seqs, mask_arrays = bnb_compression.serialise_cutting_sort(
                self._call_model, seqs, self.mask_value, self.pad_value,
                keep_start_end=True, min_length=MAX_LENGTH
            )
        elif self.comp == 'greedy':
            seqs, mask_arrays = bnb_compression.serialise_greedy(
                self._call_model, seqs, self.mask_value, self.pad_value,
                keep_start_end=True, min_length=MAX_LENGTH
            )
        if self.reverse:
            mask_arrays = _reverse_mask_arrays(mask_arrays)
        codes = bnb_compression.compress_serialisation(
            self._call_model, seqs, mask_arrays, self.mask_value,
            self._out_alphabet_sz
        )
        return codes


    def _make_masked_iter(self, iter):
        # this is nontrivial because:
        # (i) masking is expensive, should cache or only run every few epochs, IDEALLY IN PARALLEL!
        # (ii) the model on which we base the masking procedure is nontrivial, it is an expert model,
        # and there are many possible ways of getting an expert model:
        #   a. the last epoch's model
        #   b. the best model so far, according to validation set NLL
        #   c. maxentnash mixture of models
        # (iii) if we set a max sequence length (and we probably should), in addition to batching
        # we should also split sequences to fit inside this, and just compress them independently.
        raise NotImplementedError()
