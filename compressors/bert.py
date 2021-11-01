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
from transformers import BertTokenizer, TFBertForMaskedLM
from compressors.base import Compressor
from compressors.block import _batch
import compressors.bnb_compression as bnb_compression
import masking


INIT_STATE = [None, 'bert-base-uncased', 'bert-large-uncased']
FINE_TUNING = [None, 'bert', 'span-bert', 'cutting-sort']
COMPRESSION = ['L2R', 'cutting-sort']


class BERT(Compressor):
    def __init__(self, model_dir, batch_size, mask_value, pad_value,
                 init_state, fine_tuning, comp='L2R',
                 train_repeat=0, out_alphabet_sz=2):
        assert (init_state in INIT_STATE)
        assert (fine_tuning in FINE_TUNING)
        assert (comp in COMPRESSION)

        self.init_state = init_state
        self.fine_tuning = fine_tuning
        self.comp = comp
        self.repeat = train_repeat

        self._out_alphabet_sz = out_alphabet_sz
        if self.init_state is not None:
            # if we are starting from a pretrained model, we expect
            # a specific input alphabet size:
            self._in_alphabet_sz = BertTokenizer.from_pretrained(
                self.init_state).vocab_size()
        else:
            self._in_alphabet_sz = None

        self._batch_size = batch_size
        self.mask_value = mask_value
        self.pad_value = pad_value

        self.model_fname = os.path.join(
            model_dir,
            (self.init_state + '-' or '') +
            (self.fine_tuning + '-' or '') +
            self.comp + '-' + str(self.repeat)
        )


    def train(self, alphabet_size, iter_train, iter_val):
        # apply masking and batching to the iterators
        iter_train = self._make_masked_iter(iter_train)
        iter_val = self._make_masked_iter(iter_val)

        assert(self._in_alphabet_sz is None or self._in_alphabet_sz == alphabet_size)
        self._in_alphabet_sz = alphabet_size

        raise NotImplementedError()
        """
        if trained model already exists:
            load it
        elif self.init_state is not None:
            load pretrained model
        else:
            load untrained model

        if self.fine_tuning is not None and trained model does not exist:
            train it
        """
        return self._out_alphabet_sz


    def compress(self, seq):
        self.compressmany([seq] * self._batch_size)[0]


    def compressmany(self, seqs):
        for batch_codes in map(self._compress_batch,
                               _batch(seqs, self._batch_size)):
            for code in batch_codes:
                yield code


    def _compress_batch(self, seqs):
        if self.comp == 'L2R':
            seqs, mask_arrays = bnb_compression.serialise_l2r(
                seqs, self.pad_value)
        elif self.comp == 'cutting-sort':
            seqs, mask_arrays = bnb_compression.serialise_cutting_sort(
                self.model, seqs, self.mask_value, self.pad_value,
                keep_start_end=True
            )
        codes = bnb_compression.compress_serialisation(
            self.model, seqs, mask_arrays, self.mask_value,
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
