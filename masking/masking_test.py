import numpy as np
import masking


def test_apply_mask():
    seqs = np.array([[1, 2, 3, 0], [4, 5, 6, 7]], dtype=np.int32)
    mask = (seqs != 0)
    result = masking._apply_mask(
        seqs, mask, 8, 8, 0
    )
    assert(result.shape == seqs.shape)
    assert(result[0, 3] == 0)


def test_bert_masking():
    result = masking.bert_masking(
        [np.array([1, 2, 3], dtype=np.int32),
         np.array([4, 5, 6, 7], dtype=np.int32)],
        0, 8, 8
    )
    assert(result.shape == (2, 4))


def test_span_bert_masking():
    result = masking.span_bert_masking(
        [np.array([1, 2, 3], dtype=np.int32),
         np.array([4, 5, 6, 7], dtype=np.int32)],
        0, 8, 8
    )
    assert(result.shape == (2, 4))


def test_bnb_masking():
    def _model(xs):
        assert(xs.shape == (2, 4))
        ps = np.ones(xs.shape + (16,))
        return ps / np.sum(ps, axis=-1, keepdims=True)

    result = masking.bnb_masking(
        _model,
        [np.array([1, 2, 3], dtype=np.int32),
         np.array([4, 5, 6, 7], dtype=np.int32)],
        0, 8, 8
    )
    assert(result.shape == (2, 4))


def test_cut_sort_masking():
    def _model(xs):
        assert(xs.shape == (2, 4))
        ps = np.ones(xs.shape + (16,))
        return ps / np.sum(ps, axis=-1, keepdims=True)

    result = masking.cut_sort_masking(
        _model,
        [np.array([1, 2, 3], dtype=np.int32),
         np.array([4, 5, 6, 7], dtype=np.int32)],
        0, 8, 8
    )
    assert(result.shape == (2, 4))
