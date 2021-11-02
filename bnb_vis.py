"""
This script contains visualisations of the B&B masking process
on a pretrained BERT model.
"""


import argparse as ap
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForMaskedLM
from compressors.bnb_compression import (serialise_bnb, compress_serialisation, serialise_greedy,
                                         serialise_l2r, serialise_cutting_sort)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForMaskedLM.from_pretrained("bert-base-uncased")
mask_value = tokenizer._convert_token_to_id(tokenizer.mask_token)
pad_value = tokenizer._convert_token_to_id(tokenizer.pad_token)


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("text", type=str, help="The text to apply to.")
    parser.add_argument("ent_bud", type=float, help="Entropy budget.")
    args = parser.parse_args()

    encoded_input = tokenizer(args.text, return_tensors='np')
    seqs = [encoded_input['input_ids'][0].astype(np.int32)]

    def _model(xs):
        # tile extra parameters across the batch
        types = np.repeat(encoded_input['token_type_ids'], xs.shape[0], axis=0)
        att_mask = np.repeat(encoded_input['attention_mask'], xs.shape[0], axis=0)

        ps = tf.nn.softmax(model(
            input_ids=xs,
            token_type_ids=types,
            attention_mask=att_mask
        ).logits, axis=-1).numpy()
        return ps

    results = serialise_bnb(_model, seqs, mask_value, pad_value, args.ent_bud,
                            keep_start_end=True)
    bnb_codes = compress_serialisation(_model, results[0], results[1], mask_value, 2)
    printable_results = np.where(results[1] == 1, mask_value, results[0][np.newaxis, :, :])

    print("B&B REVEAL ORDER")
    for seq in printable_results[:, 0, :]:
        print(tokenizer.convert_tokens_to_string(
            map(tokenizer._convert_id_to_token, seq)))

    results = serialise_l2r(seqs, pad_value)
    l2r_codes = compress_serialisation(_model, results[0], results[1], mask_value, 2)

    results = serialise_cutting_sort(_model, seqs, mask_value, pad_value, keep_start_end=True)
    cut_codes = compress_serialisation(_model, results[0], results[1], mask_value, 2)
    printable_results = np.where(results[1] == 1, mask_value, results[0][np.newaxis, :, :])

    print("CUT REVEAL ORDER")
    for seq in printable_results[:, 0, :]:
        print(tokenizer.convert_tokens_to_string(
            map(tokenizer._convert_id_to_token, seq)))

    results = serialise_greedy(_model, seqs, mask_value, pad_value, keep_start_end=True)
    greedy_codes = compress_serialisation(_model, results[0], results[1], mask_value, 2)
    printable_results = np.where(results[1] == 1, mask_value, results[0][np.newaxis, :, :])

    print("GREEDY REVEAL ORDER")
    for seq in printable_results[:, 0, :]:
        print(tokenizer.convert_tokens_to_string(
            map(tokenizer._convert_id_to_token, seq)))

    print("BNB Code:", "".join(map(str, bnb_codes[0])))
    print("Cutting Code:", "".join(map(str, cut_codes[0])))
    print("Greedy Code:", "".join(map(str, greedy_codes[0])))
    print("Code lengths:")
    print("BNB =", len(bnb_codes[0]), ", L2R =", len(l2r_codes[0]),
          "Cut =", len(cut_codes[0]), "Greedy =", len(greedy_codes[0]))
