"""An alternative to `pair_up.py` when you have lots of
label types, each of which not taking many values.
Hopefully this should produce more compact data, rather
than spitting out millions of pairs.
"""


import os
import itertools
import argparse as ap
import sqlite3 as sql
import progressbar as pgb
from pair_up import add_sequences, get_reflexive_sequences


def get_pairs_all_lbltypes(db, n, m):
    """For each labeltype, for each label value,
    for each non-training sequence partition, randomly
    pick `n` sequences. Likewise for the training
    partition, with `m` sequences. Within each label
    type, these are then paired up.
    
    Output size should be:
    2 * num-label-types * num-combs-of-lbls * n * m
    where num-combs-of-lbls = (num-values-of-lbltype) ^ 2
    if all label types have the same number of label values.
    """
    q = """SELECT seqid FROM
    Sequences NATURAL JOIN Labels
    WHERE seqpart = ? AND lbltype = ? AND lbl = ?
    ORDER BY RANDOM() LIMIT ?
    """
    cur = db.cursor()
    cur.execute("""
    SELECT lbltype FROM LabelTypes
    """)
    lbltypes = list(map(lambda t: t[0], cur.fetchall()))
    for lbltype in lbltypes:
        cur.execute("""
        SELECT lbl FROM LabelDictionary
        WHERE lbltype = ?
        """, (lbltype,))
        lbls = list(map(lambda t: t[0], cur.fetchall()))
        data = [[], [], []]  # sequences for each seqpart to pair
        for lbl in lbls:
            for seqpart, size in [(0, m), (1, n), (2, n)]:
                cur.execute(q, (seqpart, lbltype, lbl, size))
                data[seqpart] += list(map(lambda t: t[0], cur.fetchall()))
        for seqid_train in data[0]:
            for seqid_val in data[1]:
                yield (seqid_train, seqid_val, 1)
                yield (seqid_val, seqid_train, 1)
            for seqid_val in data[2]:
                yield (seqid_train, seqid_val, 2)
                yield (seqid_val, seqid_train, 2)


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("db", type=str,
                        help="Filename of the DB to load.")
    parser.add_argument("n", type=int,
                        help="Number of TRAINING seqs to pick for each label/labeltype.")
    parser.add_argument("m", type=int,
                        help="Number of VAL/TEST seqs each to pick for each label/labeltype.")
    parser.add_argument("--use-comma", type=str, default=None,
                        help="If set, use the given token's ID for commas. Else "
                             "generate a new character.")
    parser.add_argument("--squash-start-end", action='store_true',
                        help="Set this if your sequences have special start/end "
                             "symbols and you want the output, paired sentences "
                             "to *only* retain these at the start of the "
                             "first sentence and the end of the last sentence. "
                             "Tldr: set this for BERT models.")
    args = parser.parse_args()

    if not os.path.isfile(args.db):
        print("Error: cannot find ", args.db)
        exit(-1)

    db = sql.connect(args.db)
    cur = db.cursor()
    cur.execute("PRAGMA FOREIGN_KEYS = ON")

    assert(args.n > 0)
    assert(args.m > 0)
    
    if args.use_comma is None:
        # create a new comma character
        cur.execute("SELECT MAX(tokid) + 1 FROM Alphabet")
        comma_id = cur.fetchone()[0]
        cur.execute("INSERT INTO Alphabet(tokid, tokval) VALUES (?, ?)",
                    (comma_id, '<COMMA>'))
    else:
        # lookup the given token
        cur.execute("SELECT tokid FROM Alphabet WHERE tokval = ?",
                    (args.use_comma,))
        comma_id = cur.fetchone()
        if comma_id is None:
            print("Error: token '", args.use_comma, "' does not exist.")
            exit(-1)
        else:
            comma_id = comma_id[0]

    add_sequences(
        db, comma_id,
        pgb.progressbar(
            itertools.chain(
                get_pairs_all_lbltypes(db, args.n, args.m),
                get_reflexive_sequences(db)
            )
        ),
        args.squash_start_end
    )

    db.commit()
    db.close()
