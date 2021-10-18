"""
Constructs pairings of all datasets (which do not have them already)
to allow training on pairs of sequences.
"""


import os
import itertools
import argparse as ap
import sqlite3 as sql
import progressbar2 as pgb


def add_sequences(db, comma_id, iter):
    cur = db.cursor()
    cur.execute("SELECT MAX(seqid) + 1 FROM Sentences")
    next_seq_id = cur.fetchone()[0]
    for seq_left, seq_right, seqpart_out in iter:
        # create a new sequence entry
        cur.execute("""
            INSERT INTO Sequences(seqid, seqpart, seq_is_pair)
            VALUES (?, ?, 1)
        """, (next_seq_id, seqpart_out))
        # copy leftmost sequence
        cur.execute("""
            INSERT INTO SequenceValues(seqid, svidx, tokid)
            SELECT ?, svidx, tokid FROM SequenceValues WHERE seqid = ?
        """, (next_seq_id, seq_left))
        # get leftmost sequence length
        cur.execute("""
            SELECT MAX(svidx) + 1 FROM SequenceVallues WHERE seqid = ?
        """, (seq_left,))
        seq_left_sz = cur.fetchone()[0]
        # insert comma
        cur.execute("""
            INSERT INTO SequenceValues(seqid, svidx, tokid)
            VALUES (?, ?, ?)
        """, (next_seq_id, seq_left_sz, comma_id))
        # copy rightmost sequence
        cur.execute("""
            INSERT INTO SequenceValues(seqid, svidx, tokid)
            SELECT ?, svidx + ?, tokid FROM SequenceValues WHERE seqid = ?
        """, (next_seq_id, seq_left_sz + 1, seq_right))
        # advance output sequence ID
        next_seq_id += 1


def get_reflexive_sequences(db):  # get all sequences matched in a pair with themselves
    cur = db.cursor()
    cur.execute("SELECT seqid, seqpart FROM Sequences")
    return map(lambda t: (t[0], t[0], t[1]), cur.fetchall())


def get_evaluation_sequences(db):  # get all nontrain sequences paired with a train sequence
    cur = db.cursor()
    # make sure to select ORDERED PAIRS of such sequences
    # (hence the "UNION ALL".)
    cur.execute("""
        SELECT A.seqid, B.seqid, A.seqpart
        FROM Sequences AS A JOIN Sequences AS B
        ON A.seqpart != B.seqpart
        WHERE A.seqpart != 0

        UNION ALL

        SELECT A.seqid, B.seqid, B.seqpart
        FROM Sequences AS A JOIN Sequences AS B
        ON A.seqpart != B.seqpart
        WHERE B.seqpart != 0
    """)
    return cur.fetchall()


def get_all_train_sequences(db):  # get all train sequences paired with all other train sequences
    cur = db.cursor()
    cur.execute("""
        SELECT A.seqid, B.seqid, 0
        FROM Sequences AS A JOIN Sequences AS B
        ON A.seqid != B.seqid
        WHERE A.seqpart = 0 AND B.seqpart = 0
    """)
    return cur.fetchall()


def get_cyclic_shift_train_pairing(db):  # match the nth train sequence with the (n+1)st
    cur = db.cursor()
    cur.execute("""
        SELECT seqid FROM Sequences
        WHERE seqpart = 0
        ORDER BY seqid ASC
    """)
    last_seqid = None
    for seqid in cur.fetchall():
        seqid = seqid[0]
        if last_seqid is not None:
            yield(last_seqid, seqid, 0)
        last_seqid = seqid


TRAIN_SET_POLICY = {
    'all': get_all_train_sequences,
    'cyclic_shift': get_cyclic_shift_train_pairing
}


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("db", type=str,
                        help="Filename of the DB to load.")
    parser.add_argument("policy", type=str,
                        help="The policy to use to construct the training dataset. "
                             "Must be one of: " + ", ".join(TRAIN_SET_POLICY.keys()))
    args = parser.parse_args()

    if not os.path.isfile(args.db):
        print("Error: cannot find ", args.db)
        exit(-1)

    if args.policy not in TRAIN_SET_POLICY:
        print("Invalid training set construction policy '",
              args.policy, "'. Must be one of:",
              ", ".join(TRAIN_SET_POLICY.keys()))
        exit(-1)

    db = sql.connect(args.db)
    cur = db.cursor()

    # create a new comma character
    cur.execute("SELECT MAX(tokid) + 1 FROM Alphabet")
    comma_id = cur.fetchone()[0]
    cur.execute("INSERT INTO Alphabet(tokid, tokval) VALULES (?, ?)",
                (comma_id, '<COMMA>'))

    add_sequences(
        db, comma_id,
        pgb.progressbar(
            itertools.chain(
                [
                    get_reflexive_sequences(db),
                    get_evaluation_sequences(db),
                    TRAIN_SET_POLICY[args.policy](db)
                ]
            )
        )
    )

    db.commit()
    db.close()
