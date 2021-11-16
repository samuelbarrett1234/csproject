"""
Constructs pairings of all datasets (which do not have them already)
to allow training on pairs of sequences.
"""


import os
import itertools
import argparse as ap
import sqlite3 as sql
import progressbar as pgb


def add_sequences(db, comma_id, iter, squash_start_end):
    cur = db.cursor()
    cur.execute("SELECT MAX(seqid) + 1 FROM Sequences")
    next_seq_id = cur.fetchone()[0]
    for seq_left, seq_right, seqpart_out in iter:
        # do not error on producing repeats
        cur.execute("""
            SELECT 1 FROM SequencePairings
            WHERE seqid_left = ? AND seqid_right = ?
        """, (seq_left, seq_right))
        if cur.fetchone() is not None:
            continue

        # create a new sequence entry
        cur.execute("""
            INSERT INTO Sequences(seqid, seqpart, seq_is_pair)
            VALUES (?, ?, 1)
        """, (next_seq_id, seqpart_out))

        # mark it as paired
        cur.execute("""
            INSERT INTO SequencePairings(seqid_left, seqid_right, seqid_out)
            VALUES (?, ?, ?)
        """, (seq_left, seq_right, next_seq_id))

        # get leftmost sequence length
        cur.execute("""
            SELECT MAX(svidx) + 1 FROM SequenceValues WHERE seqid = ?
        """, (seq_left,))
        seq_left_sz = cur.fetchone()[0]

        if squash_start_end:
            # copy leftmost sequence
            # don't copy end token
            cur.execute("""
                INSERT INTO SequenceValues(seqid, svidx, tokid)
                SELECT ?, svidx, tokid FROM SequenceValues
                WHERE seqid = ? AND svidx + 1 < ?
            """, (next_seq_id, seq_left, seq_left_sz))

            # insert comma
            cur.execute("""
                INSERT INTO SequenceValues(seqid, svidx, tokid)
                VALUES (?, ?, ?)
            """, (next_seq_id, seq_left_sz - 1, comma_id))

            # copy rightmost sequence
            # don't copy first token
            cur.execute("""
                INSERT INTO SequenceValues(seqid, svidx, tokid)
                SELECT ?, svidx + ? - 1, tokid FROM SequenceValues
                WHERE seqid = ? AND svidx > 0
            """, (next_seq_id, seq_left_sz, seq_right))
        else:
            # copy leftmost sequence
            cur.execute("""
                INSERT INTO SequenceValues(seqid, svidx, tokid)
                SELECT ?, svidx, tokid FROM SequenceValues
                WHERE seqid = ?
            """, (next_seq_id, seq_left))

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
    cur.execute("SELECT seqid, seqpart FROM Sequences WHERE seq_is_pair = 0")
    return map(lambda t: (t[0], t[0], t[1]), cur.fetchall())


def get_all_evaluation_sequences(db):  # get all nontrain sequences paired with a train sequence
    cur = db.cursor()
    # make sure to select ORDERED PAIRS of such sequences
    # (hence the "UNION ALL".)
    cur.execute("""
        SELECT A.seqid, B.seqid, B.seqpart
        FROM Sequences AS A JOIN Sequences AS B
        ON A.seqpart != B.seqpart
        WHERE A.seqpart = 0 AND A.seq_is_pair = 0
        AND B.seq_is_pair = 0
    """)
    for A_seqid, B_seqid, B_seqpart in cur.fetchall():
        # yield both ways round
        yield (A_seqid, B_seqid, B_seqpart)
        yield (B_seqid, A_seqid, B_seqpart)


def lbl_subsample(db, seqpart, lbltype, subsampler):
    cur1 = db.cursor()
    cur2 = db.cursor()
    cur1.execute("SELECT lbl FROM LabelDictionary NATURAL JOIN LabelTypes "
                 "WHERE lbltype_name = ?", (lbltype,))
    for lbl in map(lambda t: t[0], cur1.fetchall()):
        cur2.execute("SELECT seqid FROM Sequences NATURAL JOIN Labels "
                     "NATURAL JOIN LabelTypes WHERE lbltype_name = ? "
                     "AND lbl = ? AND seqpart = ?",
                     (lbltype, lbl, seqpart))
        for seqid in subsampler(map(lambda t: t[0], cur2.fetchall())):
            yield seqid


def subsample_top_k(k):
    def f(iter):
        i = 0
        for item in iter:
            yield item
            i += 1
            if i == k:
                break

    return f


def get_evaluation_sequences_with_k(lbltype, k):  # pick `k` train samples for each label in `lbltype`
    def f(db):
        cur = db.cursor()
        cur.execute("CREATE TEMP TABLE ChosenSeqs1(seqid INTEGER PRIMARY KEY)")
        cur.executemany(
            "INSERT INTO ChosenSeqs1(seqid) VALUES (?)",
            map(lambda seqid: (seqid,), lbl_subsample(
                db, 0, lbltype, subsample_top_k(k)
            ))
        )
        # make sure to select ORDERED PAIRS of such sequences
        # (hence the "UNION ALL".)
        cur.execute("""
            SELECT A.seqid, B.seqid, B.seqpart
            FROM ChosenSeqs1 AS A JOIN Sequences AS B
            WHERE B.seqpart > 0 AND B.seq_is_pair = 0
        """)
        for A_seqid, B_seqid, B_seqpart in cur.fetchall():
            # yield both ways round
            yield (A_seqid, B_seqid, B_seqpart)
            yield (B_seqid, A_seqid, B_seqpart)

    return f


def get_all_train_sequences(db):  # get all train sequences paired with all other train sequences
    cur = db.cursor()
    cur.execute("""
        SELECT A.seqid, B.seqid, 0
        FROM Sequences AS A JOIN Sequences AS B
        ON A.seqid != B.seqid
        WHERE A.seqpart = 0 AND B.seqpart = 0
        AND A.seq_is_pair = 0 AND B.seq_is_pair = 0
    """)
    return cur.fetchall()


def get_cyclic_shift_train_pairing(db):  # match the nth train sequence with the (n+1)st
    cur = db.cursor()
    cur.execute("""
        SELECT seqid FROM Sequences
        WHERE seqpart = 0 AND seq_is_pair = 0
        ORDER BY seqid ASC
    """)
    last_seqid = None
    for i, seqid in enumerate(cur.fetchall()):
        seqid = seqid[0]
        if last_seqid is not None:
            if i % 2 == 0:  # sometimes yield either <x,y> or <y,x>
                yield(last_seqid, seqid, 0)
            else:
                yield(seqid, last_seqid, 0)
        last_seqid = seqid


def get_random_train_pairing(db):
    cur = db.cursor()
    cur.execute("""
        SELECT seqid FROM Sequences
        WHERE seqpart = 0 AND seq_is_pair = 0
        ORDER BY seqid ASC
    """)
    cur2 = db.cursor()
    cur.execute("""
        SELECT seqid FROM Sequences
        WHERE seqpart = 0 AND seq_is_pair = 0
        ORDER BY RANDOM()
    """)
    for i, seqids in enumerate(zip(cur.fetchall(), cur2.fetchall())):
        seqid1, seqid2 = seqids
        if i % 2 == 0:  # sometimes yield either <x,y> or <y,x>
            yield(seqid1, seqid2, 0)
        else:
            yield(seqid2, seqid1, 0)


def get_val_seq_lbl_matrix(lbltype, k):  # pick `k` val samples for each label in `lbltype`
    def f(db):
        SEQPART = 1
        cur = db.cursor()
        cur.execute("CREATE TEMP TABLE ChosenSeqs2(seqid INTEGER PRIMARY KEY)")
        cur.executemany(
            "INSERT INTO ChosenSeqs2(seqid) VALUES (?)",
            map(lambda seqid: (seqid,), lbl_subsample(
                db, SEQPART, lbltype, subsample_top_k(k)
            ))
        )
        # make sure to select ORDERED PAIRS of such sequences
        # (hence the "UNION ALL".)
        cur.execute("""
            SELECT A.seqid, B.seqid
            FROM ChosenSeqs2 AS A JOIN Sequences AS B
            WHERE B.seqpart = ? AND B.seq_is_pair = 0
        """, (SEQPART,))
        for A_seqid, B_seqid in cur.fetchall():
            # yield both ways round
            yield (A_seqid, B_seqid, SEQPART)
            yield (B_seqid, A_seqid, SEQPART)

    return f


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("db", type=str,
                        help="Filename of the DB to load.")
    parser.add_argument("lbltype", type=str,
                        help="The label type to use to generate pair data.")
    parser.add_argument("k", type=int,
                        help="Higher means more data, hence more accurate classifiers.")
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

    cur.execute("SELECT 1 FROM LabelTypes WHERE lbltype_name = ?",
                (args.lbltype,))
    if cur.fetchone() is None:
        print("Error: label type '", args.lbltype, "' does not exist.")
        exit(-1)

    assert(args.k > 0)

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
                get_reflexive_sequences(db),
                get_cyclic_shift_train_pairing(db),
                get_random_train_pairing(db),
                get_val_seq_lbl_matrix(args.lbltype, args.k)(db),
                get_evaluation_sequences_with_k(args.lbltype, args.k)(db),
            )
        ),
        args.squash_start_end
    )

    db.commit()
    db.close()
