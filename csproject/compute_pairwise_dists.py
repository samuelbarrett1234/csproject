"""This script computes the pairwise estimates
of distance between validation / test sequences
by using various measures going via training
sequences. This could be implemented by a single
SQL query but it is too slow.
"""


import os
import math
import sqlite3 as sql
import argparse as ap
import progressbar as pgb


def generate_pair_dists(db):
    cur = db.cursor()
    # note: we know that the outputs of this query are unique as it
    # is a superset of a key
    cur.execute("""
    SELECT lbltype, compid, ncd_formula, seqid_other, seqid_train, ncd_value, seqpart
    FROM TrainingPairings JOIN Sequences ON seqid_other = seqid
    WHERE seqpart > 0
    ORDER BY lbltype, compid, ncd_formula, seqid_other, seqid_train
    """)

    lbltype, compid, ncd_formula, tuple = None, None, None, None
    rows = []
    while True:
        # firstly, collect all relevant rows
        if tuple is not None:
            rows = [tuple[3:]]
            lbltype, compid, ncd_formula = tuple[:3]

        tuple = cur.fetchone()

        if lbltype is None:  # only triggers on first iteration
            lbltype, compid, ncd_formula = tuple[:3]

        while (lbltype, compid, ncd_formula) == tuple[:3]:
            rows.append(tuple[3:])
            tuple = cur.fetchone()
            if tuple is None:
                break

        # now do a kind of double iteration
        i, i_0, j = 0, 0, 0
        # forward j to the point where its seqid_other is different
        # to that of i but has the same seqpart
        while rows[j][0] == rows[i][0] or rows[i][3] != rows[j][3]:
            j += 1

        dist = math.inf
        while i < len(rows):
            assert (j < len(rows))
            assert (rows[i][1] == rows[j][1])
            assert (rows[i][3] == rows[j][3])

            # get shortest distance
            if rows[i][2] + rows[j][2] < dist:
                dist = rows[i][2] + rows[j][2]

            # if end of current seqid_other, need to yield
            if rows[i + 1][0] != rows[i][0]:
                yield (lbltype, compid, ncd_formula, 'mp', rows[i][0], rows[j][0], dist)
                yield (lbltype, compid, ncd_formula, 'mp', rows[j][0], rows[i][0], dist)
                dist = math.inf
                j += 1
                i = i_0
                # forward j to the point where it has the same seqpart as i
                while j < len(rows) and rows[i][3] != rows[j][3]:
                    j += 1
            else:
                i += 1
                j += 1

            while j == len(rows) and i < len(rows):
                # forward i to the point where its seqid_other is different
                while i + 1 < len(rows) and rows[i + 1][0] == rows[i][0]:
                    i += 1
                if i < len(rows):
                    i += 1

                # reset i_0 and j
                i_0 = i
                j = i

                # forward j to the point where its seqid_other is different
                # to that of i but has the same seqpart
                while j < len(rows) and (rows[j][0] == rows[i][0] or rows[i][3] != rows[j][3]):
                    j += 1


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("db", type=str,
                        help="Filename of the DB to load.")
    args = parser.parse_args()

    if not os.path.isfile(args.db):
        print("Error: cannot find ", args.db)
        exit(-1)

    db = sql.connect(args.db)
    cur = db.cursor()
    cur.execute("PRAGMA FOREIGN_KEYS = ON")

    cur.executemany("""
    INSERT INTO PairwiseDistances(
        lbltype, compid, ncd_formula, dist_aggregator,
        seqid_1, seqid_2, dist)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, pgb.progressbar(generate_pair_dists(db)))

    db.commit()
    db.close()
