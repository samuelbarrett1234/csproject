"""This script computes the pairwise estimates
of distance between validation / test sequences
by using various measures going via training
sequences. This could be implemented by a single
SQL query but it is too slow.
"""


import os
import sqlite3 as sql
import argparse as ap
import progressbar as pgb


def generate_pair_dists(db):
    cur = db.cursor()
    # get all pairs of sequences which are non-train and in the same partition,
    # alongside each lbltype/compid/ncd_formula combination
    cur.execute("""
    WITH TrainingPairingsAugmented AS (
        SELECT DISTINCT lbltype, compid, ncd_formula, seqid_other, seqpart
        FROM TrainingPairings JOIN Sequences ON seqid_other = seqid
        WHERE seqpart > 0
    )
    SELECT DISTINCT lbltype, compid, ncd_formula, A.seqid_other AS seqid_1, B.seqid_other AS seqid_2
    FROM TrainingPairingsAugmented AS A JOIN TrainingPairingsAugmented AS B
    USING (lbltype, compid, ncd_formula, seqpart)
    """)
    # for a given pair of sequences and lbltype/compid/ncd_formula combination,
    # varying over all training sequences, find the one which minimises A+B ncd values
    q = """
    SELECT MIN(A.ncd_value + B.ncd_value) AS dist
    FROM TrainingPairings AS A
    JOIN TrainingPairings AS B USING (lbltype, compid, ncd_formula, seqid_train)
    WHERE A.lbltype = ? AND A.compid = ? AND A.ncd_formula = ? AND A.seqid_other = ?
    AND B.lbltype = ? AND B.compid = ? AND B.ncd_formula = ? AND B.seqid_other = ?
    """
    while True:
        tuple = cur.fetchone()
        if tuple is None:
            return
        (lbltype, compid, ncd_formula, seqid_1, seqid_2) = tuple

        cur1 = db.cursor()
        cur1.execute(q, (lbltype, compid, ncd_formula, seqid_1, lbltype, compid, ncd_formula, seqid_2))
        dist = cur.fetchone()[0]
        yield (lbltype, compid, ncd_formula, 'mp', seqid_1, seqid_2, dist)


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
