"""
Calculate how good each NCD measure is at separating each label type.
"""


import os
import argparse as ap
import sqlite3 as sql
import scipy.sparse
import scipy.sparse.linalg
import scipy.sparse.csgraph


def compute_lbl_scores(db, seqpart=1):
    cur = db.cursor()
    cur2 = db.cursor()

    cur.execute("DELETE FROM LabelScores")

    cur.execute("SELECT MAX(seqid) + 1 FROM Sequences WHERE seq_is_pair = 0")
    N = cur.fetchone()[0]

    # compute all one-hot label matrices
    lbl_mats = {}
    lbltype_names = {}
    cur2.execute("SELECT lbltype, lbltype_name, MAX(lbl) + 1 FROM Labels "
                 "NATURAL JOIN LabelTypes GROUP BY lbltype")
    for lbltype, lbltype_name, M in cur2.fetchall():
        cur.execute("SELECT seqid, lbl FROM Labels WHERE lbltype = ?",
                    (lbltype,))
        Y = scipy.sparse.dok_matrix((N, M))
        for seqid, lbl in cur.fetchall():
            Y[seqid, lbl] = 1.0
        lbl_mats[lbltype] = Y
        lbltype_names[lbltype] = lbltype_name

    cur2.execute("SELECT DISTINCT compid, ncd_formula FROM NCDValues")
    for compid, ncd_formula in cur2.fetchall():
        # NOTE: it is important that this query symmetrizes the `ncd_value`
        # correctly
        cur.execute("""
            WITH MySequences AS (
                SELECT seqid_left, seqid_right,
                CASE WHEN seqid_left = seqid_right THEN 0.0 ELSE ncd_value
                END AS ncd_value
                FROM NCDValues NATURAL JOIN Sequences
                JOIN SequencePairings ON Sequences.seqid = seqid_out
                WHERE compid = ? AND ncd_formula = ? AND seqpart = ?
            )
            SELECT A.seqid_left, A.seqid_right, 0.5 * (A.ncd_value + B.ncd_value)
            FROM MySequences AS A JOIN MySequences AS B
            ON A.seqid_left = B.seqid_right AND A.seqid_right = B.seqid_left
        """, (compid, ncd_formula, seqpart))
        D = scipy.sparse.dok_matrix((N, N))
        for left, right, dist in cur.fetchall():
            D[(left, right)] = max(dist, 0.0)  # clip to be >= 0
        # compute Laplacian
        D = scipy.sparse.csgraph.laplacian(D, normed=True)
        # now score each label type:
        for lbltype, Y in lbl_mats.items():
            s = sum((Y.T @ D @ Y).diagonal()) / Y.shape[1]
            assert(s >= 0)
            cur.execute("""
                INSERT INTO LabelScores(compid, ncd_formula, lbltype, score)
                VALUES (?, ?, ?, ?)
            """, (compid, ncd_formula, lbltype, s))


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

    compute_lbl_scores(db)

    db.commit()
    db.close()
