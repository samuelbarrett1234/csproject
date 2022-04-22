"""This script computes the accuracies of each model on the validation and testing
datasets. This is not done in SQL because SQLite doesn't implement sufficiently
many join algorithms!
"""


import os
import argparse as ap
import sqlite3 as sql
import progressbar as pgb


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

    cur.execute(
    """CREATE TEMP TABLE PredVsTrueOnSeqparts AS
    SELECT lbltype, compid, predictor, ncd_formula, seqpart,
    SUM(true_positive) AS true_positive, SUM(true_negative) AS true_negative,
    SUM(false_positive) AS false_positive, SUM(false_negative) AS false_negative
    FROM PredictionEvaluations NATURAL JOIN Sequences
    WHERE seqpart > 0
    GROUP BY lbltype, compid, predictor, ncd_formula, seqpart
    """)
    cur.execute(
    """CREATE INDEX pvt_index
    ON PredVsTrueOnSeqparts(
        lbltype, compid, predictor, ncd_formula, seqpart)
    """
    )
    cur.execute("SELECT COUNT(*) FROM PredVsTrueOnSeqparts")
    L = cur.fetchone()[0] // 2

    q_select = """
    SELECT true_positive, true_negative,
    false_positive, false_negative
    FROM PredVsTrueOnSeqparts
    WHERE lbltype = ? AND compid = ? AND predictor = ?
    AND ncd_formula = ? AND seqpart = ?
    """
    q_insert = """
    INSERT INTO Results(
        lbltype, compid, predictor, ncd_formula,
        val_true_positive, val_true_negative, val_false_positive, val_false_negative,
        test_true_positive, test_true_negative, test_false_positive, test_false_negative
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    cur2 = db.cursor()
    cur.execute("""
    SELECT DISTINCT lbltype, compid, predictor, ncd_formula
    FROM PredVsTrueOnSeqparts
    EXCEPT
    SELECT lbltype, compid, predictor, ncd_formula
    FROM Results
    """)
    with pgb.ProgressBar(max_value=L) as pbar:
        while True:
            tuple = cur.fetchone()
            pbar.update(1)
            if tuple is None:
                break

            lbltype, compid, predictor, ncd_formula = tuple

            cur2.execute(q_select, (lbltype, compid, predictor, ncd_formula, 1))
            val_stats = cur2.fetchone()
            cur2.execute(q_select, (lbltype, compid, predictor, ncd_formula, 2))
            test_stats = cur2.fetchone()

            cur2.execute(q_insert,
                (lbltype, compid, predictor, ncd_formula,) + val_stats + test_stats)

    db.commit()
    db.close()
