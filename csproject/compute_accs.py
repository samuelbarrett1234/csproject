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
    """CREATE TEMP VIEW PredVsTrueOnSeqparts AS
    WITH PredVsTrue AS (
        SELECT Predictions.lbltype, compid, predictor, ncd_formula, seqpart,
        CASE WHEN Predictions.lbl = Labels.lbl THEN 1.0 ELSE 0.0 END AS correct
        FROM Predictions NATURAL JOIN Sequences
        JOIN Labels ON Labels.seqid = Predictions.seqid AND Labels.lbltype = Predictions.lbltype
        WHERE seqpart > 0
    )
    SELECT lbltype, compid, predictor, ncd_formula, seqpart, AVG(correct) AS acc
    FROM PredVsTrue GROUP BY lbltype, compid, predictor, ncd_formula, seqpart
    """)

    cur2 = db.cursor()
    cur.execute("SELECT DISTINCT lbltype, compid, predictor, ncd_formula "
                "FROM PredVsTrueOnSeqparts AS A WHERE NOT EXISTS ("
                "SELECT 1 FROM ResultAccuracies AS B WHERE A.lbltype = B.lbltype AND "
                "A.compid = B.compid AND A.predictor = B.predictor AND "
                "A.ncd_formula = B.ncd_formula)")
    for lbltype, compid, predictor, ncd_formula in pgb.progressbar(cur.fetchall()):
        cur2.execute("SELECT acc FROM PredVsTrueOnSeqparts WHERE "
                     "lbltype = ? AND compid = ? AND predictor = ? AND "
                     "ncd_formula = ? AND seqpart = ?",
                     (lbltype, compid, predictor, ncd_formula, 1))
        val_acc = cur2.fetchone()

        if val_acc is None:
            print("Error: bad data. Perhaps you forgot to clear old data?")
            exit(-1)
        else:
            val_acc = val_acc[0]

        cur2.execute("SELECT acc FROM PredVsTrueOnSeqparts WHERE "
                     "lbltype = ? AND compid = ? AND predictor = ? AND "
                     "ncd_formula = ? AND seqpart = ?",
                     (lbltype, compid, predictor, ncd_formula, 2))
        test_acc = cur2.fetchone()

        if test_acc is None:
            print("Error: bad data. Perhaps you forgot to clear old data?")
            exit(-1)
        else:
            test_acc = test_acc[0]

        cur2.execute("INSERT INTO ResultAccuracies(lbltype, compid, predictor, ncd_formula, "
                     "val_acc, test_acc) VALUES (?, ?, ?, ?, ?, ?)",
                     (lbltype, compid, predictor, ncd_formula, val_acc, test_acc))

    db.commit()
    db.close()
