"""
After classification, this script collects together the performances
of all of the model variations, and outputs them to a CSV.
"""


import os
import csv
import itertools
import argparse as ap
import sqlite3 as sql
import progressbar as pgb


def compute_accuracy(db):
    cur = db.cursor()
    cur.execute("""
    WITH PredVsTrue AS (
        SELECT lbltype, compid, predictor, ncd_formula, seqpart,
        CASE WHEN Predictions.lbl = Labels.lbl THEN 1.0 ELSE 0.0 END AS correct
        FROM Predictions NATURAL JOIN Sequences
        JOIN Labels ON Labels.seqid = Predictions.seqid AND Labels.lbltype = Predictions.lbltype
    ),
    Train AS (
        SELECT lbltype, compid, predictor, ncd_formula, AVG(correct) AS train_acc
        FROM PredVsTrue WHERE seqpart = 0 GROUP BY lbltype, compid, predictor, ncd_formula
    ),
    Val AS (
        SELECT lbltype, compid, predictor, ncd_formula, AVG(correct) AS val_acc
        FROM PredVsTrue WHERE seqpart = 1 GROUP BY lbltype, compid, predictor, ncd_formula
    ),
    Test AS (
        SELECT lbltype, compid, predictor, ncd_formula, AVG(correct) AS test_acc
        FROM PredVsTrue WHERE seqpart = 2 GROUP BY lbltype, compid, predictor, ncd_formula
    ),
    SELECT lbltype, compname, comprepeat, compiteration, predictor, ncd_formula,
    train_acc, val_acc, test_acc
    FROM Train NATURAL JOIN Val NATURAL JOIN Test NATURAL JOIN Compressors
    """)


METRICS = {
    'accuracy': compute_accuracy
}


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("db", type=str,
                        help="Filename of the DB to load.")
    parser.add_argument("folder", type=str,
                        help="Folder to write the output results to.")
    args = parser.parse_args()

    if not os.path.isfile(args.db):
        print("Error: cannot find ", args.db)
        exit(-1)

    if not os.path.isdir(args.folder):
        print("Error: output directory does not exist: ", args.db)
        exit(-1)

    db = sql.connect(args.db)
    cur = db.cursor()
    cur.execute("PRAGMA FOREIGN_KEYS = ON")

    for name, func in METRICS.items():
        out_fname = os.path.join(args.folder, name + ".txt")

        if os.path.isfile(out_fname):
            print("WARNING:", out_fname, "already exists. Skipping.")
            continue
        print("Writing", name, "to", out_fname, "...")

        with open(out_fname, "w") as f:
            writer = csv.writer(f)
            writer.writerow(('Label_Type', 'Compression_Algorithm',
                             'Compression_Repeat', 'Compression_Iteration',
                             'Predictor', 'NCD_Formula',
                             'TrainScore', 'ValScore', 'TestScore'))
            writer.writerows(pgb.progressbar(func(db)))

    db.close()
