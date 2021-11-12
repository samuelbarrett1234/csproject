"""
After classification, this script collects together the performances
of all of the model variations, and outputs them to a CSV.
"""


import os
import csv
import argparse as ap
import sqlite3 as sql
import progressbar as pgb


def compute_accuracy(db):
    cur = db.cursor()
    cur.execute("""
    WITH PredVsTrue AS (
        SELECT Predictions.lbltype, compid, predictor, ncd_formula, seqpart,
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
    )
    SELECT lbltype_name, compname, comprepeat, predictor, ncd_formula,
    train_acc, val_acc, test_acc
    FROM Train NATURAL JOIN Val NATURAL JOIN Test NATURAL JOIN Compressors NATURAL JOIN LabelTypes
    """)
    return (
        ('Label_Type', 'Compression_Algorithm',
         'Compression_Repeat',
         'Predictor', 'NCD_Formula',
         'TrainScore', 'ValScore', 'TestScore'),
        cur.fetchall()
    )


def compute_compression_sizes(db):
    cur = db.cursor()
    cur.execute("""
    WITH Train AS (
        SELECT compid, AVG(compsz) AS train_sz
        FROM CompressionSizes NATURAL JOIN Sequences
        WHERE seqpart = 0
        GROUP BY compid
    ),
    Val AS (
        SELECT compid, AVG(compsz) AS val_sz
        FROM CompressionSizes NATURAL JOIN Sequences
        WHERE seqpart = 1
        GROUP BY compid
    ),
    Test AS (
        SELECT compid, AVG(compsz) AS test_sz
        FROM CompressionSizes NATURAL JOIN Sequences
        WHERE seqpart = 2
        GROUP BY compid
    ),
    InputDimension AS (
        SELECT COUNT(*) AS n FROM Alphabet
    )
    SELECT compname, comprepeat, compd, n, train_sz, val_sz, test_sz
    FROM Compressors NATURAL JOIN Train NATURAL JOIN Val NATURAL JOIN Test
    NATURAL JOIN InputDimension
    """)
    return (
        ('Compression_Algorithm', 'Compression_Repeat', 'Out_Dim',
         'In_Dim', 'Train', 'Val', 'Test'),
        cur.fetchall()
    )


def compute_compression_ratios(db):
    cur = db.cursor()
    cur.execute("""
    WITH SequenceLengths AS (
        SELECT seqid, MAX(svidx) + 1 AS slen FROM SequenceValues
        GROUP BY seqid
    ),
    InputDimension AS (
        SELECT COUNT(*) AS n FROM Alphabet
    ),
    Train AS (
        SELECT compid, AVG(compsz / slen) AS train_rt
        FROM CompressionSizes NATURAL JOIN SequenceLengths
        NATURAL JOIN Sequences WHERE seqpart = 0
        GROUP BY compid
    ),
    Val AS (
        SELECT compid, AVG(compsz / slen) AS val_rt
        FROM CompressionSizes NATURAL JOIN SequenceLengths
        NATURAL JOIN Sequences WHERE seqpart = 0
        GROUP BY compid
    ),
    Test AS (
        SELECT compid, AVG(compsz / slen) AS test_rt
        FROM CompressionSizes NATURAL JOIN SequenceLengths
        NATURAL JOIN Sequences WHERE seqpart = 0
        GROUP BY compid
    )
    SELECT compname, comprepeat, compd, n, train_rt, val_rt, test_rt
    FROM Compressors NATURAL JOIN Train NATURAL JOIN Val NATURAL JOIN Test
    NATURAL JOIN InputDimension
    """)
    return (
        ('Compression_Algorithm', 'Compression_Repeat', 'Out_Dim',
         'In_Dim', 'Train', 'Val', 'Test'),
        cur.fetchall()
    )


METRICS = {
    'accuracy': compute_accuracy,
    'sizes': compute_compression_sizes,
    'ratios': compute_compression_ratios
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
            headers, data = func(db)
            writer.writerow(headers)
            writer.writerows(pgb.progressbar(data))

    db.close()
