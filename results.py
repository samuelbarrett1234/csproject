"""
After classification, this script collects together the performances
of all of the model variations, and outputs them to a CSV.
"""


import os
import csv
import argparse as ap
import sqlite3 as sql
import scipy.sparse
import scipy.sparse.linalg
import scipy.sparse.csgraph
import progressbar as pgb


def compute_accuracy(db):
    cur = db.cursor()
    cur.execute("""
    SELECT lbltype_name, compname, comprepeat, predictor, ncd_formula,
    train_acc, val_acc, test_acc
    FROM ResultAccuracies NATURAL JOIN Compressors NATURAL JOIN LabelTypes
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
    WITH InputDimension AS (
        SELECT COUNT(*) AS n FROM Alphabet
    )
    SELECT compname, comprepeat, compd, n, train_sz, val_sz, test_sz
    FROM Compressors NATURAL JOIN ResultSizes NATURAL JOIN InputDimension
    """)
    return (
        ('Compression_Algorithm', 'Compression_Repeat', 'Out_Dim',
         'In_Dim', 'Train', 'Val', 'Test'),
        cur.fetchall()
    )


def compute_compression_ratios(db):
    cur = db.cursor()
    cur.execute("""
    WITH InputDimension AS (
        SELECT COUNT(*) AS n FROM Alphabet
    )
    SELECT compname, comprepeat, compd, n, train_rt, val_rt, test_rt
    FROM Compressors NATURAL JOIN ResultRatios NATURAL JOIN InputDimension
    """)
    return (
        ('Compression_Algorithm', 'Compression_Repeat', 'Out_Dim',
         'In_Dim', 'Train', 'Val', 'Test'),
        cur.fetchall()
    )


def compute_lbl_scores(db):
    results = []

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

    cur2.execute("SELECT DISTINCT compid, compname, ncd_formula "
                 "FROM NCDValues NATURAL JOIN Compressors")
    for compid, compname, ncd_formula in cur2.fetchall():
        # NOTE: it is important that this query symmetrizes the `ncd_value`
        # correctly
        cur.execute("""
            WITH MySequences AS (
                SELECT seqid_left, seqid_right,
                CASE WHEN seqid_left = seqid_right THEN 0.0 ELSE ncd_value
                END AS ncd_value
                FROM NCDValues NATURAL JOIN Sequences
                JOIN SequencePairings ON Sequences.seqid = seqid_out
                WHERE compid = ? AND ncd_formula = ? AND seqpart = 1
            )
            SELECT A.seqid_left, A.seqid_right, 0.5 * (A.ncd_value + B.ncd_value)
            FROM MySequences AS A JOIN MySequences AS B
            ON A.seqid_left = B.seqid_right AND A.seqid_right = B.seqid_left
        """, (compid, ncd_formula))
        D = scipy.sparse.dok_matrix((N, N))
        for left, right, dist in cur.fetchall():
            D[(left, right)] = max(dist, 0.0)  # clip to be >= 0
        # compute Laplacian
        D = scipy.sparse.csgraph.laplacian(D, normed=True)
        # now score each label type:
        for lbltype, Y in lbl_mats.items():
            s = sum((Y.T @ D @ Y).diagonal()) / Y.shape[1]
            assert(s >= 0)
            # insert into DB but also to the list `results` which we will return
            cur.execute("""
                INSERT INTO LabelScores(compid, ncd_formula, lbltype, score)
                VALUES (?, ?, ?, ?)
            """, (compid, ncd_formula, lbltype, s))
            results.append(
                (compname, ncd_formula, lbltype_names[lbltype], s)
            )

    return (('Compression_Algorithm', 'NCD_Formula', 'Label_Type', 'Score'), results)


METRICS = {
    'accuracy': compute_accuracy,
    'sizes': compute_compression_sizes,
    'ratios': compute_compression_ratios,
    'lbl_scores': compute_lbl_scores
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

    db.commit()
    db.close()
