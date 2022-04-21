"""Run this after `compute_cuts.py`, to determine how significant each cut is.
"""


import os
import argparse as ap
import sqlite3 as sql
import pandas as pd


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("db", type=str,
                        help="Filename of the DB to load.")
    parser.add_argument("out_csv", type=str,
                        help="Output CSV file.")
    args = parser.parse_args()

    if not os.path.isfile(args.db):
        print("Error: cannot find ", args.db)
        exit(-1)

    db = sql.connect(args.db)

    cur = db.cursor()
    cur.execute("""
        WITH
        TrueCuts AS (
            SELECT lbltype, compid, ncd_formula, dist_aggregator, seqpart, cut_value AS true_cut
            FROM Cuts WHERE n_rand = 0
        ),
        NullCuts AS (
            SELECT lbltype, compid, ncd_formula, dist_aggregator, seqpart,
            AVG(cut_value) AS avg_null_cut, AVG(cut_value * cut_value) AS avg_sq_null_cut
            FROM Cuts WHERE n_rand > 0
            GROUP BY lbltype, compid, ncd_formula, dist_aggregator, seqpart
        )
        SELECT lbltype_name, compname, ncd_formula, dist_aggregator, seqpart,
        true_cut, avg_null_cut, avg_sq_null_cut
        FROM TrueCuts NATURAL JOIN NullCuts NATURAL JOIN LabelTypes NATURAL JOIN Compressors
    """)

    rows = []
    for (lbltype_name, compname, ncd_formula, dist_aggregator, seqpart,
         true_cut, avg_null_cut, avg_sq_null_cut) in cur.fetchall():
        std = (avg_sq_null_cut - avg_null_cut ** 2.0) ** 0.5
        if std > 0.0:
            z_score = (true_cut - avg_null_cut) / std
            rows.append((lbltype_name, compname, ncd_formula, dist_aggregator, seqpart, z_score))

    df = pd.DataFrame(rows, columns=['lbltype_name', 'compname', 'ncd_formula', 'dist_aggregator', 'seqpart', 'z_score'])
    df.to_csv(args.out_csv)

    # db.commit()  # not changed anything, but I've kept this here to remind you to uncomment if this changes!
    db.close()
