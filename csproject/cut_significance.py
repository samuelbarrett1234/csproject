"""Run this after `compute_cuts.py`, to determine how significant each cut is.
"""


import os
import argparse as ap
import sqlite3 as sql
import pandas as pd
import numpy as np
import progressbar as pgb


def process_cut_values(true_cut, null_cuts):
    true_cut = np.array(true_cut)
    null_cuts = np.sort(np.array(null_cuts))
    avg = np.mean(null_cuts)
    std = np.std(null_cuts)

    # error case
    if std == 0:
        return None, None

    z_score = (true_cut - avg) / std
    pos = np.searchsorted(null_cuts, true_cut)
    cumulative_prob = pos / len(null_cuts)
    return z_score, 1.0 - cumulative_prob


def get_cut_values(db):
    cur = db.cursor()
    cur.execute("""
    SELECT DISTINCT lbltype_name, compname, lbltype, compid,
    ncd_formula, dist_aggregator, seqpart
    FROM Cuts NATURAL JOIN LabelTypes NATURAL JOIN Compressors
    """)
    cur2 = db.cursor()
    for lbltype_name, compname, lbltype, compid, ncd_formula, dist_aggregator, seqpart in pgb.progressbar(cur.fetchall()):
        cur2.execute("""
        SELECT cut_value FROM Cuts WHERE
        lbltype = ? AND compid = ? AND ncd_formula = ? AND
        dist_aggregator = ? AND seqpart = ? AND n_rand = 0
        """, (lbltype, compid, ncd_formula, dist_aggregator, seqpart))
        true_cut = cur2.fetchone()[0]

        cur2.execute("""
        SELECT cut_value FROM Cuts WHERE
        lbltype = ? AND compid = ? AND ncd_formula = ? AND
        dist_aggregator = ? AND seqpart = ? AND n_rand > 0
        """, (lbltype, compid, ncd_formula, dist_aggregator, seqpart))
        null_cuts = list(map(lambda t: t[0], cur2.fetchall()))

        yield (lbltype_name, compname, ncd_formula, dist_aggregator, seqpart, true_cut, null_cuts)


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

    rows = []
    for (lbltype, compid, ncd_formula, dist_aggregator,
         seqpart, true_cut, null_cuts) in get_cut_values(db):
        z_score, quantile = process_cut_values(true_cut, null_cuts)
        rows.append((lbltype, compid, ncd_formula, dist_aggregator,
                     seqpart, z_score, quantile))

    df = pd.DataFrame(
        rows,
        columns=['lbltype_name', 'compname', 'ncd_formula',
                 'dist_aggregator', 'seqpart', 'z_score', 'emp_prob']).set_index(
                     ['lbltype_name', 'compname', 'ncd_formula',
                      'dist_aggregator', 'seqpart'])
    df.to_csv(args.out_csv)

    # db.commit()  # not changed anything, but I've kept this here to remind you to uncomment if this changes!
    db.close()
