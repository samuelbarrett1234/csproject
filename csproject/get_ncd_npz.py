"""A simple script for converting the TrainingPairings data
into NPZ arrays, forgetting about the identities of the sequences.
Produces a large collection of files. Prints to stdout all of the
files created.
Guarantee: seqids (train and val/test respectively) will be sorted
ascending. This should make you able to reconstruct which positions
in the output arrays correspond to which sequences, and guarantee
a consistent order across files.
"""


import os
import argparse as ap
import sqlite3 as sql
import numpy as np


def get_dist_matrices(db, seqid_trains, seqid_vals, seqid_tests, compid, ncd_formula):
    D_val = np.zeros((len(seqid_vals), len(seqid_trains)))
    D_test = np.zeros((len(seqid_tests), len(seqid_trains)))

    cur = db.cursor()
    cur.execute("""
    SELECT seqid_other, seqid_train, ncd_value
    FROM TrainingPairings
    WHERE compid = ? AND ncd_formula = ?
    """, (compid, ncd_formula))
    for i, j, d in cur.fetchall():
        if i in seqid_vals:
            D_val[seqid_vals[i], seqid_trains[j]] = d
        elif i in seqid_tests:
            D_test[seqid_tests[i], seqid_trains[j]] = d
        else:  # else is a train node
            assert(i in seqid_trains)

    mean, std = (
        np.mean(D_val, axis=0, keepdims=True),
        np.std(D_val, axis=0, keepdims=True)
    )

    # avoid division by 0
    std = np.where(std > 0.0, std, 1.0)

    return (D_val - mean) / std, (D_test - mean) / std


def get_labels(db, seqid_others, seqpart, lbltype):
    y = np.zeros(len(seqid_others), dtype=np.int32)

    cur = db.cursor()

    cur.execute("""
    SELECT seqid, lbl
    FROM Labels NATURAL JOIN Sequences
    WHERE seqpart = ? AND lbltype = ?
    """, (seqpart, lbltype))
    for i, l in cur.fetchall():
        y[seqid_others[i]] = l

    return y


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("db", type=str,
                        help="Filename of the DB to load.")
    parser.add_argument("out_folder", type=str,
                        help="Folder to dump all of the output files.")
    args = parser.parse_args()

    if not os.path.isfile(args.db):
        print("Error: cannot find DB", args.db)
        exit(-1)

    if not os.path.isdir(args.out_folder):
        print("Error: cannot find output folder", args.out_folder)
        exit(-1)

    db = sql.connect(args.db)

    cur = db.cursor()

    # get a mapping of all training sequences to indices, in increasing order
    cur.execute("SELECT seqid FROM Sequences WHERE seqpart = 0 AND seq_is_pair = 0 ORDER BY seqid ASC")
    seqid_trains = dict([(j[0], i) for i, j in enumerate(cur.fetchall())])

    # get a mapping of all val sequences to indices, in increasing order
    cur.execute("SELECT seqid FROM Sequences WHERE seqpart = 1 AND seq_is_pair = 0 ORDER BY seqid ASC")
    seqid_vals = dict([(j[0], i) for i, j in enumerate(cur.fetchall())])

    # get a mapping of all test sequences to indices, in increasing order
    cur.execute("SELECT seqid FROM Sequences WHERE seqpart = 2 AND seq_is_pair = 0 ORDER BY seqid ASC")
    seqid_tests = dict([(j[0], i) for i, j in enumerate(cur.fetchall())])

    # loop through all of the output types
    cur.execute("SELECT DISTINCT compid, ncd_formula FROM TrainingPairings")
    cns = cur.fetchall()
    cur.execute("SELECT lbltype FROM LabelTypes")
    lbltypes = [t[0] for t in cur.fetchall()]

    for compid, ncd_formula in cns:
        D_val, D_test = get_dist_matrices(
            db, seqid_trains, seqid_vals, seqid_tests,
            compid, ncd_formula)

        for lbltype in lbltypes:
            val_fname = f"lbltype-{lbltype}-comp-{compid}-ncd-{ncd_formula}-train.npz"  # not a typo
            test_fname = f"lbltype-{lbltype}-comp-{compid}-ncd-{ncd_formula}-test.npz"
            val_fname = os.path.join(args.out_folder, val_fname)
            test_fname = os.path.join(args.out_folder, test_fname)

            y_val = get_labels(db, seqid_vals, 1, lbltype)
            y_test = get_labels(db, seqid_tests, 2, lbltype)

            np.savez(val_fname, { 'x': D_val, 'y': y_val })
            print(val_fname)
            np.savez(test_fname, { 'x': D_test, 'y': y_test })
            print(test_fname)
