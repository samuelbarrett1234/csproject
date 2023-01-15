"""
An extended version of the `classify` script, used for
training classifiers on the validation set (where labels
are available). (Note: cannot train on the training set
because those aren't suitably paired-up. See `pair_up.py`.)
"""


import os
import argparse as ap
import sqlite3 as sql
import numpy as np
from sklearn import svm
import progressbar as pgb


def get_data(db, lbltype, compid, ncd_formula, seqpart):
    cur = db.cursor()
    cur.execute("""
    SELECT Labels.lbl, seqid_train, seqid_other, ncd_value
    FROM TrainingPairings JOIN Labels ON
    TrainingPairings.lbltype = Labels.lbltype AND Labels.seqid = seqid_other
    JOIN Sequences ON Labels.seqid = Sequences.seqid
    WHERE TrainingPairings.lbltype = ? AND compid = ? AND ncd_formula = ? AND seqpart = ?
    """, (lbltype, compid, ncd_formula, seqpart))

    # turn SQL row data into Python objects
    train_seqs = set()
    labellings = {}
    dists = {}
    for lbl, train, other, dist in cur.fetchall():
        train_seqs.add(train)
        labellings[other] = lbl
        dists[(other, train)] = dist

    assert(len(dists) == len(labellings) * len(train_seqs))  # cross product

    # construct numpy arrays
    X = np.zeros((len(labellings), len(train_seqs)), dtype=np.float32)
    y = np.zeros((len(labellings),), dtype=np.int32)
    ids = []
    for i, other_and_lbl in enumerate(labellings.items()):
        other, lbl = other_and_lbl
        y[i] = lbl
        ids.append(other)
        for j, train in enumerate(train_seqs):
            X[i, j] = dists[(other, train)]

    return X, y, ids


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("db", type=str,
                        help="Filename of the DB to load.")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--svm", type=float, default=None,
                     help="SVM classifier. Value passed is C, "
                          "the regularization param (see sklearn).")
    args = parser.parse_args()

    if not os.path.isfile(args.db):
        print("Error: cannot find ", args.db)
        exit(-1)

    db = sql.connect(args.db)

    if args.svm is not None:
        PREDICTOR_NAME = 'svm-' + str(args.svm)
        PREDICTOR = lambda: svm.LinearSVC(C=args.svm, max_iter=100000)

    cur_in = db.cursor()
    cur_out = db.cursor()

    # enumerate all of the different contexts in which we need to train a model
    cur_in.execute("SELECT DISTINCT lbltype, compid, ncd_formula FROM TrainingPairings")
    for lbltype, compid, ncd_formula in pgb.progressbar(cur_in.fetchall()):
        # get the relevant data
        X_train, y_train, train_ids = get_data(db, lbltype, compid, ncd_formula, 1)
        X_test, y_test, test_ids = get_data(db, lbltype, compid, ncd_formula, 2)
        # construct and train model
        model = PREDICTOR()
        model.fit(X_train, y_train)
        # evaluate
        y_train_preds = model.predict(X_train)
        y_test_preds = model.predict(X_test)
        # insert results to DB
        for id, pred in zip(train_ids + test_ids, np.concatenate((y_train_preds, y_test_preds))):
            cur_out.execute(
                "INSERT INTO Predictions(predictor, lbltype, ncd_formula, compid, seqid, lbl) VALUES (?, ?, ?, ?, ?, ?)",
                (PREDICTOR_NAME, lbltype, ncd_formula, compid, id, int(pred)))

    db.commit()
    db.close()
