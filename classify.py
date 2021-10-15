"""
Once all of the similarity distances (NCDs) have been computed,
this script uses different classification methods to classify the labels.
"""


import os
import itertools
import argparse as ap
import sqlite3 as sql
import progressbar as pgb


def nearest_neighbour(db):
    cur = db.cursor()
    cur.execute("""
    WITH TrainingPairings AS (
        SELECT seqid_left AS seqid_train, seqid_right AS seqid_other,
        lbltype, lbl, ncd_formula, ncd_value, compid
        FROM SequencePairings JOIN Sequences ON seqid_left = Sequences.seqid
        WHERE Sequences.seqpart = 0
        JOIN NCDValues ON seqid_out = NCDValues.seqid
        JOIN Labels ON Sequences.seqid = Labels.seqid
        UNION ALL
        SELECT seqid_right AS seqid_train, seqid_left AS seqid_other,
        lbltype, lbl, ncd_formula, ncd_value, compid
        FROM SequencePairings JOIN Sequences ON seqid_right = Sequences.seqid
        WHERE Sequences.seqpart = 0
        JOIN NCDValues ON seqid_out = NCDValues.seqid
        JOIN Labels ON Sequences.seqid = Labels.seqid
    ),
    Labellings AS (
        SELECT lbltype, ncd_formula, compid, seqid_other AS seqid, lbl, MIN(ncd_value)
        FROM TrainingPairings
        GROUP BY seqid_other, lbltype, ncd_formula, compid
    )
    SELECT DISTINCT lbltype, ncd_formula, compid, seqid, lbl FROM Labellings
    """)
    return cur.fetchall()


CLASSIFICATION_METHODS = {
    '1-NN': nearest_neighbour
}


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("db", type=str,
                        help="Filename of the DB to load.")
    args = parser.parse_args()

    if not os.path.isfile(args.db):
        print("Error: cannot find ", args.db)
        exit(-1)

    db = sql.connect(args.db)

    def apply(name_runner):
        name, runner = name_runner
        for row in runner(db):
            yield (name,) + row

    cur = db.cursor()
    cur.executemany("""
    INSERT INTO Predictions(predictor, lbltype, ncd_formula, compid, seqid, lbl)
    VALUES (?, ?, ?, ?, ?, ?)
    """,
    pgb.progressbar(itertools.chain(map(apply, CLASSIFICATION_METHODS.items())))
    )

    db.commit()
    db.close()
