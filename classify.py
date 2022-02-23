"""
Once all of the similarity distances (NCDs) have been computed,
this script uses different classification methods to classify the labels.
"""


import os
import heapq
import argparse as ap
import sqlite3 as sql
import progressbar as pgb


class KNNClassificationAggregator:
    """An SQLite aggregation function for performing K-nearest-neighbour
    classification.
    """
    def __init__(self, k):
        self.k = k
        self.data = []  # heap, mapping distances to labels

    def step(self, lbl, dist):
        heapq.heappush(self.data, (dist, lbl))

    def finalize(self):
        # extract top K elements, put them
        # into a dict which maps each label
        # to its frequency
        top = {}
        for _ in range(self.k):
            # if there are fewer than k data points
            if len(self.data) == 0:
                break

            lbl = heapq.heappop(self.data)[1]
            if lbl not in top:
                top[lbl] = 1
            else:
                top[lbl] += 1

        assert(len(top) > 0)

        # reset heap
        self.data = []

        # return vote
        max_freq = max(top.values())
        for lbl, freq in top.items():
            if freq == max_freq:
                return lbl


def knn(db, k):
    cur = db.cursor()
    db.create_aggregate("KNN_CLASSIFY", 2,
                        lambda: KNNClassificationAggregator(k))

    try:
        cur.execute("""
        WITH TrainingPairings AS (
            SELECT seqid_left AS seqid_train, seqid_right AS seqid_other,
            lbltype, lbl, ncd_formula, ncd_value, compid
            FROM SequencePairings JOIN Sequences ON seqid_left = Sequences.seqid
            JOIN NCDValues ON seqid_out = NCDValues.seqid
            JOIN Labels ON Sequences.seqid = Labels.seqid
            WHERE Sequences.seqpart = 0
            UNION
            SELECT seqid_right AS seqid_train, seqid_left AS seqid_other,
            lbltype, lbl, ncd_formula, ncd_value, compid
            FROM SequencePairings JOIN Sequences ON seqid_right = Sequences.seqid
            JOIN NCDValues ON seqid_out = NCDValues.seqid
            JOIN Labels ON Sequences.seqid = Labels.seqid
            WHERE Sequences.seqpart = 0
        ),
        Labellings AS (
            SELECT seqid_other AS seqid, lbltype, ncd_formula, compid,
            KNN_CLASSIFY(lbl, ncd_value) AS lbl
            FROM TrainingPairings
            GROUP BY seqid_other, lbltype, ncd_formula, compid
        )
        SELECT lbltype, ncd_formula, compid, seqid, lbl
        FROM Labellings
        """)
        for row in cur.fetchall():
            yield row

    finally:
        # to clean up, delete the aggregation function
        db.create_aggregate("KNN_CLASSIFY", 2, None)


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("db", type=str,
                        help="Filename of the DB to load.")
    parser.add_argument("k", type=int,
                        help="The value of k for the KNN classifier.")
    args = parser.parse_args()

    if not os.path.isfile(args.db):
        print("Error: cannot find ", args.db)
        exit(-1)

    db = sql.connect(args.db)

    PREDICTOR_NAME = str(args.k) + "-NN"
    PREDICTOR = lambda db: knn(db, args.k)

    def apply():
        for row in PREDICTOR(db):
            yield (PREDICTOR_NAME,) + row

    cur = db.cursor()
    cur.execute("PRAGMA FOREIGN_KEYS = ON")
    cur.executemany("""
    INSERT INTO Predictions(predictor, lbltype, ncd_formula, compid, seqid, lbl)
    VALUES (?, ?, ?, ?, ?, ?)
    """,
    pgb.progressbar(apply())
    )

    db.commit()
    db.close()
