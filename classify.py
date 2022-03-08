"""
Once all of the similarity distances (NCDs) have been computed,
this script uses different classification methods to classify the labels.
"""


import os
import math
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


class AvgDistClassificationAggregator:
    """An SQLite aggregation function for classifying
    points based on their average distance.
    """
    def __init__(self, log=True):
        self.log = log
        self.data = {}  # mapping from labels to NCD distances

    def step(self, lbl, dist):
        if self.log:
            dist = math.log(max(dist, 1.0e-9))

        if lbl not in self.data:
            self.data[lbl] = []

        self.data[lbl].append(dist)

    def finalize(self):
        try:
            self.data = dict([
                (lbl, sum(vs) / len(vs)) for lbl, vs in self.data.items()
            ])
            # return lowest avg
            min_v = min(self.data.values())
            for lbl, v in self.data.items():
                if v == min_v:
                    return lbl
        finally:
            self.data = {}  # reset


class QuantileClassificationAggregator:
    """An SQLite aggregation function for classifying
    points based on a quantile distance between the
    distance distributions. Lower quantile values mean
    comparisons closer to the minimum.
    """
    def __init__(self, quantile):
        quantile = float(quantile)
        assert(quantile >= 0.0 and quantile < 1.0)
        self.quantile = quantile
        self.data = {}  # mapping from labels to NCD distances

    def step(self, lbl, dist):
        if lbl not in self.data:
            self.data[lbl] = []

        self.data[lbl].append(dist)

    def finalize(self):
        try:
            self.data = dict([
                (lbl, sorted(vs)[int(self.quantile * len(vs))]) for lbl, vs in self.data.items()
            ])
            # return class associated with the lowest quantile
            min_v = min(self.data.values())
            for lbl, v in self.data.items():
                if v == min_v:
                    return lbl
        finally:
            self.data = {}  # reset


def execute(db, predictor, predictor_name):
    cur = db.cursor()
    db.create_aggregate("CLASSIFY", 2,
                        predictor)

    try:
        cur.execute("""
        WITH Labellings AS (
            SELECT seqid_other AS seqid, lbltype, ncd_formula, compid,
            CLASSIFY(lbl, ncd_value) AS lbl
            FROM TrainingPairings
            GROUP BY seqid_other, lbltype, ncd_formula, compid
        )
        SELECT ?, lbltype, ncd_formula, compid, seqid, lbl
        FROM Labellings
        """, (predictor_name,))
        return cur.fetchall()

    finally:
        # to clean up, delete the aggregation function
        db.create_aggregate("CLASSIFY", 2, None)


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("db", type=str,
                        help="Filename of the DB to load.")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--knn", type=int, default=None,
                     help="KNN classifier, with given integer K.")
    grp.add_argument("--quantile", type=float, default=None,
                     help="Quantile classifier, with given quantile (smaller means closer to min).")
    grp.add_argument("--avg", action='store_true',
                     help="Choose label based on the class with smallest average distance.")
    grp.add_argument("--avg-log", action='store_true',
                     help="Choose label based on the class with smallest average log-distance.")
    args = parser.parse_args()

    if not os.path.isfile(args.db):
        print("Error: cannot find ", args.db)
        exit(-1)

    db = sql.connect(args.db)

    # construct the predictor
    if args.knn is not None:
        PREDICTOR_NAME = str(args.knn) + "-NN"
        PREDICTOR = lambda: KNNClassificationAggregator(args.knn)
    elif args.quantile:
        PREDICTOR_NAME = str(args.quantile) + "-quantile"
        PREDICTOR = lambda: QuantileClassificationAggregator(args.quantile)
    elif args.avg:
        PREDICTOR_NAME = 'AVG'
        PREDICTOR = lambda: AvgDistClassificationAggregator(log=False)
    elif args.avg_log:
        PREDICTOR_NAME = 'AVG-LOG'
        PREDICTOR = lambda: AvgDistClassificationAggregator(log=True)

    cur = db.cursor()
    cur.execute("PRAGMA FOREIGN_KEYS = ON")
    cur.executemany("""
    INSERT INTO Predictions(predictor, lbltype, ncd_formula, compid, seqid, lbl)
    VALUES (?, ?, ?, ?, ?, ?)
    """,
    pgb.progressbar(execute(db, PREDICTOR, PREDICTOR_NAME))
    )

    db.commit()
    db.close()
