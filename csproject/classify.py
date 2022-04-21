"""
Once all of the similarity distances (NCDs) have been computed,
this script uses different classification methods to classify the labels.
"""


import os
import copy
import math
import heapq
import argparse as ap
import sqlite3 as sql
import progressbar as pgb


class KNNClassificationAggregator:
    """A structure for performing K-nearest-neighbour
    classification.
    """
    def __init__(self, k):
        self.k = k
        self.data = []  # heap, mapping distances to training sequences

    def step(self, seqid_train, dist):
        heapq.heappush(self.data, (dist, seqid_train))

    def predict(self, lbl_map):
        # extract top K elements, put them
        # into a dict which maps each label
        # to its frequency
        top = {}
        data_copy = copy.copy(self.data)
        for _ in range(self.k):
            # if there are fewer than k data points
            if len(data_copy) == 0:
                break

            seqid_train = heapq.heappop(data_copy)[1]
            lbl = lbl_map[seqid_train]
            if lbl not in top:
                top[lbl] = 1
            else:
                top[lbl] += 1

        assert(len(top) > 0)

        # return vote
        max_freq = max(top.values())
        for lbl, freq in top.items():
            if freq == max_freq:
                return lbl


class AvgDistClassificationAggregator:
    """A structure for classifying
    points based on their average distance.
    """
    def __init__(self, log=True):
        self.log = log
        self.data = {}  # mapping from seqid_train to NCD distance

    def step(self, seqid_train, dist):
        if self.log:
            dist = math.log(max(dist, 1.0e-9))

        self.data[seqid_train] = dist

    def predict(self, lbl_map):
        # convert to mapping from labels to distances
        lbl_data = {}
        for seqid_train, dist in self.data.items():
            lbl = lbl_map[seqid_train]
            if lbl not in lbl_data:
                lbl_data[lbl] = [dist]
            else:
                lbl_data[lbl].append(dist)
        # convert to mapping from labels to average distances
        lbl_data = dict([
            (lbl, sum(vs) / len(vs)) for lbl, vs in lbl_data.items()
        ])
        # return lowest avg
        min_v = min(lbl_data.values())
        for lbl, v in lbl_data.items():
            if v == min_v:
                return lbl


class QuantileClassificationAggregator:
    """A structure for classifying
    points based on a quantile distance between the
    distance distributions. Lower quantile values mean
    comparisons closer to the minimum.
    """
    def __init__(self, quantile):
        quantile = float(quantile)
        assert(quantile >= 0.0 and quantile < 1.0)
        self.quantile = quantile
        self.data = {}  # mapping from seqid_train to NCD distance

    def step(self, seqid_train, dist):
        self.data[seqid_train] = dist

    def predict(self, lbl_map):
        # convert to mapping from labels to distances
        lbl_data = {}
        for seqid_train, dist in self.data.items():
            lbl = lbl_map[seqid_train]
            if lbl not in lbl_data:
                lbl_data[lbl] = [dist]
            else:
                lbl_data[lbl].append(dist)
        # convert to mapping from labels to the right quantile
        lbl_data = dict([
            (lbl, sorted(vs)[int(self.quantile * len(vs))]) for lbl, vs in lbl_data.items()
        ])
        # return class associated with the lowest quantile
        min_v = min(lbl_data.values())
        for lbl, v in lbl_data.items():
            if v == min_v:
                return lbl


class NCDDataGetter:
    def __init__(self, db):
        self.cur = db.cursor()
        # more reasons why I don't like SQLite
        self.cur.execute(
            "SELECT DISTINCT compid, ncd_formula, seqid_other "
            "FROM TrainingPairings"
        )
        self.L = 0
        while self.cur.fetchone() is not None:
            self.L += 1

    def __len__(self):
        return self.L

    def __iter__(self):
        self.cur.execute("""
        SELECT compid, ncd_formula, seqid_other, seqid_train, ncd_value
        FROM TrainingPairings
        ORDER BY compid, ncd_formula, seqid_other
        """)
        self.cur_row = self.cur.fetchone()
        return self

    def __next__(self):
        if self.cur_row is None:
            raise StopIteration()

        idx = self.cur_row[:3]
        data = []
        while self.cur_row is not None and self.cur_row[:3] == idx:
            data.append(self.cur_row[3:])
            self.cur_row = self.cur.fetchone()

        return (idx, data)


class Executor:
    def __init__(self, db, predictor, predictor_name):
        self.predictor = predictor
        self.pred_name = predictor_name
        self.ncd_getter = NCDDataGetter(db)
        self.cur = db.cursor()
        self.cur.execute("""
        WITH RelevantTrain AS (
            SELECT DISTINCT seqid_train AS seqid
            FROM TrainingPairings
        )
        SELECT lbltype, seqid, lbl
        FROM RelevantTrain NATURAL JOIN Labels
        """)
        self.lblmaps = {}
        for lbltype, seqid, lbl in self.cur.fetchall():
            if lbltype not in self.lblmaps:
                self.lblmaps[lbltype] = {}
            self.lblmaps[lbltype][seqid] = lbl
        self.cur.execute("SELECT lbltype FROM LabelTypes")
        self.lbltypes = list(map(lambda t: t[0], self.cur.fetchall()))

    def __len__(self):
        return len(self.lbltypes) * len(self.ncd_getter)

    def __iter__(self):
        self.ncd_iter = iter(self.ncd_getter)
        self.lbltype_index = len(self.lbltypes)
        self.idx = None
        return self

    def __next__(self):
        if self.lbltype_index == len(self.lbltypes):
            self.idx, data = next(self.ncd_iter)

            self.lbltype_index = 0
            self.pred = self.predictor()

            for seqid_train, dist in data:
                self.pred.step(seqid_train, dist)

        # get current label type and then advance
        lbltype = self.lbltypes[self.lbltype_index]
        self.lbltype_index += 1
        # now actually return the result
        return (self.pred_name, lbltype,) + self.idx + (self.pred.predict(self.lblmaps[lbltype]),)


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
    INSERT INTO Predictions(predictor, lbltype, compid, ncd_formula, seqid, lbl)
    VALUES (?, ?, ?, ?, ?, ?)
    """,
    pgb.progressbar(Executor(db, PREDICTOR, PREDICTOR_NAME))
    )

    db.commit()
    db.close()
