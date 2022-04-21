"""
Classify by computing an MST.
Requires that the table `PairwiseDistances` be filled out.
"""


import os
import argparse as ap
import sqlite3 as sql
import numpy as np
import networkx as nx
import pandas as pd
import progressbar as pgb


def get_graphs(db):
    cur = db.cursor()
    # note: we have the guarantee that the sequence partitions of seqid_1
    # and seqid_2 are the same
    cur.execute(
        "SELECT DISTINCT compid, ncd_formula, dist_aggregator, seqpart "
        "FROM PairwiseDistances JOIN Sequences ON seqid = seqid_1 "
        "WHERE seqpart > 0"
        )
    cur2 = db.cursor()
    while True:
        tuple = cur.fetchone()
        if tuple is None:
            return
        compid, ncd_formula, dist_aggregator, seqpart = tuple

        cur2.execute(
            "SELECT seqid_1, seqid_2, seqid_train, dist "
            "FROM PairwiseDistances "
            "JOIN Sequences ON seqid_1 = Sequences.seqid "
            "WHERE compid = ? AND ncd_formula = ? AND dist_aggregator = ? AND seqpart = ?",
            (compid, ncd_formula, dist_aggregator, seqpart)
        )

        G = nx.from_pandas_edgelist(
            pd.DataFrame(cur2.fetchall(),
                         columns=["seqid_1", "seqid_2", "seqid_train", "dist"]),
            "seqid_1", "seqid_2", ["dist", "seqid_train"]
        )

        yield compid, ncd_formula, dist_aggregator, seqpart, G


def classify(lbltypes, seqid_train_lbls, datum):
    compid, ncd_formula, dist_aggregator, _, G = datum

    # get MST
    G_mst = nx.minimum_spanning_tree(G, weight="dist")

    for seqid in G_mst.nodes():
        for lbltype in lbltypes:
            # vote on neighbouring edge labels on that node in the MST
            lbls = {}
            for _, _, seqid_train in G_mst.edges(seqid, data="seqid_train"):
                lbl = seqid_train_lbls[(lbltype, seqid_train)]
                if lbl not in lbls:
                    lbls[lbl] = 0
                lbls[lbl] += 1

            # get highest vote
            max_count = -1
            max_lbl = None
            for lbl, n in lbls.items():
                if n > max_count:
                    max_count = n
                    max_lbl = lbl

            yield (lbltype, compid, ncd_formula, 'MST-' + dist_aggregator, seqid, max_lbl)


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("db", type=str,
                        help="Filename of the DB to load.")
    args = parser.parse_args()

    if not os.path.isfile(args.db):
        print("Error: cannot find ", args.db)
        exit(-1)

    db = sql.connect(args.db)

    cur = db.cursor()

    # get list of all label types
    cur.execute("SELECT lbltype FROM LabelTypes")
    lbltypes = list(map(lambda t: t[0], cur.fetchall()))

    # get a dict which maps each training sequence and label type
    # to its corresponding label
    cur.execute("""
    WITH RelevantTrain AS (
        SELECT DISTINCT seqid_train AS seqid
        FROM TrainingPairings
    )
    SELECT lbltype, seqid, lbl
    FROM RelevantTrain NATURAL JOIN Labels
    """)
    seqid_train_lbls = dict(
        ((lbltype, seqid), lbl) for lbltype, seqid, lbl in cur.fetchall()
    )

    cur.executemany(
        """INSERT INTO Predictions(
            lbltype, compid, ncd_formula, predictor, seqid, lbl
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        # generator expression: first get each graph (with hyperparameters) in
        # the DB, then classify each node within that graph and yield it.
        pgb.progressbar((row for datum in get_graphs(db)
                             for row in classify(lbltypes, seqid_train_lbls, datum)))
    )

    db.commit()
    db.close()
