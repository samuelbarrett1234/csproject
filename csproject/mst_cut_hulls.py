"""
For each distance matrix, compute MST, then for all possible edges
to remove (which induces a cut), calcuate the precision/recall of
the result, then compute the hull of the corresponding set of points
which acts as an evaluation measure which is invariant to class
imbalances, indicating how well-separable the data is.
"""


import os
import argparse as ap
import sqlite3 as sql
import numpy as np
import networkx as nx
import pandas as pd
import progressbar as pgb
from scipy.spatial import ConvexHull


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


def get_bipartitions(G):
    G_mst = nx.minimum_spanning_tree(G, weight="dist")
    for (u, v) in G_mst.edges():
        G_removed = G_mst.copy()
        G_removed.remove_edge(u, v)
        yield tuple(nx.connected_components(G_removed))


def calculate_stats(class_freq, precisions, recalls):
    precisions, recalls = np.array(precisions), np.array(recalls)
    extra_datas = np.array([(1.0, 0.0), (class_freq, 1.0), (0.0, 0.0)])
    data = np.concatenate((np.stack((precisions, recalls)).T,
                           extra_datas),
                          axis=0)
    return (
        ConvexHull(data).volume,  # true value
        0.5 * (1.0 + class_freq)  # baseline value
    )


def get_rows(db):
    cur = db.cursor()

    # get list of all label types
    cur.execute("SELECT lbltype FROM LabelTypes")
    lbltypes = list(map(lambda t: t[0], cur.fetchall()))

    cur.execute("SELECT lbltype, lbltype_name FROM LabelTypes")
    lbltype_names = dict(cur.fetchall())

    cur.execute("SELECT compid, compname FROM Compressors")
    compnames = dict(cur.fetchall())

    # get a dict which maps each training sequence and label type
    # to its corresponding label
    cur.execute("""
    WITH RelevantOther AS (
        SELECT DISTINCT seqid_other AS seqid
        FROM TrainingPairings
    )
    SELECT lbltype, seqid, lbl
    FROM RelevantOther NATURAL JOIN Labels
    """)
    seqid_lbls = dict(
        ((lbltype, seqid), lbl) for lbltype, seqid, lbl in cur.fetchall()
    )

    for compid, ncd_formula, dist_aggregator, seqpart, G in get_graphs(db):
        # cache the bipartitions between lbltypes
        biparts = list(get_bipartitions(G))

        for lbltype in lbltypes:
            # compute class frequency:
            class_freq = 0
            for seqid in G.nodes():
                class_freq += seqid_lbls[(lbltype, seqid)]
            class_freq /= len(G.nodes())

            # compute precision/recall for each bipartition
            precisions, recalls = [], []
            for C, _ in biparts:
                # calculate number of true/false negatives/positives
                # along with the class frequency
                tp, tn, fp, fn = 0, 0, 0, 0
                class_freq = 0
                for seqid in G.nodes():
                    if seqid_lbls[(lbltype, seqid)] == 1:
                        class_freq += 1
                        if seqid in C:
                            tp += 1  # or fn for other way around
                        else:
                            fn += 1  # or tp for other way around
                    else:  # if seqid_lbls[(lbltype, seqid)] == 0
                        if seqid not in C:
                            tn += 1  # or fp for other way around
                        else:
                            fp += 1  # or tn for other way around

                # any classification which finds no true positives
                # is completely irrelevant to this analysis
                # (and, don't worry, there will always exist a
                # cut which makes this > 0, provided at least one
                # instance of that label exists)

                if tp > 0:
                    precisions.append(tp / (tp + fp))
                    recalls.append(tp / (tp + fn))

                if fn > 0:  # version 2 (other way around w.r.t `C`)
                    precisions.append(fn / (fn + tn))
                    recalls.append(fn / (tp + fn))

            true_score, baseline_score = calculate_stats(class_freq, precisions, recalls)

            yield (
                lbltype_names[lbltype], compnames[compid],
                ncd_formula, dist_aggregator, seqpart,
                true_score, baseline_score
            )


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("db", type=str,
                        help="Filename of the DB to load.")
    parser.add_argument("out_csv", type=str,
                        help="Output CSV filename.")
    args = parser.parse_args()

    if not os.path.isfile(args.db):
        print("Error: cannot find ", args.db)
        exit(-1)

    db = sql.connect(args.db)

    df = pd.DataFrame(
        pgb.progressbar(get_rows(db)),
        columns=['lbltype_name', 'compname', 'ncd_formula',
                 'dist_aggregator', 'seqpart',
                 'hull_area', 'baseline_hull_area']).set_index(
                     ['lbltype_name', 'compname', 'ncd_formula',
                      'dist_aggregator', 'seqpart'])
    df.to_csv(args.out_csv)

    # db.commit()  # not changed anything, but I've kept this here to remind you to uncomment if this changes!
    db.close()
