"""For binary labels, how well does each NCD measure
perform, when performance is considered by inducing
a 'better than average' cut?
Requires that the table `PairwiseDistances` be filled out.
"""


from contextlib import redirect_stdout
import os
import argparse as ap
import sqlite3 as sql
import numpy as np
import pandas as pd
import networkx as nx
import progressbar as pgb


def get_graph(db, compid, ncd_formula, seqpart, dist_agg):
    cur = db.cursor()
    cur.execute(
        "SELECT seqid_1, seqid_2, dist FROM PairwiseDistances "
        "JOIN Sequences ON seqid = seqid_1 "
        "WHERE compid = ? AND ncd_formula = ? AND seqpart = ? "
        "AND dist_aggregator = ?",
        (compid, ncd_formula, seqpart, dist_agg)
    )
    return nx.from_pandas_edgelist(
        pd.DataFrame(cur.fetchall(),
                     columns=["seqid_1", "seqid_2", "dist"]),
        "seqid_1", "seqid_2", ["dist"]
    )


def get_lbl_vector(db, lbltype, seqpart):
    cur = db.cursor()
    cur.execute(
        "SELECT lbl FROM Labels NATURAL JOIN Sequences "
        "WHERE lbltype = ? AND seqpart = ? ORDER BY seqid",
        (lbltype, seqpart))
    y = np.array(list(map(lambda t: t[0], cur.fetchall())), dtype=np.int32)

    assert(np.all(y <= 1))  # binary labels only!

    return y


def get_cut_and_rands(db, G, lbltype, seqpart, n_rand):
    y_true = get_lbl_vector(db, lbltype, seqpart)

    # crucial that these have the same order:
    sorting_permutation = np.argsort(np.array(G.nodes(), dtype=np.int32))
    y_true = y_true[np.argsort(sorting_permutation)]

    # now ready to compute the cut
    true_score = nx.cut_size(G, np.argwhere(y_true == 1)[:, 0], weight='weight')

    null_scores = []
    for _ in range(n_rand):
        y_randoms = np.copy(y_true)
        np.random.shuffle(y_randoms)
        null_scores.append(nx.cut_size(
            G, np.argwhere(y_randoms == 1)[:, 0], weight='weight'))

    return true_score, np.sort(null_scores)


def get_loop(db):
    cur = db.cursor()

    # compute and cache the label types up front
    cur.execute(
        "SELECT lbltype, lbltype_name FROM LabelTypes "
        "NATURAL JOIN LabelDictionary "
        "GROUP BY lbltype HAVING MAX(lbl) = 1")
    lbltypes = cur.fetchall()

    cur.execute("SELECT DISTINCT compid, ncd_formula, dist_aggregator FROM PairwiseDistances")
    while True:
        tuple = cur.fetchone()
        if tuple is None:
            return
        compid, ncd_formula, dist_aggregator = tuple

        # validation graph
        G = get_graph(db, compid, ncd_formula, 1, dist_aggregator)
        for lbltype, lbltype_name in lbltypes:
            yield lbltype_name, lbltype, compid, ncd_formula, dist_aggregator, G, 1

        # test graph
        G = get_graph(db, compid, ncd_formula, 2, dist_aggregator)
        for lbltype, lbltype_name in lbltypes:
            yield lbltype_name, lbltype, compid, ncd_formula, dist_aggregator, G, 2


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("db", type=str,
                        help="Filename of the DB to load.")
    parser.add_argument("num_alternatives", type=int,
                        help="The number of randomisations to apply (higher "
                             "means more memory and time, but more accurate "
                             "results.")
    args = parser.parse_args()

    if not os.path.isfile(args.db):
        print("Error: cannot find ", args.db)
        exit(-1)

    db = sql.connect(args.db)

    cur = db.cursor()
    cur.execute("DELETE FROM Cuts")

    lbltype_names_seen = set()
    for lbltype_name, lbltype, compid, ncd_formula, dist_aggregator, G, seqpart in pgb.progressbar(get_loop(db), redirect_stdout=True):
        if lbltype_name not in lbltype_names_seen:
            print(lbltype_name)
            lbltype_names_seen.add(lbltype_name)

        true_cut, null_cuts = get_cut_and_rands(
            db, G, lbltype, seqpart, args.num_alternatives)

        full = [(0, float(true_cut))] + [(i + 1, float(null_cuts[i])) for i in range(len(null_cuts))]

        cur.executemany(
            "INSERT INTO Cuts(lbltype, compid, ncd_formula, "
            "dist_aggregator, seqpart, n_rand, cut_value) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (lbltype, compid, ncd_formula, dist_aggregator, seqpart, i, c) for i, c in full
            ])

    db.commit()
    db.close()
