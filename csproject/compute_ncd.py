"""
Compute all of the similarity distances (NCDs).
"""


import os
import itertools
import argparse as ap
import sqlite3 as sql
import progressbar as pgb


# NOTE: all of the below formulae, where applicable, use the approximation
# C(x|y) = C(x, y) - C(y)
# since in practice it is hard to construct a general mechanism for C(x|y)
# (see https://arxiv.org/pdf/cs/0111054.pdf; this is acceptable)
NCD_FORMULAE = {
    # https://arxiv.org/pdf/1006.3520.pdf def 3.1
    'inf-dist': lambda xy, x, y: xy - min([x, y]),
    # https://arxiv.org/pdf/cs/0111054.pdf def V.1
    'norm-inf-dist-1': lambda xy, x, y: 2.0 - (x + y) / xy,
    # https://arxiv.org/pdf/cs/0312044.pdf def 3.1
    'norm-inf-dist-2': lambda xy, x, y: (xy - min([x, y])) / max([x, y]),
    # my own idea, but almost surely tried before?
    'mutual-inf-esque': lambda xy, x, y: x + y - xy
}


def apply_ncd(data):
    # unpacking
    comp_data, ncd_formula_data = data
    xy, x, y, seqid, compid = comp_data
    ncd_formula, ncd_function = ncd_formula_data

    return seqid, compid, ncd_formula, ncd_function(xy, x, y)


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("db", type=str,
                        help="Filename of the DB to load.")
    args = parser.parse_args()

    if not os.path.isfile(args.db):
        print("Error: cannot find ", args.db)
        exit(-1)

    db = sql.connect(args.db)
    cur1 = db.cursor()
    cur1.execute("PRAGMA FOREIGN_KEYS = ON")

    cur1.execute("""
    SELECT (XY.compsz + YX.compsz) / 2.0 AS xy_compsz, X.compsz AS x_compsz, Y.compsz AS y_compsz,
    SP1.seqid_out AS seqid, XY.compid
    FROM SequencePairings AS SP1 JOIN SequencePairings AS SP2
    ON SP1.seqid_left = SP2.seqid_right AND SP1.seqid_right = SP2.seqid_left
    JOIN CompressionSizes AS XY ON SP1.seqid_out = XY.seqid
    JOIN CompressionSizes AS YX ON SP2.seqid_out = YX.seqid AND XY.compid = YX.compid
    JOIN CompressionSizes AS X ON XY.compid = X.compid AND SP1.seqid_left = X.seqid
    JOIN CompressionSizes AS Y ON XY.compid = Y.compid AND SP1.seqid_right = Y.seqid
    """)

    cur2 = db.cursor()
    cur2.executemany(
        "INSERT INTO NCDValues(seqid, compid, ncd_formula, ncd_value) VALUES (?, ?, ?, ?)",
        pgb.progressbar(map(
            apply_ncd,
            itertools.product(
                cur1.fetchall(),
                NCD_FORMULAE.items())))
    )

    cur2.execute("DELETE FROM TrainingPairings")
    cur2.execute("""
    INSERT INTO TrainingPairings(
        compid, ncd_formula, seqid_train, seqid_other, ncd_value
    )
    SELECT compid, ncd_formula, seqid_left AS seqid_train, seqid_right AS seqid_other, ncd_value
    FROM SequencePairings JOIN Sequences ON seqid_left = Sequences.seqid
    JOIN NCDValues ON seqid_out = NCDValues.seqid
    WHERE Sequences.seqpart = 0
    UNION
    SELECT compid, ncd_formula, seqid_right AS seqid_train, seqid_left AS seqid_other, ncd_value
    FROM SequencePairings JOIN Sequences ON seqid_right = Sequences.seqid
    JOIN NCDValues ON seqid_out = NCDValues.seqid
    WHERE Sequences.seqpart = 0
    """)

    db.commit()
    db.close()
