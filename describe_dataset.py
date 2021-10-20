"""Simple script for printing some dataset statistics.
"""


import os
import argparse as ap
import sqlite3 as sql


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("db", type=str,
                        help="Filename of the DB to describe.")
    args = parser.parse_args()

    if not os.path.isfile(args.db):
        print("Error: ", args.db, "does not exist.")
        exit(-1)

    db = sql.connect(args.db)
    cur = db.cursor()
    cur.execute("PRAGMA FOREIGN_KEYS = ON")

    cur.execute("SELECT COUNT(*) FROM Sequences")
    print("Total dataset size is", cur.fetchone()[0])

    cur.execute("SELECT seqpart, COUNT(seqid) FROM Sequences GROUP BY seqpart")
    print("Dataset split is", cur.fetchall())

    cur.execute("SELECT COUNT(*) FROM Alphabet")
    print("Alphabet size is", cur.fetchone()[0])

    cur.execute("""
        WITH SeqLens AS (
            SELECT MAX(svidx) + 1 AS seqlen, seqid FROM SequenceValues GROUP BY seqid
        )
        SELECT AVG(seqlen) FROM SeqLens NATURAL JOIN Sequences GROUP BY seqpart
    """)
    print("Average sentence length in each partition is", cur.fetchall())

    cur.execute("SELECT COUNT(*) FROM LabelTypes")
    print("There are", cur.fetchone()[0], "label types")

    cur.execute("SELECT COUNT(lbl) FROM LabelDictionary GROUP BY lbltype")
    print("The label types' number of labels is respectively:", cur.fetchall())

    db.commit()
    db.close()
