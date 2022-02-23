"""
A script for clearing anything added to the DB
from the following scripts:
- compute_ncd.py
- classify.py
- results.py
This script is handy if you'd like to recompute all results from
scratch without having to re-run the compressors.
TODO: instead could just make the above scripts more resilient to
existing data.
"""


import os
import sqlite3 as sql
import argparse as ap


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
    cur.execute("PRAGMA FOREIGN_KEYS = ON")

    cur.execute("DELETE FROM NCDValues")
    cur.execute("DELETE FROM Predictions")
    cur.execute("DELETE FROM LabelScores")
    cur.execute("DELETE FROM ResultAccuracies")

    db.commit()
    db.close()
