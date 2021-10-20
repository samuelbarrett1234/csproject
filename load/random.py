"""This script randomly generates data, to be used as a baseline.
It is expected that this data is incompressible.
"""


import os
import sys
import time
import random
import argparse as ap
import sqlite3 as sql


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("db", type=str,
                        help="Filename of the DB to create.")
    parser.add_argument("--alphabet-sz", type=int, default=32,
                        help="The number of characters in the alphabet.")
    parser.add_argument("--seed", type=int, default=int(time.time()),
                        help="Random seed. Defaults to current time.")
    parser.add_argument("--n-seqs", type=int, default=1000,
                        help="The number of sequences in the dataset.")
    parser.add_argument("--n-lbl-types", type=int, default=2,
                        help="The number of label types.")
    parser.add_argument("--avg-seq-len", type=float, default=50.0,
                        help="The average sequence length.")
    args = parser.parse_args()

    random.seed(args.seed)

    if os.path.isfile(args.db):
        print("Error: ", args.db, "exists. This script creates a new DB.")
        exit(-1)

    create_db_sql = os.path.join(os.path.dirname(sys.argv[0]), "create_db.sql")
    if not os.path.isfile(create_db_sql):
        print("Error: cannot find the DB creation script `create_db.sql`.",
              "Expected at:", create_db_sql)
        exit(-1)

    db = sql.connect(args.db)
    cur = db.cursor()

    with open(create_db_sql, "r") as f:
        cur.execute(f.read())

    seq_ids = list(range(args.n_seqs))
    seq_parts = [0] * int(args.n_seqs * 0.8) + [1] * int(args.n_seqs)
    seq_parts = seq_parts + [2] * (len(seq_ids) - len(seq_parts))
    cur.executemany("INSERT INTO Sequences(seqid, seqpart) VALUES (?, ?)",
                    zip(seq_ids, seq_parts))

    cur.executemany("INSERT INTO Alphabet(tokid) VALUES (?)",
                    map(lambda t: (t,), range(args.alphabet_sz)))

    def gen_seqs():
        for seq_id in seq_ids:
            n = random.randint(args.avg_seq_len // 2, 3 * (args.avg_seq_len // 2))
            for i in range(n):
                t = random.randint(0, args.alphabet_sz - 1)
                yield (seq_id, i, t)

    cur.executemany("INSERT INTO SequenceValues(seqid, svidx, tokid) VALUES (?, ?, ?)",
                    gen_seqs())

    cur.executemany("INSERT INTO LabelTypes(lbltype) VALUES (?)",
                    map(lambda i: (i,), range(args.n_lbl_types)))

    cur.executemany("INSERT INTO LabelDictionary(lbltype, lbl) VALUES (?, ?)",
                    zip(range(args.n_lbl_types), range(2, args.n_lbl_types + 2)))

    def gen_lbls():
        for seq_id in seq_ids:
            for lbltype in range(args.n_lbl_types):
                lbl = random.randint(2, lbltype + 1)
                yield seq_id, lbltype, lbl

    cur.executemany("INSERT INTO Labels(seqid, lbltype, lbl) VALUES (?, ?, ?)",
                    gen_lbls())

    db.commit()
    db.close()
