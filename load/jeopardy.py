"""This script loads the Jeopardy dataset. Available from
https://www.reddit.com/r/datasets/comments/1uyd0t/200000_jeopardy_questions_in_a_json_file/
"""


import os
import sys
import json
import bisect
import argparse as ap
import sqlite3 as sql
import progressbar as pgb
from transformers import BertTokenizer


class Alphabet:
    def __init__(self):
        self.vals = {}

    def at(self, x):
        if x in self.vals:
            return self.vals[x]
        else:
            self.vals[x] = len(self.vals)
            return len(self.vals) - 1

    def iterate(self):
        return self.vals.items()


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def labels(row):
    dt = row['air_date'].split('-')
    return {
        'category': row['category'],
        'value': row['value'],
        'round': row['round'],
        'show_number': row['show_number'],
        'year': str(dt[0]),
        'month': str(dt[0]) + '-' + str(dt[1]),
        'day': row['air_date']
    }


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("db", type=str,
                        help="Filename of the DB to create.")
    parser.add_argument("jeopardy", type=str,
                        help="Path to the Jeopardy dataset JSON file.")
    parser.add_argument("--limit", default=None, type=int,
                        help="Limit the total size of the dataset.")
    args = parser.parse_args()

    if os.path.isfile(args.db):
        print("Error: ", args.db, "exists. This script creates a new DB.")
        exit(-1)

    if not os.path.isfile(args.jeopardy):
        print("Error: ", args.jeopardy, "does not exist.")
        exit(-1)

    create_db_sql = os.path.join(os.path.dirname(sys.argv[0]), "../", "create_db.sql")
    if not os.path.isfile(create_db_sql):
        print("Error: cannot find the DB creation script `create_db.sql`.",
              "Expected at:", create_db_sql)
        exit(-1)

    db = sql.connect(args.db)
    cur = db.cursor()

    with open(create_db_sql, "r") as f:
        cur.executescript(f.read())

    cur.execute("PRAGMA FOREIGN_KEYS = ON")
    cur.execute("BEGIN TRANSACTION")
    cur.execute("PRAGMA DEFER_FOREIGN_KEYS = ON")

    seqid = 0
    lbl_types = {}
    lbl_type_ids = Alphabet()
    with open(args.jeopardy, "r") as f:
        js = json.load(f)

        if args.limit is not None:
            js = js[:min(args.limit, len(js))]

        # 80/10/10 data split
        data_split = [int(0.8 * len(js)), int(0.05 * len(js))]
        data_split[1] += data_split[0]
        data_split.append(len(js) - data_split[1])

        for i, row in pgb.progressbar(enumerate(js)):
            # compute which partition this sequence falls into
            split = bisect.bisect_left(data_split, i)

            # save an entry for the sequence itself:
            cur.execute("""
                INSERT INTO Sequences(seqid, seqpart)
                VALUES (?, ?)
            """, (seqid, split))

            # compute and save labels
            lbls = labels(row)
            for lbl_type, lbl in lbls.items():
                if lbl_type not in lbl_types:
                    lbl_types[lbl_type] = Alphabet()
                cur.execute("""
                    INSERT INTO Labels(seqid, lbltype, lbl)
                    VALUES (?, ?, ?)
                """, (seqid, lbl_type_ids.at(lbl_type), lbl_types[lbl_type].at(lbl)))

            # compute and save the sequence values:
            for i, s in enumerate(tokenizer(row['question'])['input_ids']):
                cur.execute("""
                    INSERT INTO SequenceValues(seqid, svidx, tokid)
                    VALUES (?, ?, ?)
                """, (seqid, i, s))

            # move onto next sequence
            seqid += 1

    # now save all of the alphabets and label types
    cur.executemany("""
        INSERT INTO Alphabet(tokval, tokid) VALUES (?, ?)
    """, dict(tokenizer.vocab).items())
    cur.executemany("""
        INSERT INTO LabelTypes(lbltype_name, lbltype) VALUES (?, ?)
    """, lbl_type_ids.iterate())
    for lbltype_name, lbl_alphabet in lbl_types.items():
        lbltype = lbl_type_ids.at(lbltype_name)
        cur.executemany("""
            INSERT INTO LabelDictionary(lbltype, lblval, lbl)
            VALUES (?, ?, ?)
        """, map(lambda t: (lbltype,) + t, lbl_alphabet.iterate()))

    cur.execute("COMMIT")
    db.commit()
    db.close()
