"""This script loads the Reuters Corpus from NLTK.
"""


import os
import sys
import argparse as ap
import sqlite3 as sql
import progressbar as pgb
from transformers import BertTokenizer
import nltk
from nltk.corpus import reuters


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def preprocess_doc(text_name):
    # replace first instance of newline with full stop, and remove all other newlines.
    return reuters.raw(text_name).replace("\n", ".", 1).replace("\n", "")


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("db", type=str,
                        help="Filename of the DB to create.")
    args = parser.parse_args()

    if os.path.isfile(args.db):
        print("Error: ", args.db, "exists. This script creates a new DB.")
        exit(-1)

    # look for DB creation script in the parent directory of the
    # `load/` folder where this script is present
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

    cats = list(reuters.categories())

    cur.executemany("""
        INSERT INTO LabelTypes(lbltype, lbltype_name) VALUES (?, ?)
    """, enumerate(cats))

    # all labels are binary yes/no indicators
    cur.execute("""
    INSERT INTO LabelDictionary(lbltype, lbl, lblval)
    SELECT DISTINCT lbltype, 0, 'No'  FROM LabelTypes
    UNION
    SELECT DISTINCT lbltype, 1, 'Yes'  FROM LabelTypes
    """)

    # sorting the file IDs is necessary for the validation set splitter
    # to produce even results based on the seqid
    for seqid, text_name in pgb.progressbar(enumerate(sorted(reuters.fileids()))):
        seq = tokenizer(preprocess_doc(text_name))['input_ids']
        split = (0 if text_name.startswith('train')
            else (1 if seqid % 2 == 0 else 2))

        cur.execute("""
            INSERT INTO Sequences(seqid, seqpart) VALUES (?, ?)
        """, (seqid, split))
        
        # save the sequence values:
        for j, s in enumerate(seq):
            cur.execute("""
                INSERT INTO SequenceValues(seqid, svidx, tokid)
                VALUES (?, ?, ?)
            """, (seqid, j, s))

        my_lbls = reuters.categories(text_name)
        # labels time
        for j, cat in enumerate(cats):
            cur.execute("""
                INSERT INTO Labels(lbltype, seqid, lbl) VALUES (?, ?, ?)
            """, (j, seqid, 1 if cat in my_lbls else 0))

    # save BERT tokeniser
    cur.executemany("""
        INSERT INTO Alphabet(tokval, tokid) VALUES (?, ?)
    """, dict(tokenizer.vocab).items())

    cur.execute("COMMIT")
    db.commit()
    db.close()
