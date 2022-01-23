"""This script loads the sentiment140 dataset. Available from
https://www.kaggle.com/kazanova/sentiment140
(Warning: make sure you shuffle the file lines before calling
this script, if you are using a limit. Otherwise all of the
sequences will have the same label.)
"""


import re
import io
import os
import sys
import csv
import argparse as ap
import sqlite3 as sql
import progressbar as pgb
from transformers import BertTokenizer


# copied from https://stackoverflow.com/questions/6718633/python-regular-expression-again-match-url
URL_REGEX = r"((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*\/?"
AT_AND_HASH_REGEX = r"(@|#)[a-zA-Z0-9]+"
EMOJIS = r"(\<3)|(\:\))|(\:\()|(\:D)|(\:P)|(\;\))|(\;\()|(\;D)|(\;P)"
tweet_re = re.compile('(' + URL_REGEX + ')|(' + AT_AND_HASH_REGEX + ')|(' + EMOJIS + ')')


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
    return {
        'sent': ('neg' if int(row[0]) == 0 else 'pos')
    }


def preprocess_tweet(tweet):
    # several things needed for cleaning here:
    # remove hyperlinks, hashtags and @s
    # filter out non-English tweets?
    # filter out unicode characters / emojis
    # remove emojis like :) or <3
    # strip whitespace on either side
    # and things like &amp; &quot;
    tweet = tweet.replace('&quot;', '"').replace('&amp;', '&').replace('&lt;', '<').replace('&gt', '>')
    return tweet_re.sub('', tweet).strip()


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("db", type=str,
                        help="Filename of the DB to create.")
    parser.add_argument("sent140", type=str,
                        help="Path to the sentiment140 CSV file.")
    parser.add_argument("--limit", default=None, type=int,
                        help="Limit the total size of the dataset.")
    args = parser.parse_args()

    if os.path.isfile(args.db):
        print("Error: ", args.db, "exists. This script creates a new DB.")
        exit(-1)

    if not os.path.isfile(args.sent140):
        print("Error: ", args.sent140, "does not exist.")
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
    with io.open(args.sent140, mode="r", encoding="latin-1") as f:
        csvf = csv.reader(f)
        for i, row in pgb.progressbar(enumerate(csvf)):
            # limit cutoff, if applicable
            # WARNING: this assumes the file is sorted, NOT ordered
            # on the label, as it is by default!
            if args.limit is not None and i >= args.limit:
                break

            # reject strings with non acsii characters
            try:
                _ = row[-1].encode(encoding='UTF-8', errors='strict')
            except UnicodeEncodeError:
                print("Warning: non-UTF-8 characters encountered on line", i, ", skipping...")
                continue
            
            # 80/10/10 data split
            if i % 10 < 8:
                split = 0  # train
            elif i % 10 == 8:
                split = 1  # val
            else:
                split = 2  # test

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
            for j, s in enumerate(tokenizer(preprocess_tweet(row[-1]))['input_ids']):
                cur.execute("""
                    INSERT INTO SequenceValues(seqid, svidx, tokid)
                    VALUES (?, ?, ?)
                """, (seqid, j, s))

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
