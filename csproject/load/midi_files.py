"""This script loads an arbitrary collection of MIDI files.
"""


import os
import sys
import glob
import bisect
import pathlib
import argparse as ap
import sqlite3 as sql
import progressbar as pgb
from miditok import OctupleEncoding
from miditoolkit import MidiFile


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


tokenizer = OctupleEncoding()


def process_tokenised_seq(seq):
    # `seq` contains lots of data, but we're only going to be modelling
    # the melody.

    # extract pitch sequence
    seq = [s[0] for s in seq]
    # increment by 4 to leave space for the 4 special tokens
    seq = [s + 4 for s in seq]
    # add [CLS]/[SEP]
    seq = [0] + seq + [1]
    return seq


def sentence_length(q):
    # bucket the actual lengths into groups of 5, stopping at 30(=6*5)
    return min(len(q) // 5, 6)


def labels(file_path, midi_file, seq_tokenised):
    lbls = {
        'slen': sentence_length(seq_tokenised)
    }
    # use folder as a proxy for genre or other labelling
    # schemes
    # it is very important that we reverse this list,
    # that way it allows for the possibility that not
    # all MIDI files are found at the same file system
    # depth
    parents = list(map(lambda p: p.stem,
        pathlib.Path(file_path).resolve().parents
    ))[::-1]
    for i, p in enumerate(parents):
        lbls['parent' + str(i)] = p
    return lbls


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("db", type=str,
                        help="Filename of the DB to create.")
    parser.add_argument("midi_files", type=str,
                        help="The files to process.")
    args = parser.parse_args()

    if os.path.isfile(args.db):
        print("Error: ", args.db, "exists. This script creates a new DB.")
        exit(-1)

    create_db_sql = os.path.join(os.path.dirname(sys.argv[0]), "../", "create_db.sql")
    if not os.path.isfile(create_db_sql):
        print("Error: cannot find the DB creation script `create_db.sql`.",
              "Expected at:", create_db_sql)
        exit(-1)

    files = list(filter(os.path.isfile,
                        glob.glob(args.midi_files, recursive=True)))
    if len(files) == 0:
        print("Error: no files were matched by pattern", args.midi_files)
        exit(-1)
    else:
        print("Processing", len(files), "files...")

    db = sql.connect(args.db)
    cur = db.cursor()

    with open(create_db_sql, "r") as f:
        cur.executescript(f.read())

    cur.execute("PRAGMA FOREIGN_KEYS = ON")
    cur.execute("BEGIN TRANSACTION")
    cur.execute("PRAGMA DEFER_FOREIGN_KEYS = ON")

    # 80/10/10 data split
    data_split = [int(0.9 * len(files)), int(0.05 * len(files))]
    data_split[1] += data_split[0]
    data_split.append(len(files))

    lbl_types = {}
    lbl_type_ids = Alphabet()
    for seqid, filename in enumerate(pgb.progressbar(files)):
        try:
            mf = MidiFile(filename)
            seq = tokenizer.midi_to_tokens(mf)
        except Exception as e:
            print("Failed to load Midi file", filename)
            print("Error was:", e)
            continue

        # compute which partition this sequence falls into
        split = bisect.bisect_left(data_split, seqid)
        assert(split < len(data_split))

        # save an entry for the sequence itself:
        cur.execute("""
            INSERT INTO Sequences(seqid, seqpart)
            VALUES (?, ?)
        """, (seqid, split))

        # compute and save labels
        lbls = labels(filename, mf, seq)
        for lbl_type, lbl in lbls.items():
            if lbl_type not in lbl_types:
                lbl_types[lbl_type] = Alphabet()
            cur.execute("""
                INSERT INTO Labels(seqid, lbltype, lbl)
                VALUES (?, ?, ?)
            """, (seqid, lbl_type_ids.at(lbl_type), lbl_types[lbl_type].at(lbl)))

        for i, s in enumerate(process_tokenised_seq(seq)):
            cur.execute("""
                INSERT INTO SequenceValues(seqid, svidx, tokid)
                VALUES (?, ?, ?)
            """, (seqid, i, s))

    # save the special values in the alphabet
    cur.executemany("""
        INSERT INTO Alphabet(tokval, tokid) VALUES (?, ?)
    """, [
        ('[CLS]', 0),
        ('[SEP]', 1),
        ('[MASK]', 2),
        ('[PAD]', 3)
    ])
    # save all of the other values
    cur.execute("""
        INSERT INTO Alphabet(tokid, tokval)
        SELECT DISTINCT tokid, CAST(tokid AS TEXT)
        FROM SequenceValues
        WHERE tokid NOT IN (0, 1, 2, 3)
    """)
    cur.executemany("""
        INSERT INTO LabelTypes(lbltype_name, lbltype) VALUES (?, ?)
    """, lbl_type_ids.iterate())
    for lbltype_name, lbl_alphabet in lbl_types.items():
        lbltype = lbl_type_ids.at(lbltype_name)
        cur.executemany("""
            INSERT INTO LabelDictionary(lbltype, lblval, lbl)
            VALUES (?, ?, ?)
        """, map(lambda t: (lbltype,) + t, lbl_alphabet.iterate()))

    # note that in our automated labelling scheme
    # based on file paths, we will probably end up
    # with a lot of useless labels (Useless in the
    # sense that they are the same for all sequences).
    # WLOG we can delete any label which takes a single
    # value.
    cur.execute("""
        WITH BoringLabels AS (
            SELECT lbltype
            FROM LabelDictionary
            GROUP BY lbltype
            HAVING COUNT(lbl) = 1
        )
        DELETE FROM LabelTypes
        WHERE EXISTS (
            SELECT 1 FROM BoringLabels
            WHERE BoringLabels.lbltype = LabelTypes.lbltype
        )
    """)

    cur.execute("COMMIT")
    db.commit()
    db.close()
