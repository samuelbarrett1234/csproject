"""
Given a compression scheme, apply it to all of the data to generate
the compressions.
NOTE: atomicity of the DB is important here, only commit at the end
in case training fails.
NOTE: don't forget that for SOME compressors, repeat calls to this
script will give different results.
NOTE: might want auxiliary data like the chosen BERT masking procedure
"""


import os
import datetime
import argparse as ap
import sqlite3 as sql
import progressbar as pgb
import compressors as comp


COMPRESSORS = {
    'Huffman-2-ary-3-gram': comp.Chain([comp.Block(3), comp.Huffman(2)])
}


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("db", type=str,
                        help="Filename of the DB to load.")
    parser.add_argument("compressor", type=str,
                        help="The name of the compressor to run. Must "
                             "be one of: " + ", ".join(COMPRESSORS.keys()))
    args = parser.parse_args()

    if not os.path.isfile(args.db):
        print("Error: cannot find ", args.db)
        exit(-1)

    if args.compressor not in COMPRESSORS:
        print("Error:", args.compressor, "is not a valid compressor.")
        print("Allowed values are:", ", ".join(COMPRESSORS.keys()))
        exit(-1)

    db = sql.connect(args.db)

    # Start by getting the vocab size

    cur = db.cursor()
    cur.execute("SELECT MAX(tokid) + 1 FROM Alphabet")
    alphabet_size = cur.fetchone()[0]

    # Now insert into the DB the new compressor's details

    cur.execute("SELECT MAX(compid) + 1 FROM Compressors")
    compid = cur.fetchone()
    if compid is None:
        compid = 0
    else:
        compid = compid[0]
    cur.execute("SELECT MAX(comprepeat) + 1 FROM Compressors WHERE compname = ?",
                (args.compressor,))
    comprepeat = cur.fetchone()
    if comprepeat is None:
        comprepeat = 0
    else:
        comprepeat = comprepeat[0]
    compdate = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")

    # ITERATORS

    def create_seq_iterator(ids_iterator):  # given an iterator over sequence IDs, yield their tokens
        cur = db.cursor()
        for id in ids_iterator:
            cur.execute("SELECT tokid FROM SequenceValues WHERE seqid = ? "
                        "ORDER BY svidx ASCENDING", (id,))
            yield list(map(lambda t: t[0], cur.fetchall()))

    def iterate_over_all():
        # it is very important that the ordering of the returned IDs
        # is not arbitrary
        cur = db.cursor()
        cur.execute("SELECT seqid FROM Sequences ORDER BY seqid ASCENDING")
        return map(lambda t: t[0], cur.fetchall())

    def iterate_over_part(part, shuffle):
        cur = db.cursor()
        s = ("ORDER BY RANDOM()" if shuffle else "ORDER BY seqid ASCENDING")
        cur.execute("SELECT seqid FROM Sequences WHERE seqpart = ? " + s, (part,))
        return map(lambda t: t[0], cur.fetchall())

    def iter_train():
        return create_seq_iterator(iterate_over_part(0, True))

    def iter_val():
        return create_seq_iterator(iterate_over_part(1, False))

    def iter_all():
        return create_seq_iterator(iterate_over_all())

    # NOW CONSTRUCT COMPRESSOR

    print("Creating/training compressor...")

    comp = COMPRESSORS[args.compressor]()
    compd = comp.train(alphabet_size, iter_train, iter_val)

    # SAVE THE COMPRESSOR'S METADATA

    cur.execute("INSERT INTO Compressors(compid, compname, comprepeat, compdate, "
                "compd) VALUES (?, ?, ?, ?, ?)",
                (compid, args.compressor, comprepeat, compdate, compd))

    # RUN THE COMPRESSOR ON THE ENTIRE DATASET, SAVE THE RESULTING COMPRESSION SIZES

    print("Compressing dataset...")

    cur.executemany(
        "INSERT INTO CompressionSizes(compid, seqid, compsz) VALUES (?, ?, ?)",
        pgb.progressbar(
            map(
                lambda data, seqid: (compid, seqid, len(data)),
                zip(comp.compressmany(iter_all()), iterate_over_all())
            )
        ))

    db.commit()
    db.close()
