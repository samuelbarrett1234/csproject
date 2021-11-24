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


# compressors accept two arguments in the constructor:
# the data directory, to save any model information (only
# used by deep learning models), and `rep`, the repeat index,
# used for constructing distinct filenames for model data
# corresponding to different runs
COMPRESSORS = {
    'bzip2': lambda data_dir, name, rep: comp.Chain([comp.Huffman(256), comp.BZ2()]),
    'gzip': lambda data_dir, name, rep: comp.Chain([comp.Huffman(256), comp.GZip()]),
    'lzma': lambda data_dir, name, rep: comp.Chain([comp.Huffman(256), comp.LZMA()]),
    'zlib': lambda data_dir, name, rep: comp.Chain([comp.Huffman(256), comp.ZLib()]),
    'bert': lambda data_dir, name, rep, **kwargs: comp.BERT(
        data_dir, name, train_repeat=rep, out_alphabet_sz=256,  # byte
        **kwargs
    )
}


def parse_config(config_str):
    if config_str == '':
        return None

    for stmt in config_str.split(','):
        stmt = stmt.split('=')
        assert(len(stmt) == 2)
        # now try to infer the type of stmt[1]
        if stmt[1] in ('True', 'False', 'None'):
            yield stmt[0], eval(stmt[1])
        else:
            try:
                yield stmt[0], int(stmt[1])
            except ValueError:
                try:
                    yield stmt[0], float(stmt[1])
                except ValueError:
                    yield stmt[0], stmt[1]


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("db", type=str,
                        help="Filename of the DB to load.")
    parser.add_argument("model_data_dir", type=str,
                        help="Directory to place model data.")
    parser.add_argument("compressor", type=str,
                        help="The name of the compressor to run. Must "
                             "be one of: " + ", ".join(COMPRESSORS.keys()))
    parser.add_argument("--comp-config", type=str, default="",
                        help="Optional configurations to pass "
                             "to model constructor. Must be a comma-"
                             "separated list of key=value pairs.")
    args = parser.parse_args()

    if not os.path.isfile(args.db):
        print("Error: cannot find ", args.db)
        exit(-1)

    if not os.path.isdir(args.model_data_dir):
        print("Error:", args.model_data_dir, "is not a directory.")
        exit(-1)

    if args.compressor not in COMPRESSORS:
        print("Error:", args.compressor, "is not a valid compressor.")
        print("Allowed values are:", ", ".join(COMPRESSORS.keys()))
        exit(-1)

    config = dict(parse_config(args.comp_config))
    compname = args.compressor
    if len(config) > 0:
        compname += '-' + args.comp_config
    print("*** BEGINNING TRAINING AND RUNNING OF", compname, "***")

    db = sql.connect(args.db, check_same_thread=False)
    cur = db.cursor()
    cur.execute("PRAGMA FOREIGN_KEYS = ON")

    # Start by getting the vocab size

    cur.execute("SELECT MAX(tokid) + 1 FROM Alphabet")
    alphabet_size = cur.fetchone()[0]

    # Now insert into the DB the new compressor's details

    cur.execute("SELECT MAX(compid) + 1 FROM Compressors")
    compid = cur.fetchone()
    if compid is None:
        compid = 0
    elif compid[0] is None:
        compid = 0
    else:
        compid = compid[0]
    cur.execute("SELECT MAX(comprepeat) + 1 FROM Compressors WHERE compname = ?",
                (compname,))
    comprepeat = cur.fetchone()
    if comprepeat is None:
        comprepeat = 0
    elif comprepeat[0] is None:
        comprepeat = 0
    else:
        comprepeat = comprepeat[0]
    compdate = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")

    # ITERATORS

    def create_seq_iterator(ids_iterator):  # given an iterator over sequence IDs, yield their tokens
        cur = db.cursor()
        for id in ids_iterator:
            cur.execute("SELECT tokid FROM SequenceValues WHERE seqid = ? "
                        "ORDER BY svidx ASC", (id,))
            yield list(map(lambda t: t[0], cur.fetchall()))

    def iterate_over_all():
        # it is very important that the ordering of the returned IDs
        # is not arbitrary
        cur = db.cursor()
        cur.execute("SELECT seqid FROM Sequences ORDER BY seqid ASC")
        return map(lambda t: t[0], cur.fetchall())

    def iterate_over_part(part, shuffle):
        cur = db.cursor()
        s = ("ORDER BY RANDOM()" if shuffle else "ORDER BY seqid ASC")
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

    comp = COMPRESSORS[args.compressor](args.model_data_dir, compname, comprepeat, **config)
    compd = comp.train(alphabet_size, iter_train, iter_val)

    # SAVE THE COMPRESSOR'S METADATA

    # insert compressor type if not exists
    cur.execute("SELECT 1 FROM CompressorArchitecture WHERE compname = ?",
                (compname,))
    if cur.fetchone() is None:
        # TODO: extend compressor API with a `.is_deep()` method
        cur.execute("INSERT INTO CompressorArchitecture(compname, "
                    "compd, comp_deep) "
                    "VALUES (?, ?, ?)",
                    (compname, compd, 1 if args.compressor == 'bert' else 0))
        cur.executemany(
            "INSERT INTO CompressorArchitectureConfig("
            "compname, comp_config_key, comp_config_value) "
            "VALUES(?, ?, ?)",
            [(compname, k, v) for k, v in config.items()]
        )
    # insert compressor instance
    cur.execute("INSERT INTO Compressors(compid, compname, comprepeat, compdate) "
                "VALUES (?, ?, ?, ?)",
                (compid, compname, comprepeat, compdate))

    # RUN THE COMPRESSOR ON THE ENTIRE DATASET, SAVE THE RESULTING COMPRESSION SIZES

    print("Compressing dataset...")

    cur.executemany(
        "INSERT INTO CompressionSizes(compid, seqid, compsz) VALUES (?, ?, ?)",
        pgb.progressbar(
            map(
                lambda data_seqid: (compid, data_seqid[1], len(data_seqid[0])),
                zip(comp.compressmany(iter_all()), iterate_over_all())
            )
        ))

    db.commit()
    db.close()
