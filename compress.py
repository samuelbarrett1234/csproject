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
import argparse as ap


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("db", type=str,
                        help="Filename of the DB to load.")
    cmp = parser.add_mutually_exclusive_group()
    # TODO: add different compressor types (with args) to `cmp`
    args = parser.parse_args()

    if not os.path.isfile(args.db):
        print("Error: cannot find ", args.db)
        exit(-1)

    raise NotImplementedError()
