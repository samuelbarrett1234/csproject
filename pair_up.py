"""
Constructs pairings of all datasets (which do not have them already)
to allow training on pairs of sequences.
"""


import os
import argparse as ap


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("db", type=str,
                        help="Filename of the DB to load.")
    # TODO: add arguments about proportions of +ve <x,x> vs -ve <x,y> samples, and singletons <x>
    args = parser.parse_args()

    if not os.path.isfile(args.db):
        print("Error: cannot find ", args.db)
        exit(-1)

    raise NotImplementedError()
