"""
After classification, this script collects together the performances
of all of the model variations, and outputs them to a CSV.
"""


import os
import argparse as ap


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("db", type=str,
                        help="Filename of the DB to load.")
    args = parser.parse_args()

    if not os.path.isfile(args.db):
        print("Error: cannot find ", args.db)
        exit(-1)

    raise NotImplementedError()
