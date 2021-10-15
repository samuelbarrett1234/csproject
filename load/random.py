"""This script randomly generates data, to be used as a baseline.
It is expected that this data is incompressible.
"""


import os
import argparse as ap


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("db", type=str,
                        help="Filename of the DB to create.")
    parser.add_argument("alphabet-sz", type=int,
                        help="The number of characters in the alphabet.")
    parser.add_argument("length-rate", type=float,
                        help="The rate parameter of the Poisson distribution for length.")
    args = parser.parse_args()

    if os.path.isfile(args.db):
        print("Error: ", args.db, "exists. This script creates a new DB.")
        exit(-1)

    raise NotImplementedError()
