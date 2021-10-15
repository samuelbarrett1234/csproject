"""
Compute all of the similarity distances (NCDs).
"""


import os
import argparse as ap


NCD_FORMULAE = {

}


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("db", type=str,
                        help="Filename of the DB to load.")
    args = parser.parse_args()

    if not os.path.isfile(args.db):
        print("Error: cannot find ", args.db)
        exit(-1)

    raise NotImplementedError()
