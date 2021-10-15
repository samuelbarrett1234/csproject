"""
Once all of the similarity distances (NCDs) have been computed,
this script uses different classification methods to classify the labels.
"""


import os
import argparse as ap


CLASSIFICATION_METHODS = {
    
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
