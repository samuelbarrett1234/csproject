# CS Project
My Fourth Year CS Project at the University of Oxford.
In this repository I have implemented a framework for testing different compression algorithms' abilities to classify data via the _normalised compression distance_ method.
Included in the compression algorithms are novel BERT-based ones that I have developed.

## How to use this repository

- Create an SQLite database by running `create_db.sql`.
- In the `load/` folder you will find scripts for loading the data.
    - `load/random.py` creates a randomly generated dataset, for testing/comparison/control purposes
- Once the data has been loaded, call `pair_up.py`, which creates pairings of the data points to train on (recall that the NCD method operates on pairs of inputs). This makes use of a _comma code_.
- Once the data has been paired-up, it's time to start compressing it. Since compressors are basically string-to-string functions, the script to perform this is called `map.py`.
It applies the compressor to all of the data in the DB.
Some compressors have to train first, like the BERT compressor.
- Once you have invoked the relevant compressors, `compute_ncd.py` will use the compressibilities achieved to compute all of the different NCD similarities we are interested in.
- Once the similarities are computed, `classify.py` uses all of the different classification methods (e.g. 1-NN) to assign label distributions to the sequences.
- Once the classifications have been performed, `results.py` collects the results together and outputs it in a CSV file.

## What to expect

- Where applicable, scripts are designed to be no-ops when called twice in succession.
- Where possible the scripts are parallelised. There should be no reason to want to run scripts in parallel.
- Scripts are mostly organised by their "axis of parameterisation". Specifically, there are many independent things we would like to compare (the possibility of a dataset having many labels, the compressor itself, the NCD formula, and the classification method given the dissimilarities).
