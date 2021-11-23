# CS Project
My Fourth Year CS Project at the University of Oxford.
In this repository I have implemented a framework for testing different compression algorithms' abilities to classify data via the _normalised compression distance_ method.
Included in the compression algorithms are novel BERT-based ones that I have developed.

## Setup

- Install the requirements given in `requirements.txt`.
- Open the Python shell, run `import nltk` then `nltk.download('punkt')`.
- In the root of the repository directory, run `python compressors/setup.py build_ext --inplace`.

## How to use this repository

- Pick a script in the `load/` folder to create a new DB and load the data. See below for a discussion.
- If the dataset requires unkification, `unkify.py` does this.
- Once the data has been loaded, call `pair_up.py`, which creates pairings of the data points to train on (recall that the NCD method operates on pairs of inputs). This makes use of a _comma code_.
**Warning:** by default, this script creates a new element in the alphabet to denote commas.
However, if you are using a pretrained model, this will cause an error at training time.
Instead in this case, you must pass the token which denotes the separator.
For BERT's NLP tokenisation, this is the `[SEP]` token, so you would add `--use-comma "[SEP]"` to the command.
You must also use `--squash-start-end` for NLP BERT-tokenised data.
- Once the data has been paired-up, it's time to start compressing it, using `compress.py`.
It applies the compressor to all of the data in the DB.
Some compressors have to train first, like the BERT compressor.
- Once you have invoked the relevant compressors, `compute_ncd.py` will use the compressibilities achieved to compute all of the different NCD similarities we are interested in.
- Once the similarities are computed, `classify.py` uses all of the different classification methods (e.g. 1-NN) to assign label distributions to the sequences.
- Once the classifications have been performed, `results.py` collects the results together and outputs it to a CSV file, but also storing in the DB.
- While you can analyse the results by hand, the script `check_hypotheses.py` does this in an automated fashion.
It also outputs its findings to CSV.

## What to expect

- There is one database file for each dataset (where _dataset_ is taken to mean labelled, partitioned sequences over some predetermined alphabet). Note that it is permitted to have many labellings per dataset.
- Where possible the scripts are parallelised. There should be no reason to want to run scripts in parallel.
- Scripts are mostly organised by their "axis of parameterisation". Specifically, there are many independent things we would like to compare (the possibility of a dataset having many labels, the compressor itself, the NCD formula, and the classification method given the dissimilarities).

## Repository structure
- The files in the root of the repository are all executable scripts.
- The files in the `load/` folder are all executable scripts.
- The `compressors` folder is a package, containing compression algorithms.
- The `bnb` folder is a package, containing code for solving the Branch and Bound optimisation problem for masking.

# Datasets
- The `load/gen_random.py` data loader is clearly not associated with any dataset in particular, and just generates arbitrary data for testing.
- The `load/jeopardy.py` dataset can be downloaded from [the JSON file linked here](https://www.reddit.com/r/datasets/comments/1uyd0t/200000_jeopardy_questions_in_a_json_file/).