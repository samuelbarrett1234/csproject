# CS Project
My Fourth Year CS Project at the University of Oxford.
In this repository I have implemented a framework for testing different compression algorithms' abilities to classify data via the _normalised compression distance_ method.
Included in the compression algorithms are novel BERT-based ones that I have developed.

## Setup

- Install the requirements given in `requirements.txt`.
- Open the Python shell, run `import nltk` then `nltk.download('punkt')` and `nltk.download('reuters')`.
- In the root of the `csproject` subdirectory of the root, run `python compressors/setup.py build_ext --inplace`.
- Create a directory named whatever you like (e.g. `build`) in the root of this repository, change directory into it, and run `cmake ../csproject_pairwise_dists/ && make`.
This will create an executable which can be run as `build/csproject_pairwise_dists $DB` where `$DB` is your database filename.
It is just a more efficient way of computing the pairwise distances than in Python!

## How to use this repository

- Pick a script in the `load/` folder to create a new DB and load the data. See below for a discussion.
- If the dataset requires unkification, `unkify.py` does this.
- Once the data has been loaded, call `pair_up.py`, which creates pairings of the data points to train on (recall that the NCD method operates on pairs of inputs). This makes use of a _comma code_.
**Warning:** by default, this script creates a new element in the alphabet to denote commas.
However, if you are using a pretrained model, this will cause an error at training time.
Instead in this case, you must pass the token which denotes the separator.
For BERT's NLP tokenisation, this is the `[SEP]` token, so you would add `--use-comma "[SEP]"` to the command.
You must also use `--squash-start-end` for NLP BERT-tokenised data.
If you have multiple label types and you want to pair them all up in a single command, the following may be useful:
`sqlite3 <DB-FILENAME> "SELECT lbltype_name FROM LabelTypes" | xargs -I {} python pair_up.py <DB-FILENAME> {} <YOUR-K-HERE> --use-comma "[SEP]" --squash-start-end`
- Once the data has been paired-up, it's time to start compressing it, using `compress.py`.
It applies the compressor to all of the data in the DB.
Some compressors have to train first, like the BERT compressor.
- Once you have invoked the relevant compressors, `compute_ncd.py` will use the compressibilities achieved to compute all of the different NCD similarities we are interested in.
- Then, run the executable you should've built earlier using CMake, called `csproject_pairwise_dists`, passing the DB as the only parameter.
This fills in a handy table called `PairwiseDistances` which is useful for downstream classification.
- Once the similarities are computed, `classify.py` can be used to make classifications (you input the type of classifier you want to use).
There are additional classification scripts, e.g. `classify_mst.py`.
- Then, `compute_accs.py` computes the accuracies of each prediction method.

## What to expect

- There is one database file for each dataset (where _dataset_ is taken to mean labelled, partitioned sequences over some predetermined alphabet). Note that it is permitted to have many labellings per dataset.
- Where possible the scripts are parallelised. There should be no reason to want to run scripts in parallel.
- Scripts are mostly organised by their "axis of parameterisation". Specifically, there are many independent things we would like to compare (the possibility of a dataset having many labels, the compressor itself, the NCD formula, and the classification method given the dissimilarities).

## Repository structure
- All Python code is in `csproject/`
- These are all executable Python scripts.
- The files in the `csproject/load/` folder are all executable scripts, for loading various datasets.
- The `csproject/compressors` folder is a package, containing compression algorithms.
- The `csproject/bnb` folder is a package, containing code for producing different token orderings, given a model.
- The `csproject_pairwise_dists/` contains a CMake C++ project which is just an optimised version of a particular SQL query.
SQLite makes me sad sometimes.

# Datasets
- The `load/gen_random.py` data loader is clearly not associated with any dataset in particular, and just generates arbitrary data for testing.
- The `load/jeopardy.py` dataset can be downloaded from [the JSON file linked here](https://www.reddit.com/r/datasets/comments/1uyd0t/200000_jeopardy_questions_in_a_json_file/).
- The `load/sent140.py` dataset is available from [Kaggle](https://www.kaggle.com/kazanova/sentiment140).
