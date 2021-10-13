/*
DATA MODEL

Sequences from a dataset are partitioned into train, val and test sets according to `seqpart`.
The whole dataset of sequences are labelled by at least one label type, but possibly more.
Every sequence has tokens from a fixed alphabet.
Sequences can be concatenated to form longer sequences (but we don't do this recursively).
Compressors can then compress a Sequence, where we record the resulting size.
Compressors may not output their data to an alphabet of the same size, so each compressor also makes
note of the alphabet size they use, for comparability between compressors.
Then, given a sequence pair (x, y) and the compression values C(x,y), C(x) and C(y) for a given
compressor C, and an NCD formula, we store the dissimilarity values in NCDValues.
Given a label type and a sequence x and all relevant contrasting sequences y and all of the NCD(x, y) values
for a given NCD formula, together with the labels associated with each of the ys, we can
predict a label for x, which is stored in the Predictions table.
*/

CREATE TABLE Sequences(
    seqid INTEGER PRIMARY KEY,
    seqpart INTEGER NOT NULL,  -- 0 for train, 1 for val, 2 for test
    seq_is_pair INTEGER NOT NULL DEFAULT 0,  -- 0 for singleton, 1 for pair
    CHECK(seq_is_pair IN (0, 1))
);

CREATE INDEX seq_by_part ON Sequences(seqpart, seqid);

CREATE TABLE Alphabet(
    tokid INTEGER PRIMARY KEY,  -- must be zero indexed, so that MAX(tokid)+1 is the alphabet size
    tokval TEXT NULL  -- optional, interpretation depends on the data
);

CREATE TABLE SequenceValues(
    seqid INTEGER NOT NULL REFERENCES Sequences(seqid) ON DELETE CASCADE,
    svidx INTEGER NOT NULL,  -- must be zero indexed
    tokid INTEGER NOT NULL REFERENCES Alphabet(tokid) ON DELETE CASCADE,
    PRIMARY KEY(seqid, svidx)
);

CREATE TABLE Labels(
    -- INVARIANT: any referenced `seqid` has `seq_is_pair == 0`.
    seqid INTEGER NOT NULL REFERENCES Sequences(seqid) ON DELETE CASCADE,
    lbltype INTEGER NOT NULL,
    lbl INTEGER NOT NULL
);

CREATE TABLE SequencePairings(
    -- INVARIANT: `seqid_out.seq_is_pair = 1`,
    -- `seqid_left.seq_is_pair = 0`, `seqid_right.seq_is_pair = 0`
    seqid_out INTEGER PRIMARY KEY REFERENCES Sequences(seqid) ON DELETE CASCADE,
    seqid_left INTEGER NOT NULL REFERENCES Sequences(seqid) ON DELETE CASCADE,
    seqid_right INTEGER NOT NULL REFERENCES Sequences(seqid) ON DELETE CASCADE
);

CREATE INDEX seq_pair_left ON SequencePairings(seqid_left, seqid_right, seqid_out);
CREATE INDEX seq_pair_right ON SequencePairings(seqid_right, seqid_left, seqid_out);

CREATE TABLE Compressors(
    compid INTEGER PRIMARY KEY,
    compname TEXT NOT NULL,  -- the name of the compressor
    compd INTEGER NOT NULL  -- the size of the compressor's output alphabet
);

CREATE TABLE CompressionSizes(
    seqid INTEGER NOT NULL REFERENCES Sequences(seqid) ON DELETE CASCADE,
    compid TEXT NOT NULL REFERENCES Compressors(compid) ON DELETE CASCADE,
    compsz INTEGER NOT NULL,  -- the number of characters that the compressor took the given sequence down to
    PRIMARY KEY(compid, seqid)
);

CREATE TABLE NCDValues(
    -- INVARIANT: `seqid.seq_is_pair = 1`
    seqid INTEGER NOT NULL REFERENCES Sequences(seqid) ON DELETE CASCADE,
    compid TEXT NOT NULL REFERENCES Compressors(compid) ON DELETE CASCADE,
    ncd_formula TEXT NOT NULL,
    ncd_value REAL NOT NULL,
    PRIMARY KEY(ncd_formula, compid, seqid)
);

CREATE TABLE Predictions(
    -- INVARIANT: `seqid.seq_is_pair = 0`
    seqid INTEGER NOT NULL REFERENCES Sequences(seqid) ON DELETE CASCADE,
    compid TEXT NOT NULL REFERENCES Compressors(compid) ON DELETE CASCADE,
    ncd_formula TEXT NOT NULL,
    predictor TEXT NOT NULL,
    lbltype INTEGER NOT NULL,
    lbl INTEGER NOT NULL,
    PRIMARY KEY(lbltype, predictor, ncd_formula, compid, seqid)
);
