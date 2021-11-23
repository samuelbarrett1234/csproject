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

CREATE INDEX seq_vals_idx ON SequenceValues(tokid, seqid);

CREATE TABLE LabelTypes(
    lbltype INTEGER PRIMARY KEY,
    lbltype_name TEXT NULL
);

CREATE TABLE LabelDictionary(
    lbltype INTEGER NOT NULL REFERENCES LabelTypes(lbltype) ON DELETE CASCADE,
    lbl INTEGER NOT NULL,  -- must be zero indexed, so that MAX(lbl)+1 is the number of labels
    lblval TEXT NULL,  -- a textual representation of the label
    PRIMARY KEY(lbltype, lbl)
);

CREATE TABLE Labels(
    -- INVARIANT: any referenced `seqid` has `seq_is_pair == 0`.
    seqid INTEGER NOT NULL REFERENCES Sequences(seqid) ON DELETE CASCADE,
    lbltype INTEGER NOT NULL REFERENCES LabelTypes(lbltype) ON DELETE CASCADE,
    lbl INTEGER NOT NULL,
    FOREIGN KEY (lbltype, lbl) REFERENCES LabelDictionary(lbltype, lbl)
);

CREATE INDEX lbl_idx ON Labels(lbl, seqid);

CREATE TABLE SequencePairings(
    -- INVARIANT: `seqid_out.seq_is_pair = 1`,
    -- `seqid_left.seq_is_pair = 0`, `seqid_right.seq_is_pair = 0`
    seqid_out INTEGER PRIMARY KEY REFERENCES Sequences(seqid) ON DELETE CASCADE,
    seqid_left INTEGER NOT NULL REFERENCES Sequences(seqid) ON DELETE CASCADE,
    seqid_right INTEGER NOT NULL REFERENCES Sequences(seqid) ON DELETE CASCADE,
    UNIQUE(seqid_left, seqid_right)  -- the ORDERED pair is unique
);

CREATE INDEX seq_pair_left ON SequencePairings(seqid_left, seqid_right, seqid_out);
CREATE INDEX seq_pair_right ON SequencePairings(seqid_right, seqid_left, seqid_out);

CREATE TABLE CompressorArchitecture(
    compname TEXT PRIMARY KEY,  -- the name of the compressor

    compd INTEGER NOT NULL,  -- the size of the compressor's output alphabet
    comp_fine_tuning TEXT NULL,  -- the fine-tuning method, if applicable
    comp_method TEXT NULL, -- the compression "submethod", if applicable
    comp_deep INTEGER NOT NULL,  -- 1 if a deep model, else 0

    CHECK(comp_deep IN (0, 1))
);

CREATE TABLE Compressors(
    compid INTEGER PRIMARY KEY,
    compname TEXT NOT NULL REFERENCES CompressorArchitecture(compname),

    -- if the compressor relies on randomness and you want to do repeats,
    -- store the index of the repeat here:
    comprepeat INTEGER NOT NULL DEFAULT 0,
    compdate TEXT NOT NULL,  -- date time of script invocation for creating this compressor

    UNIQUE(compname, comprepeat)
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
    lbltype INTEGER NOT NULL REFERENCES LabelTypes(lbltype) ON DELETE CASCADE,
    lbl INTEGER NOT NULL,
    PRIMARY KEY(lbltype, predictor, ncd_formula, compid, seqid)
);


CREATE VIEW ResultAccuracies AS
WITH PredVsTrue AS (
    SELECT Predictions.lbltype, compid, predictor, ncd_formula, seqpart,
    CASE WHEN Predictions.lbl = Labels.lbl THEN 1.0 ELSE 0.0 END AS correct
    FROM Predictions NATURAL JOIN Sequences
    JOIN Labels ON Labels.seqid = Predictions.seqid AND Labels.lbltype = Predictions.lbltype
),
Train AS (
    SELECT lbltype, compid, predictor, ncd_formula, AVG(correct) AS train_acc
    FROM PredVsTrue WHERE seqpart = 0 GROUP BY lbltype, compid, predictor, ncd_formula
),
Val AS (
    SELECT lbltype, compid, predictor, ncd_formula, AVG(correct) AS val_acc
    FROM PredVsTrue WHERE seqpart = 1 GROUP BY lbltype, compid, predictor, ncd_formula
),
Test AS (
    SELECT lbltype, compid, predictor, ncd_formula, AVG(correct) AS test_acc
    FROM PredVsTrue WHERE seqpart = 2 GROUP BY lbltype, compid, predictor, ncd_formula
)
SELECT * FROM Train NATURAL JOIN Val NATURAL JOIN Test;


CREATE VIEW ResultSizes AS
WITH Train AS (
    SELECT compid, AVG(compsz) AS train_sz
    FROM CompressionSizes NATURAL JOIN Sequences
    WHERE seqpart = 0
    GROUP BY compid
),
Val AS (
    SELECT compid, AVG(compsz) AS val_sz
    FROM CompressionSizes NATURAL JOIN Sequences
    WHERE seqpart = 1
    GROUP BY compid
),
Test AS (
    SELECT compid, AVG(compsz) AS test_sz
    FROM CompressionSizes NATURAL JOIN Sequences
    WHERE seqpart = 2
    GROUP BY compid
)
SELECT * FROM Train NATURAL JOIN Val NATURAL JOIN Test;


CREATE VIEW ResultRatios AS
WITH SequenceLengths AS (
    SELECT seqid, MAX(svidx) + 1 AS slen FROM SequenceValues
    GROUP BY seqid
),
Train AS (
    SELECT compid, AVG(compsz / slen) AS train_rt
    FROM CompressionSizes NATURAL JOIN SequenceLengths
    NATURAL JOIN Sequences WHERE seqpart = 0
    GROUP BY compid
),
Val AS (
    SELECT compid, AVG(compsz / slen) AS val_rt
    FROM CompressionSizes NATURAL JOIN SequenceLengths
    NATURAL JOIN Sequences WHERE seqpart = 0
    GROUP BY compid
),
Test AS (
    SELECT compid, AVG(compsz / slen) AS test_rt
    FROM CompressionSizes NATURAL JOIN SequenceLengths
    NATURAL JOIN Sequences WHERE seqpart = 0
    GROUP BY compid
)
SELECT * FROM Train NATURAL JOIN Val NATURAL JOIN Test;


CREATE TABLE LabelScores(
    compid INTEGER NOT NULL REFERENCES Compressors(compid),
    ncd_formula TEXT NOT NULL,
    lbltype INTEGER NOT NULL REFERENCES LabelTypes(lbltype),
    score REAL NOT NULL,
    PRIMARY KEY(compid, ncd_formula, lbltype)
);
