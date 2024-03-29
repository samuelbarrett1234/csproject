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

PRAGMA journal_mode = wal;

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
    PRIMARY KEY (lbltype, seqid),
    FOREIGN KEY (lbltype, lbl) REFERENCES LabelDictionary(lbltype, lbl)
);

CREATE INDEX lbl_idx ON Labels(lbltype, lbl, seqid);
CREATE INDEX lbl_idx_2 ON Labels(seqid, lbltype);

-- for each label type, which label is the most common amongst the training set?
CREATE VIEW ModalClasses AS
WITH class_counts AS (
	SELECT lbltype, lbl, COUNT(seqid) AS n
	FROM Sequences NATURAL JOIN Labels
	WHERE seqpart = 0
	GROUP BY lbltype, lbl
)
SELECT lbltype, lbl, MAX(n) AS class_count FROM class_counts
GROUP BY lbltype;

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
    comp_deep INTEGER NOT NULL,  -- 1 if a deep model, else 0

    CHECK(comp_deep IN (0, 1))
);

CREATE TABLE CompressorArchitectureConfig(
    compname TEXT NOT NULL REFERENCES CompressorArchitecture(compname),
    comp_config_key TEXT NOT NULL,
    comp_config_value TEXT NOT NULL,
    PRIMARY KEY(compname, comp_config_key)
);

CREATE TABLE Compressors(
    compid INTEGER PRIMARY KEY,
    compname TEXT NOT NULL REFERENCES CompressorArchitecture(compname),

    -- if the compressor relies on randomness and you want to do repeats,
    -- store the index of the repeat here:
    comprepeat INTEGER NOT NULL DEFAULT 0,
    compdate TEXT NOT NULL,  -- date time of script invocation for creating this compressor

    -- the amount of time training took (Close to 0 for compressors 
    -- which do not require training. However, this may not actually
    -- be 0 in such cases, so do *not* rely on it being 0 for, say,
    -- standard compression algorithms which do not train.)
    -- WARNING: for models which have already been trained, this only
    -- counts the time it takes to load the model from a file! Thus
    -- you probably want to argmax over all models from a fixed training
    -- configuration to account for this.
    comp_train_time REAL NULL,

    -- the amount of time required to compress the whole dataset.
    comp_compress_time REAL NULL,

    UNIQUE(compname, comprepeat)
);

CREATE TABLE CompressionSizes(
    seqid INTEGER NOT NULL REFERENCES Sequences(seqid) ON DELETE CASCADE,
    compid INTEGER NOT NULL REFERENCES Compressors(compid) ON DELETE CASCADE,
    compsz INTEGER NOT NULL,  -- the number of characters that the compressor took the given sequence down to
    PRIMARY KEY(compid, seqid)
);

CREATE TABLE NCDValues(
    -- INVARIANT: `seqid.seq_is_pair = 1`
    seqid INTEGER NOT NULL REFERENCES Sequences(seqid) ON DELETE CASCADE,
    compid INTEGER NOT NULL REFERENCES Compressors(compid) ON DELETE CASCADE,
    ncd_formula TEXT NOT NULL,
    ncd_value REAL NOT NULL,
    PRIMARY KEY(ncd_formula, compid, seqid)
);

CREATE TABLE TrainingPairings (
    ncd_formula TEXT NOT NULL,
    compid INTEGER NOT NULL,
    seqid_train INTEGER NOT NULL,
    seqid_other INTEGER NOT NULL,
    ncd_value REAL NOT NULL,
    PRIMARY KEY (compid, ncd_formula, seqid_other, seqid_train)
);  /*This table is computed by the following query and acts as a materialised view:

SELECT compid, ncd_formula, seqid_left AS seqid_train, seqid_right AS seqid_other, ncd_value
FROM SequencePairings JOIN Sequences ON seqid_left = Sequences.seqid
JOIN NCDValues ON seqid_out = NCDValues.seqid
WHERE Sequences.seqpart = 0
UNION
SELECT compid, ncd_formula, seqid_right AS seqid_train, seqid_left AS seqid_other, ncd_value
FROM SequencePairings JOIN Sequences ON seqid_right = Sequences.seqid
JOIN NCDValues ON seqid_out = NCDValues.seqid
WHERE Sequences.seqpart = 0
*/

CREATE INDEX TrainPairIndex ON TrainingPairings(compid, ncd_formula, seqid_other, seqid_train);

CREATE TABLE Predictions(
    -- INVARIANT: `seqid.seq_is_pair = 0`
    seqid INTEGER NOT NULL REFERENCES Sequences(seqid) ON DELETE CASCADE,
    compid INTEGER NOT NULL REFERENCES Compressors(compid) ON DELETE CASCADE,
    ncd_formula TEXT NOT NULL,
    predictor TEXT NOT NULL,
    lbltype INTEGER NOT NULL REFERENCES LabelTypes(lbltype) ON DELETE CASCADE,
    lbl INTEGER NOT NULL,
    PRIMARY KEY(lbltype, predictor, ncd_formula, compid, seqid)
);
CREATE INDEX PredictionsIndex ON Predictions(seqid, lbltype, compid, ncd_formula, predictor);


CREATE VIEW PredictionEvaluations AS
SELECT seqid, compid, ncd_formula, predictor, lbltype,
Predictions.lbl as pred_lbl, Labels.lbl as true_lbl,
(CASE WHEN Predictions.lbl = Labels.lbl THEN 1 ELSE 0 END) AS correct,
-- these fields only make sense in the binary case
(CASE WHEN Predictions.lbl = 1 AND Labels.lbl = 1 THEN 1 ELSE 0 END) AS true_positive,
(CASE WHEN Predictions.lbl = 0 AND Labels.lbl = 0 THEN 1 ELSE 0 END) AS true_negative,
(CASE WHEN Predictions.lbl = 1 AND Labels.lbl = 0 THEN 1 ELSE 0 END) AS false_positive,
(CASE WHEN Predictions.lbl = 0 AND Labels.lbl = 1 THEN 1 ELSE 0 END) AS false_negative
FROM Predictions JOIN Labels USING (seqid, lbltype);


CREATE TABLE Results(
    lbltype INTEGER NOT NULL,
    compid INTEGER NOT NULL,
    predictor TEXT NOT NULL,
    ncd_formula TEXT NOT NULL,

    val_true_positive INTEGER NOT NULL,
    val_true_negative INTEGER NOT NULL,
    val_false_positive INTEGER NOT NULL,
    val_false_negative INTEGER NOT NULL,

    test_true_positive INTEGER NOT NULL,
    test_true_negative INTEGER NOT NULL,
    test_false_positive INTEGER NOT NULL,
    test_false_negative INTEGER NOT NULL,

    PRIMARY KEY(lbltype, compid, predictor, ncd_formula)
);


CREATE VIEW ResultAccuracies AS
SELECT lbltype, compid, predictor, ncd_formula,
-- view old fields
val_true_positive, val_true_negative, val_false_positive, val_false_negative,
test_true_positive, test_true_negative, test_false_positive, test_false_negative,
-- compute total counts and correct counts, hence accuracy
(val_true_positive + val_true_negative + val_false_positive + val_false_negative) AS val_total,
(test_true_positive + test_true_negative + test_false_positive + test_false_negative) AS test_total,
(val_true_positive + val_true_negative) AS val_correct,
(test_true_positive + test_true_negative) AS test_correct,
(val_true_positive + val_true_negative) / (val_true_positive + val_true_negative + val_false_positive + val_false_negative) AS val_acc,
(test_true_positive + test_true_negative) / (test_true_positive + test_true_negative + test_false_positive + test_false_negative) AS test_acc,
-- compute F1 scores
val_true_positive / (val_true_positive + 0.5 * (val_false_negative + val_false_positive)) AS val_f1,
test_true_positive / (test_true_positive + 0.5 * (test_false_negative + test_false_positive)) AS test_f1
FROM Results;


CREATE VIEW BestCompPredMethod AS
SELECT lbltype, compid, predictor, ncd_formula, MAX(val_acc) AS val_acc, test_acc
FROM ResultAccuracies
GROUP BY lbltype, compid;

-- what accuracy does the "modal classifier" achieve?
CREATE VIEW ModalAccuracies AS
SELECT lbltype, AVG(CASE WHEN Labels.lbl = ModalClasses.lbl THEN 1 ELSE 0 END) AS modal_acc
FROM Sequences NATURAL JOIN Labels JOIN ModalClasses USING (lbltype)
GROUP BY lbltype;


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


CREATE TABLE PairwiseDistances(
    compid INTEGER NOT NULL REFERENCES Compressors(compid),
    ncd_formula TEXT NOT NULL,
    dist_aggregator TEXT NOT NULL,
    seqid_1 INTEGER NOT NULL REFERENCES Sequences(seqid),
    seqid_2 INTEGER NOT NULL REFERENCES Sequences(seqid),
    seqid_train INTEGER NOT NULL REFERENCES Sequences(seqid),
    dist REAL NOT NULL,
    PRIMARY KEY(compid, ncd_formula, dist_aggregator, seqid_1, seqid_2)
);


CREATE TABLE Cuts(
    lbltype INTEGER NOT NULL REFERENCES LabelTypes(lbltype),  -- a binary label type
    compid INTEGER NOT NULL REFERENCES Compressors(compid),
    ncd_formula TEXT NOT NULL,
    dist_aggregator TEXT NOT NULL,
    seqpart INTEGER NOT NULL,
    n_rand INTEGER NOT NULL,  -- if 0 not randomised (true value), else randomised index (sorted ascending)
    cut_value REAL NOT NULL,  -- the weight of the cut
    PRIMARY KEY(lbltype, compid, ncd_formula, dist_aggregator, seqpart, n_rand)
);
