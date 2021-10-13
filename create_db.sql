/*
Data IDs mapping to binary blobs.
For convenience, `dsz` equals the number of bytes in the blob.
*/
CREATE TABLE DataLookups(
    did INTEGER PRIMARY KEY,
    dblob BLOB NOT NULL,
    dsz INTEGER NOT NULL
);

/*

*/
CREATE TABLE Concatenations(
    did_fst INTEGER NOT NULL REFERENCES DataLookups(did),
    did_snd INTEGER NOT NULL REFERENCES DataLookups(did),
    did_out INTEGER NOT NULL REFERENCES DataLookups(did)
);

CREATE TABLE Compressions(
    did_in INTEGER NOT NULL REFERENCES DataLookups(did),
    did_out INTEGER NOT NULL REFERENCES DataLookups(did)
    -- todo: compression algorithm??
);

CREATE TABLE DatasetContents(
    dsid INTEGER NOT NULL,
    did INTEGER NOT NULL REFERENCES DataLookups(did),
    UNIQUE(dsid, did)
)