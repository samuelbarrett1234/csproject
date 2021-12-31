"""
To reduce the number of label values to predict,
you can call this script, which creates a new label
type grouping the least frequent label values together.
"""

import os
import argparse as ap
import sqlite3 as sql


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("db", type=str,
                        help="Filename of the DB to modify.")
    parser.add_argument("lbltype_name", type=str,
                        help="The label type to unkify.")
    parser.add_argument("n", type=int,
                        help="The number of non-unked label "
                             "values to permit.")
    args = parser.parse_args()

    if not os.path.isfile(args.db):
        print("Error: ", args.db, "does not exist.")
        exit(-1)

    db = sql.connect(args.db)
    cur = db.cursor()
    cur.execute("PRAGMA FOREIGN_KEYS = ON")
    cur.execute("BEGIN TRANSACTION")
    cur.execute("PRAGMA DEFER_FOREIGN_KEYS = ON")

    cur.execute("SELECT 1 FROM LabelTypes WHERE lbltype_name = ?",
                (args.lbltype_name,))
    if cur.fetchone() is None:
        print("Error: label type not found in the DB.")
        exit(-1)

    new_lbltype_name = args.lbltype_name + "-unkified"
    cur.execute("SELECT 1 FROM LabelTypes WHERE lbltype_name = ?",
                (new_lbltype_name,))
    if cur.fetchone() is not None:
        print("Error: label type already unkified.")
        exit(-1)

    cur.execute("SELECT MAX(lbltype) + 1 FROM LabelTypes")
    next_lbltype_id = cur.fetchone()[0]

    cur.execute("INSERT INTO LabelTypes(lbltype, lbltype_name)"
                " VALUES (?, ?)",
                (next_lbltype_id, new_lbltype_name))

    # copy over the `args.n` most frequently occurring labels
    # from the input label type to our new label type
    cur.execute("""
    INSERT INTO LabelDictionary(lbltype, lbl, lblval)
    SELECT ?, ROW_NUMBER() OVER (ORDER BY COUNT(seqid) DESC) - 1), lblval
    FROM Labels NATURAL JOIN LabelTypes NATURAL JOIN LabelDictionary
    WHERE lbltype_name = ?
    GROUP BY lbl, lblval
    ORDER BY COUNT(seqid) DESC
    LIMIT ?
    """, (next_lbltype_id, args.lbltype_name, args.n))

    # when inserting the new UNK character, what will its ID be?
    cur.execute("SELECT MAX(lbl) + 1 FROM LabelDictionary "
                "WHERE lbltype = ?", (next_lbltype_id,))
    unk_id = cur.fetchone()[0]
    assert(unk_id <= args.n)
    # when inserting the new UNK character, what will its textual
    # representation be?
    unk_val = "UNK"
    while cur.execute("SELECT 1 FROM LabelDictionary WHERE "
                      "lbltype = ? AND lblval = ?",
                      (next_lbltype_id, unk_val)).fetchone() is not None:
        unk_val = "<" + unk_val + ">"
    # now insert it
    cur.execute("INSERT INTO LabelDictionary(lbltype, lbl, lblval)"
                " VALUES(?, ?, ?)", (next_lbltype_id, unk_id, unk_val))

    # now copy over all of the sentence labellings
    cur.execute("""
    INSERT INTO Labels(seqid, lbltype, lbl)

    WITH LabelMapping AS (

        SELECT A.lbl AS lbl_old, B.lbl AS lbl_new
        FROM LabelTypes NATURAL JOIN LabelDictionary AS A
        JOIN LabelDictionary AS B ON A.lblval = B.lblval
        WHERE lbltype_name = ? AND B.lbltype = ?

        UNION ALL

        SELECT A.lbl AS lbl_old, ? AS lbl_new
        FROM LabelTypes NATURAL JOIN LabelDictionary AS A
        WHERE lbltype_name = ? AND NOT EXISTS (
            SELECT 1 FROM LabelDictionary AS B
            WHERE A.lblval = B.lblval AND B.lbltype = ?
        )

    )

    SELECT seqid, ?, lbl_new
    FROM Labels JOIN LabelMapping
    ON Labels.lbl = LabelMapping.lbl_old
    """, (args.lbltype_name, next_lbltype_id, unk_id,
          args.lbltype_name, next_lbltype_id, next_lbltype_id))

    cur.execute("COMMIT")
    db.commit()
    db.close()
