"""Every dataset should satisfy certain sanity checks:
- there should be no token in the alphabet not present in the training set
- likewise, there should be no label (for any label type) not present in the training set
This script corrects such errors uniformly.
This task would be pretty trivial if not for the fact that IDs must be contiguous!
"""


import os
import argparse as ap
import sqlite3 as sql


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("db", type=str,
                        help="Filename of the DB to modify.")
    args = parser.parse_args()

    if not os.path.isfile(args.db):
        print("Error: ", args.db, "does not exist.")
        exit(-1)

    db = sql.connect(args.db)
    cur = db.cursor()
    cur.execute("PRAGMA FOREIGN_KEYS = ON")
    cur.execute("BEGIN TRANSACTION")
    cur.execute("PRAGMA DEFER_FOREIGN_KEYS = ON")

    # ASSUMPTION: every element of the alphabet is mentioned
    # at least *somewhere*.

    # firstly, sanity check: do we even need an UNK?
    # (we need to do this, because otherwise the introduction
    # of an extra UNK token would create exactly the problem
    # that this script purports to solve.)
    cur.execute("CREATE TEMP TABLE BadTokIds(tokid INTEGER PRIMARY KEY)")
    cur.execute("""
        INSERT INTO BadTokIds(tokid)
        SELECT tokid
        FROM SequenceValues NATURAL JOIN Sequences
        WHERE seqpart = 0
        GROUP BY tokid
        HAVING COUNT(seqid) <= 1
    """)
    cur.execute("""
        SELECT 1 FROM BadTokIds
    """)
    if cur.fetchone() is not None:
        print("UNKifying the alphabet...")
        cur.execute("SELECT COUNT(*) FROM BadTokIds")
        print("[UNKifying", cur.fetchone()[0], "words]")

        # create UNK character
        cur.execute("""
            INSERT INTO Alphabet(tokid, tokval)
            SELECT MAX(tokid) + 1, "<<UNK>>"
            FROM Alphabet
        """)
        cur.execute("""
            SELECT MAX(tokid) FROM Alphabet
        """)
        unkid = cur.fetchone()[0]
        # unkify
        cur.execute("""
            UPDATE SequenceValues
            SET tokid = ?
            WHERE EXISTS (
                SELECT 1 FROM BadTokIds
                WHERE BadTokIds.tokid = SequenceValues.tokid
            )
        """, (unkid,))
        # now for the tricky part:
        # remove the unused labels and make the result contiguous
        cur.execute("""
            DELETE FROM Alphabet
            WHERE NOT EXISTS (
                SELECT 1 FROM SequenceValues
                WHERE SequenceValues.tokid = Alphabet.tokid
            )
        """)
        cur.execute("""
            SELECT tokid FROM Alphabet ORDER BY tokid ASC
        """)
        update = list(enumerate(map(lambda t: t[0], cur.fetchall())))
        cur.executemany("""
            UPDATE Alphabet
            SET tokid = ? WHERE tokid = ?
        """, update)
        cur.executemany("""
            UPDATE SequenceValues
            SET tokid = ? WHERE tokid = ?
        """, update)
    else:
        print("Did not need to UNK the sequence alphabet.")

    # Now update the labels - similar procedure to the alphabet
    # # sanity check: do we even need an UNK?
    # (see explanation earlier)
    cur.execute("CREATE TEMP TABLE BadLbls(lbltype INTEGER, lbl INTEGER, PRIMARY KEY (lbltype, lbl))")
    cur.execute("""
        INSERT INTO BadLbls(lbltype, lbl)
        SELECT lbltype, lbl
        FROM Labels NATURAL JOIN Sequences
        WHERE seqpart = 0
        GROUP BY lbltype, lbl
        HAVING COUNT(seqid) <= 1
    """)
    cur.execute("SELECT lbltype, lbltype_name FROM LabelTypes")
    lbl_types = cur.fetchall()
    for lbltype, lbltype_name in lbl_types:
        cur.execute("""
            SELECT 1 FROM BadLbls WHERE lbltype = ?
        """, (lbltype,))
        if cur.fetchone() is not None:
            print("UNKifying label type:", lbltype_name)
            cur.execute("SELECT COUNT(*) FROM BadLbls WHERE lbltype = ?",
                        (lbltype,))
            print("[UNKifying", cur.fetchone()[0], "labels]")

            # create UNK character
            cur.execute("""
                INSERT INTO LabelDictionary(lbltype, lbl, lblval)
                SELECT ?, MAX(lbl) + 1, "<<UNK>>"
                FROM LabelDictionary
                WHERE lbltype = ?
            """, (lbltype, lbltype))
            cur.execute("""
                SELECT MAX(lbl) FROM LabelDictionary WHERE lbltype = ?
            """, (lbltype,))
            unkid = cur.fetchone()[0]
            # unkify
            cur.execute("""
                UPDATE Labels
                SET lbl = ?
                WHERE lbltype = ? AND EXISTS (
                    SELECT 1 FROM BadLbls
                    WHERE BadLbls.lbl = Labels.lbl AND BadLbls.lbltype = Labels.lbltype
                )
            """, (unkid, lbltype))
            # now for the tricky part:
            # remove the unused labels and make the result contiguous
            cur.execute("""
                DELETE FROM LabelDictionary
                WHERE NOT EXISTS (
                    SELECT 1 FROM Labels
                    WHERE Labels.lbl = LabelDictionary.lbl
                    AND Labels.lbltype = LabelDictionary.lbltype
                )
            """)
            cur.execute("""
                SELECT lbl FROM LabelDictionary WHERE lbltype = ? ORDER BY lbl ASC
            """, (lbltype,))
            update = list(map(lambda t: (t[0], t[1][0], lbltype), enumerate(cur.fetchall())))
            cur.executemany("""
                UPDATE LabelDictionary
                SET lbl = ?
                WHERE lbl = ? AND lbltype = ?
            """, update)
            cur.executemany("""
                UPDATE Labels
                SET lbl = ?
                WHERE lbl = ? AND lbltype = ?
            """, update)
        else:
            print("Did not need to UNK the label type", lbltype_name)

    cur.execute("COMMIT")
    db.commit()
    db.close()
