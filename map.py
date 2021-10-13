"""
Given a compression scheme, apply it to all of the data to generate
the compressions.
TODO: since we only need to store the compression length, we could
not bother storing the actual data?
NOTE: atomicity of the DB is important here, only commit at the end
in case training fails.
NOTE: don't forget that for SOME compressors, repeat calls to this
script will give different results.
NOTE: might want auxiliary data like the chosen BERT masking procedure
NOTE: compressors need to be easily chained, so having the ability to
store the compressed data in the DB (together with the alphabets) will
be helpful. Since, during chaining, we need to fully train the first
one before we can start the second one, we need an intermediate place
to store that data anyway.
"""