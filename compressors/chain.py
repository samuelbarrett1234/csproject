"""
This compressor trains and runs a chain of compressors.
It will have to use temporary storage (DB? Temporary file?) to
store the intermediate compression results. Because in order to
train the 2nd compressor you need the 1st compressor to have done
a pass over all of the data.
"""