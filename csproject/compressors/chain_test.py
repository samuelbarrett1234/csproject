import tempfile
import numpy as np
from compressors.chain import _shuffle_file_lines


def test_shuffle_file_lines():
    with tempfile.TemporaryFile(mode="w+") as f:
        A = np.random.randint(0, 200, size=(100, 32))
        f.write("\n".join([
            " ".join(map(str, row)) for row in A
        ]))
        f.seek(0)
        _shuffle_file_lines(f)
        f.seek(0)
        print(f.read())
        f.seek(0)
        B = np.zeros_like(A)
        for i, line in enumerate(f.readlines()):
            row = np.array(list(map(int, line.split())), dtype=np.int32)
            assert(row.shape == (32,))
            B[i, :] = row
    vals_a, cnts_a = np.unique(A, return_counts=True)
    vals_b, cnts_b = np.unique(B, return_counts=True)
    assert(np.all(vals_a == vals_b))
    assert(np.all(cnts_a == cnts_b))
