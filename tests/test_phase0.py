"""Phase 0 test — red until you fill the three blanks, then green.

Run from the repo root with:  pytest
"""

import math

from nanogpu import launch
from phase0_vector_add import vector_add


def test_vector_add_small():
    n = 100
    a = [float(i) for i in range(n)]
    b = [float(3 * i) for i in range(n)]
    c = [0.0] * n

    block_dim = 32
    grid_dim = math.ceil(n / block_dim)
    launch(grid_dim, block_dim, vector_add, a, b, c)

    assert c == [a[i] + b[i] for i in range(n)]
