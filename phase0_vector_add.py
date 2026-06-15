"""Phase 0 workload: vector add -- the 'hello world' of GPU programming.

C[i] = A[i] + B[i], with exactly ONE thread responsible for each element i.

Run it with:  python phase0_vector_add.py
"""

import math
from toygpu import launch


def vector_add(t, a, b, c):
    """One thread's job: compute a single element of output vector `c`.

    `t` is this thread's identity (a Thread). Use it to decide WHICH element
    of the arrays this thread is responsible for.
    """
    i = t.global_id()

    # GUARD: if N isn't a multiple of block_dim, the last block has a few
    # extra threads whose index runs off the end of the arrays. They must do
    # nothing. (Foreshadowing Phase 1: these idle threads are a *divergent*
    # branch -- some lanes of the last warp work, some sit out.)
    if i >= len(c):
        return

    # TODO (blank #3): write A[i] + B[i] into C[i].
    raise NotImplementedError


def main():
    N = 1000
    a = [float(x) for x in range(N)]
    b = [float(2 * x) for x in range(N)]
    c = [0.0] * N  # output "global memory", pre-allocated by the host (CPU)

    block_dim = 256                      # threads per block (a common choice)
    grid_dim = math.ceil(N / block_dim)  # enough blocks to cover all N elements

    launch(grid_dim, block_dim, vector_add, a, b, c)

    expected = [a[i] + b[i] for i in range(N)]
    assert c == expected, "vector_add produced a wrong result somewhere!"
    total = grid_dim * block_dim
    print(f"OK: added {N} elements using {grid_dim} blocks x {block_dim} "
          f"threads = {total} threads ({total - N} idle).")


if __name__ == "__main__":
    main()
