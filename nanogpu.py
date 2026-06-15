"""nanogpu -- core runtime of the toy GPU simulator.

This file models the GPU side of the host/device split: it takes a kernel
(a plain Python function describing ONE thread's job) and runs it once per
thread across a grid of blocks.

Phase 0 goal: make `launch` actually run the threads, and teach each thread
how to find its own identity in the global array of work.
"""


class Thread:
    """One thread's identity card.

    The hardware hands every thread two coordinates plus one size:
      - block_idx : which block this thread lives in     (CUDA: blockIdx.x)
      - thread_idx: its position inside that block        (CUDA: threadIdx.x)
      - block_dim : how many threads are in each block    (CUDA: blockDim.x)

    With those three numbers a thread can locate itself in the *global*
    array of all threads launched.
    """

    def __init__(self, block_idx, thread_idx, block_dim):
        self.block_idx = block_idx
        self.thread_idx = thread_idx
        self.block_dim = block_dim

    def global_id(self):
        """The single most important line in all of CUDA.

        Flatten (block_idx, thread_idx) into one global index. Think:
        how many threads came before me in all the earlier blocks, plus
        my own position inside my block?
        """
        # TODO (blank #1): return this thread's global index.
        raise NotImplementedError


def launch(grid_dim, block_dim, kernel, *args):
    """Run `kernel` once per thread. This IS the kernel launch.

    grid_dim : number of blocks in the grid   (how many teams of workers)
    block_dim: threads per block              (workers per team)
    kernel   : a function kernel(thread, *args) -- ONE thread's job
    *args    : the data the kernel needs (e.g. the input/output arrays)

    Total threads launched = grid_dim * block_dim. We run them sequentially
    here -- a real GPU runs warps of them in parallel, but the *result* is
    the same, which is exactly the point of a simulator.
    """
    # TODO (blank #2): for each block (0 .. grid_dim - 1):
    #                     for each thread in it (0 .. block_dim - 1):
    #                         build a Thread, then call kernel(thread, *args)
    raise NotImplementedError
