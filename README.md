# nanogpu

A functional GPU *architecture* simulator in Python — built to develop intuition
for how GPUs execute work and manage memory, aimed at understanding inference
systems (Dynamo, vLLM, SGLang, TRT-LLM), not at writing HDL.

Guiding principle: **every hardware concept we model must pay off as an insight
about a real inference framework.**

## Roadmap

Each phase ends in something runnable plus a real-world payoff.

- **Phase 0 — host/device + a kernel launcher.** What "launching a kernel" means.
- **Phase 1 — SIMT & warps.** Lockstep execution; warp divergence.
- **Phase 2 — memory hierarchy + tiled matmul.** Arithmetic intensity & the roofline
  (why decode is memory-bound, prefill is compute-bound).
- **Phase 3 — scheduling & occupancy.** Warp scheduler hides memory latency → why batching wins.
- **Phase 4 — many SMs.** Distributing blocks; how work scales.
- **Phase 5 — host orchestration.** Streams, launch overhead, CUDA graphs → continuous batching, Dynamo.
- **Capstone — a tiny transformer forward pass** with a KV cache: watch prefill be
  compute-bound and decode be memory-bound.

**Act II — metal & the landscape** (full plan in `CURRICULUM.md`): write real CUDA C
kernels on RunPod GPUs; read and make toy contributions to NVIDIA Dynamo; build a
multi-backend mini-compiler (GPU vs TPU vs Groq LPU vs AMD) to explain the ASIC zoo.

**Act III — the cluster** (full plan in `CURRICULUM.md`): model the interconnect hierarchy
(NVLink/NVSwitch, PCIe/C2C, RDMA/InfiniBand), NCCL-style collectives, parallelism
(TP/PP/EP), and disaggregated KV-cache transfer (NIXL) — extracting performance at scale.

## Grounded in PMPP

A hands-on companion to *Programming Massively Parallel Processors: A Hands-on
Approach*, 4th ed. (Hwu, Kirk & El Hajj, 2022). Read the chapter, then build the phase.

| Phase | PMPP chapters |
|---|---|
| 0 — host/device + kernel launcher (vector add) | 1 Introduction · 2 Heterogeneous data parallel computing |
| 1 — SIMT & warps, divergence | 4 Compute architecture and scheduling |
| 2 — memory hierarchy + tiled matmul | 3 Multidimensional grids and data · 5 Memory architecture and data locality |
| decision spine — roofline / arithmetic intensity | 5 + 6 Performance considerations |
| 3 — scheduling & occupancy | 4 + 6 |
| 4 — many SMs | 4 (block scheduling, transparent scalability) |
| 5 — host orchestration (streams) | 20 Programming a heterogeneous computing cluster (CUDA streams) |
| capstone — transformer forward + KV cache | 16 Deep learning (grounding only) |

Bonus: 10 Reduction — softmax/attention are reductions under the hood.

**Beyond PMPP (our bridge to inference):** CUDA graphs, attention / KV cache /
PagedAttention, continuous batching, prefill-decode disaggregation. These live in the
CUDA docs and the vLLM / SGLang / TRT-LLM / Dynamo literature, not the book.

## The decision spine (workload → architecture)

The through-line of every phase: the simulator is *configurable and measurable*,
so you can decide which architecture best fits a given workload — the core PM skill.
Three small pieces, grown across the phases:

1. **`Arch` spec** — the knobs you turn: number of SMs, HBM bandwidth, peak FLOPs,
   shared-memory / cache sizes, clock. This is the "architecture" you're choosing.
2. **Cost model** — each kernel tallies FLOPs executed and bytes moved through HBM.
3. **Experiment harness** — sweep a knob (or workload size) and read off the bottleneck.

Out of this falls the **roofline** and a decision rule:
arithmetic intensity = FLOPs ÷ bytes; ridge point = peak FLOPs ÷ bandwidth.
Below the ridge → memory-bound → invest in bandwidth. Above → compute-bound → invest in FLOPs.
This is why prefill (compute-bound) and decode (memory-bound) want different hardware —
the argument behind disaggregation (Dynamo) and continuous batching (vLLM).

## Glossary (the alien words)

Software side (what you write): **grid** → **block** → **thread**.
Hardware side (what runs it): **GPU** → **SM** → **warp** (32 threads in lockstep) → **lane**.
Memory, fast→slow: **registers** → **shared memory** → **L1/L2** → **HBM (global)**.

## Layout

- `nanogpu.py` — the simulator core (the `launch` runtime + `Thread`).
- `phase0_vector_add.py` — the Phase 0 workload + a self-checking runner.
- `tests/` — pytest checks (red until each phase's blanks are filled, then green).
- `CURRICULUM.md` — the full Act I / II / III plan and the build/coach contract.

```
python phase0_vector_add.py   # run the current phase
pytest                        # run the tests
```

Future-phase scaffolds are added as you reach them (graduated scaffolding): I provide the
skeleton + tests, you write the conceptual core. Nothing past Phase 0 is pre-built — that
empty space is intentional.
