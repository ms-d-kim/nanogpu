# toy-gpu — core curriculum

**Mission:** build a working GPU *architecture* simulator in Python — not silicon, not HDL —
then go to the metal (real CUDA C on a GPU) and the landscape (other ASICs). Goal: GPU
execution and memory stop being alien, and you can reason about the hardware, the kernels,
and the chip choice that best serve a given inference workload.

## How it's built: one self-similar codebase

Not three projects — one codebase that grows **outward**: `Thread` → `SM` → `GPU` → `Node`
→ `Cluster`. Every level is the same pattern — *compute units + a fabric + a cost model* —
and every level speaks one shared interface:

> given a workload, return **(time, bytes moved, bottleneck)**.

Because each level's *output* is the next level's *input*, the cluster is literally the GPU
sim composed with itself one zoom-out higher. Two disciplines keep it a learning project,
not a sinkhole:

- **Cost-model altitude, not cycle-accuracy.** We count bytes / FLOPs / latencies and predict
  the bottleneck. Nanosecond fidelity is a multi-year research project — out of scope.
- **A payoff per layer.** Every phase ends in a runnable experiment + one insight before we
  deepen. Build outward; never construct the whole edifice before the first "aha."

## What you'll be able to do when it's done

Act I (the GPU):
- Read any kernel and call it **memory-bound vs compute-bound** on sight (the roofline).
- Explain why **decode is memory-bound and prefill is compute-bound** — and why that drives **disaggregation (Dynamo)**.
- Explain why **batching and occupancy** raise throughput (latency hiding), not magic.
- Say precisely what **PagedAttention, continuous batching, prefix caching** optimize, and why.
- Pick the right GPU for a workload: **bandwidth vs FLOPs vs HBM capacity** (H100 / H200 / B200).
- Explain **kernel-launch overhead** and why **CUDA graphs** matter for agentic, many-small-op workloads.

Act II (metal & the landscape):
- **Write and review CUDA C kernels** on real GPUs (RunPod), with a reviewer's checklist (coalescing, divergence, occupancy, bank conflicts).
- **Read the Dynamo codebase and land a toy-level contribution** (Python / docs / tests first; *read* the Rust core).
- Write toy **"translation" backends / a mini-compiler** that lowers one op (matmul) to GPU vs TPU vs Groq LPU vs AMD execution + cost models.
- Explain **why the ASIC zoo exists** — map the design space (SIMT vs systolic vs static dataflow; HBM+caches vs all-SRAM; hardware-dynamic vs compiler-static scheduling) and predict which workloads each chip wins.

Act III (the cluster):
- Reason about the full **interconnect hierarchy** (HBM → NVLink/NVSwitch → PCIe/C2C → RDMA/InfiniBand → storage) as one bandwidth/latency ladder.
- Match a **parallelism strategy to the right interconnect** (tensor-parallel all-reduce → NVLink; pipeline/expert → scale-out; data-parallel → cheapest links).
- Explain **disaggregated serving** end to end: why the KV cache moves over NVLink or RDMA, and how KV-aware routing (NIXL, Dynamo) minimizes it.
- Name what **NCCL, RDMA / GPUDirect, DPUs (BlueField), and the host CPU** each contribute — and where the cluster bottleneck actually is.

## Act I — the GPU (Python simulator)

Each phase = read the PMPP chapter, build the core, run the experiment.

| Phase | You code (the core idea) | I scaffold (the plumbing) | PMPP |
|---|---|---|---|
| 0 · kernel launcher | global-thread-id math, the `launch` loop, the vector-add body | `Thread` class, runner + self-test | 1–2 |
| 1 · warps & divergence | warp grouping, lockstep with an active-lane mask, a divergence counter | warp stub, a divergent kernel, a wasted-cycle meter | 4 |
| 2 · memory & tiling | naive matmul, then tiled matmul (shared-mem), the byte/FLOP tally | memory model with cost hooks, matmul correctness test | 3, 5 |
| spine · roofline | arithmetic intensity, ridge point, bottleneck verdict, knob sweep | `Arch` knobs (SMs / bandwidth / FLOPs), table + plot printer | 5–6 |
| 3 · scheduling | warp scheduler (switch on stall), occupancy from resource limits | SM-state stub, a memory-stall workload | 4, 6 |
| 4 · many SMs | block-to-SM distribution, wave / tail-effect accounting | GPU-of-SMs container, scaling harness | 4 |
| 5 · orchestration | stream/queue model, copy-compute overlap, launch-overhead model | host event-loop stub, timeline printer | 20 |
| capstone · transformer | attention (QKᵀ, softmax-as-reduction, ·V), MLP, KV cache, prefill vs decode | reference numpy forward (correctness), prefill/decode driver | 16 + papers |

## Act II — metal & the landscape (after the capstone; reuses Act I's machinery)

| Track | You build / do | I scaffold | Goals |
|---|---|---|---|
| CUDA C on RunPod | write real kernels: vector add → tiled matmul → (stretch) fused attention; run & profile vs your Python sim | host boilerplate (`cudaMalloc`/`cudaMemcpy`/launch), `nvcc` + Makefile, a just-enough-C primer per idiom | 2, 5 |
| Dynamo: read → contribute | map the architecture (frontend / router / worker, disaggregation, KV routing), trace one request, land a small PR (Python / docs / tests) | a guided repo tour, a just-enough-Rust *reading* primer | 1 |
| The backend zoo (mini-compiler) | one IR + pluggable backends (GPU / TPU / Groq LPU / AMD), each with its own execution + cost model; run the same workload through each | the IR + backend interface skeleton, reference cost numbers, the comparison harness | 3, 4 |

**Critical path:** Act I is the spine — finish it. Interleave real CUDA C from Phase 0–1 (hands
on metal early). Skim Dynamo's architecture any time. The backend zoo waits for the capstone.

## Act III — the cluster (scale-up & scale-out)

Performance at datacenter scale is the roofline again, one level up: compute vs
*communication* bandwidth at each tier of the interconnect hierarchy, plus overlapping the
two. The Act I "GPU-of-SMs" becomes a "cluster-of-GPUs"; we model it the same way.

| Track | You build / do | I scaffold | Learn |
|---|---|---|---|
| interconnect + collectives | a topology graph (GPUs/nodes, per-link bw+latency); implement ring all-reduce / all-gather / all-to-all cost models | the topology + link skeleton, a comms harness | NVLink/NVSwitch, PCIe/C2C, RDMA, NCCL |
| parallelism cost models | shard a transformer layer with TP/PP/EP across simulated GPUs; compute comm-fraction; find where the network bottlenecks | the sharding scaffold, reference comm volumes | TP/PP/EP/DP, Megatron, MoE all-to-all |
| disaggregation + routing | model prefill→decode KV-cache transfer over NVLink vs RDMA; run a KV-aware-routing experiment | the worker/transfer skeleton, a request stream | NIXL, GPUDirect RDMA, Dynamo |

Payoff experiment — the **communication roofline**: at what model size / batch / parallelism
does the *network*, not the GPU, set throughput? That is the cluster-design decision, and
it's Dynamo's whole reason to exist.

## The build/coach contract — what "graduated scaffolding" means here

- **I give you:** docstring'd skeletons, a self-checking test per phase, example workloads, the experiment runners, and *all* host/language boilerplate. You never start from a blank file.
- **You write:** the kernels, the index math, the schedulers, the cost models, the IR lowerings — the ~5–30 lines per phase where the idea lives. I never hand you those.
- **The rule:** if a line teaches you how the hardware *thinks*, it's yours; if it's boilerplate, it's mine. Take a real swing first; I review, nudge, then reveal.

## Languages & the just-enough on-ramp

**The principle: Python for the model, C for the metal.** The simulator is a *story about* the
hardware, built for clarity — so it's Python. Real kernels *are* the hardware — so they're
CUDA C. Read C/C++ comfortably, write a little CUDA C, don't become a C++ engineer.
Using each language only where it's the right altitude is deliberate, not a compromise.

- **Python** — primary (Act I). Plain, decoded idiom-by-idiom.
- **CUDA C — you now write it.** You've never done C, and that's fine: kernels need only a *small C subset* (scalar types, pointers, array indexing, `for` loops, functions). No C++, no OOP, no templates. I provide the host boilerplate and teach each new C idiom the moment it appears. `nvcc` on RunPod.
- **Rust — to read, barely to write.** Dynamo's core is Rust (~62% of the repo) with a large Python surface and some Go. Contributing "at a toy level" means Python / docs / tests first, *reading* Rust to follow the core. A just-enough-Rust primer (ownership, `Result`/`Option`, traits at a glance) when we get there — no crate-writing.
- **Not in scope:** C++ beyond reading the odd kernel; Verilog / HDL.

## Products & concepts this maps onto

- **NVIDIA inference stack:** Dynamo (disaggregation + orchestration), TensorRT-LLM (kernel fusion, CUDA graphs, quantization), Triton / NIM (serving).
- **OSS engines:** vLLM (PagedAttention, continuous batching), SGLang (RadixAttention / prefix caching).
- **Compilers / IRs** (backend-zoo grounding): XLA / StableHLO (TPU), OpenAI Triton (GPU), MLIR, Mojo, ROCm / HIP (AMD).
- **The ASIC zoo:** NVIDIA GPU (SIMT + tensor cores, HBM), Google TPU (systolic MXU, VLIW, compiler-scheduled), Groq LPU (deterministic static dataflow, all-SRAM), AMD Instinct (CDNA, wavefront-64, Matrix Cores, ROCm).
- **Agents:** many small launches, CPU↔GPU orchestration, KV-cache reuse — why launch overhead and scheduling dominate.
- **Cluster interconnect & networking:** NVLink / NVSwitch (scale-up, NVL72), PCIe / NVLink-C2C (Grace), InfiniBand (Quantum) / RoCE (Spectrum-X), RDMA + GPUDirect, BlueField DPUs, ConnectX NICs; the comms/transfer libraries NCCL and NIXL.
- **Transferable mental models:** SIMT, warp divergence, memory coalescing, occupancy, arithmetic intensity / roofline, KV cache, systolic arrays, static vs dynamic scheduling, SRAM-vs-HBM, compiler lowering / IR, scale-up vs scale-out, the bandwidth/latency hierarchy, collective communication (all-reduce / all-to-all), communication-vs-compute overlap.
