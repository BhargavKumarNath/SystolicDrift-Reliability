# SystolicDrift

> An LLM numerical reliability and execution analysis framework.

[![Python 3.10+](figures/badges/python.svg)](https://www.python.org/)
[![NumPy 1.24+](figures/badges/numpy.svg)](https://numpy.org/)
[![PyTorch 2.1+](figures/badges/pytorch.svg)](https://pytorch.org/)
[![Model SmolLM2-135M](figures/badges/model.svg)](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
[![Tests 47 passed](figures/badges/tests.svg)](#2-run-unit-tests-47-tests-offline-safe)
[![Experiments 144 runs](figures/badges/experiments.svg)](docs/REPORT.md)
[![License MIT](figures/badges/license.svg)](LICENSE)

## Evolution

SystolicDrift began as a cycle-stepped systolic array simulator investigating numerical drift across hardware dataflows, accumulation precision, and execution schedules.

This repository is the next stage of that work. The original project established the numerical simulation foundation; this successor extends the investigation toward full-model numerical propagation, inference reproducibility (batch invariance), advanced accumulator and microscaling formats, numerical diagnostics, and mitigation and optimization.

- **Original project (`v0.1.0`):** [BhargavKumarNath/SystolicDrift](https://github.com/BhargavKumarNath/SystolicDrift)

---

## Baseline Simulator (`v0.1.0`)

**SystolicDrift** is a cycle-stepped 2D systolic array simulator in Python for isolating how **array geometry** (`8x8` vs. `16x16`), **dataflow** (`weight_stationary` vs. `output_stationary`), **tile loop order** (`M-N-K`, `K-M-N`, `K-N-M`), and **accumulator precision** (`FP32`, `FP16`, `BF16`) change floating-point rounding drift on real transformer layers.

Floating-point addition is non-associative. Even when every hardware schedule consumes bit-identical quantized inputs, the dataflow and loop order change the reduction tree along `K`, producing measurable divergence in narrow accumulators alongside distinct cycle and memory-traffic profiles.

---

## Executive Summary of Results (`SmolLM2-135M`, Layer 15)

Evaluated across **144 controlled runs** (`29.34s` total runtime on an Apple M1 Mac, 8 GB RAM) on real activations and weights extracted from `HuggingFaceTB/SmolLM2-135M` (`(M, K, N) = (16, 256, 256)`, `1,048,576` MACs per matmul):

- **Accumulator precision sets the baseline by 4 to 5 orders of magnitude**: Mean relative L2 error rises from `4.00e-8` (`FP32`) to `9.29e-4` (`FP16`) and `7.59e-3` (`BF16`, peaking at `1.94e-2`). Zero `NaN`/`Inf` values occurred across all 144 runs.
- **Dataflow and loop order create a 2.0x to 2.71x drift spread in `BF16` accumulation**: Sequential reduction across all 256 `K` steps in `OS M-N-K` (`os_seq_k`) yields `0.019448` relative L2 drift on `mlp_down_proj`, whereas chunked `WS 16x16` (`ws_chunked_k16`) drops drift to `0.007175` at a `1.32x` cycle cost (`15,872` vs. `12,032` cycles).
- **Loop order changes preload cycles without changing WS numerics**: On an `8x8` weight-stationary array (`M=16` splits into two `8`-row tiles), `K-N-M` reuses each weight tile across `M`, cutting total cycles from `61,440` (`M-N-K`) to `53,248` with identical numerical output (`0.004561` rel L2 on `attn_q_proj`).
- **Peak absolute errors concentrate in the magnitude tail**: `top1p_max_abs` equals the global `max_abs_error` (`0.076238` on `attn_q_proj`, `OS 16x16 M-N-K`, `BF16` acc), matching the heavy-tailed input activation profile (crest factor `20.17`).

Read the full analysis in [REPORT.md](docs/REPORT.md).

---

## Hardware and Numerical Execution Model

To isolate order-dependent rounding from operand quantization, `A` and `B` are quantized **once** per input dtype (`bf16` or `fp8_e4m3`) before sweeping hardware schedules. Every PE multiply and partial-sum addition executes explicit IEEE round-to-nearest-even (RNE) rounding in `acc_dtype`.

| Schedule | Mapping on `(R, C)` Grid | K-Reduction Order | Preload / Drain Overhead |
| :--- | :--- | :--- | :--- |
| `weight_stationary` (`M-N-K`, `K-M-N`) | Weight tile `(K_t <= R, N_t <= C)` fixed in PEs, activations stream East, psums flow South | Chunked (`K_t = R` per tile, `ws_chunked_kR`) | `R` preload cycles on every `(k, n)` miss |
| `weight_stationary` (`K-N-M`) | Same PE execution, outer `(k, n)` keeps weight tile resident across all `M`-tiles | Chunked (`K_t = R` per tile, `ws_chunked_kR`) | `R` preload cycles once per `(k, n)` block |
| `output_stationary` (`M-N-K`) | Output tile `(M_t <= R, N_t <= C)` fixed in PEs across all `K`-tiles | Sequential over full `K` (`os_seq_k`) | `R` drain cycles once per `(m, n)` tile |
| `output_stationary` (`K-M-N`, `K-N-M`) | Output tile resets per `K`-tile and drains to off-array accumulator buffer | Chunked (`K_t = R` per tile, `os_chunked_kR`) | `R` drain cycles on every `(m, n, k)` tile |

---

## Experimental Workloads

Extracted via forward hooks on `model.layers.15` of `HuggingFaceTB/SmolLM2-135M` (`seq_len = 16`, sliced to `K = 256, N = 256`):

| Workload Layer | Shape `(M, K, N)` | Act RMS | Act Max Abs | Crest Factor (`Max / RMS`) | Weight Std | Ref Output RMS | Ref Output Max Abs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `attn_q_proj` (`self_attn.q_proj`) | `(16, 256, 256)` | `0.3419` | `6.8957` | `20.17` | `0.1846` | `1.0072` | `8.8812` |
| `mlp_down_proj` (`mlp.down_proj`) | `(16, 256, 256)` | `0.1390` | `2.3013` | `16.56` | `0.1816` | `0.3894` | `2.2584` |

---

## Benchmark Results and Metrics

### 1. Drift by Accumulator Precision (All 144 Runs)

Error measured against the quantized-input `FP64` reference (`ref_q = quant(A) @ quant(B)` in `FP64`):

| Accumulator Dtype | Mantissa Bits | Mean Rel L2 Error | Max Rel L2 Error | Relative Drift vs. `FP32` | Non-Finite Count |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `fp32` | `23` (+1 implicit) | `4.00e-8` | `1.49e-7` | `1.0x` | `0` |
| `fp16` | `10` (+1 implicit) | `9.29e-4` | `2.53e-3` | `~23,200x` | `0` |
| `bf16` | `7` (+1 implicit) | `7.59e-3` | `1.94e-2` | `~189,700x` | `0` |

![Numerical drift across accumulator precisions and dataflow schedules](figures/drift_by_precision_and_dataflow.png)

### 2. Representative Schedule Comparison (`BF16` Accumulation)

Selected runs from `results/ablation_details.json` and `results/ablation_summary.csv` showing the accuracy-latency tradeoff:

| Layer | Input | Array | Dataflow & Order | Addition Tag | Rel L2 Error | Top-1% Rel L2 | Max Abs Error | Total Cycles | Wall Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `attn_q_proj` | `bf16` | `8x8` | `WS M-N-K` | `ws_chunked_k8` | `0.004561` | `0.003746` | `0.068220` | `61,440` | `0.50s` |
| `attn_q_proj` | `bf16` | `16x16` | `WS K-M-N` / `OS K-N-M` | `chunked_k16` | `0.005641` | `0.005372` | `0.080651` | `15,872` | `0.14s` |
| `attn_q_proj` | `bf16` | `16x16` | `OS M-N-K` | `os_seq_k` | `0.009072` | `0.005784` | `0.076238` | `12,032` | `0.13s` |
| `attn_q_proj` | `bf16` | `8x8` | `OS M-N-K` | `os_seq_k` | `0.009072` | `0.005784` | `0.076238` | `45,568` | `0.49s` |
| `mlp_down_proj` | `bf16` | `16x16` | `WS K-M-N` / `OS K-N-M` | `chunked_k16` | `0.007175` | `0.005865` | `0.025038` | `15,872` | `0.14s` |
| `mlp_down_proj` | `bf16` | `8x8` | `OS M-N-K` | `os_seq_k` | `0.019448` | `0.017391` | `0.058732` | `45,568` | `0.49s` |
| `attn_q_proj` | `fp8_e4m3` | `16x16` | `WS K-M-N` / `OS K-N-M` | `chunked_k16` | `0.005245` | `0.004805` | `0.086342` | `15,872` | `0.14s` |
| `attn_q_proj` | `fp8_e4m3` | `8x8` | `OS M-N-K` | `os_seq_k` | `0.009159` | `0.004951` | `0.076210` | `45,568` | `0.49s` |
| `mlp_down_proj` | `fp8_e4m3` | `16x16` | `WS K-M-N` / `OS K-N-M` | `chunked_k16` | `0.007262` | `0.006795` | `0.023582` | `15,872` | `0.14s` |
| `mlp_down_proj` | `fp8_e4m3` | `8x8` | `OS M-N-K` | `os_seq_k` | `0.018960` | `0.015762` | `0.048489` | `45,568` | `0.49s` |

Full per-run rows (including `cosine_sim`, `rmse`, `top1p_*`, `e2e_*`, `utilization`, `weight_loads`, and `output_drains`) are stored in `results/ablation_summary.csv` and `results/ablation_details.json`.

![Accuracy vs. latency tradeoff across schedules](figures/drift_vs_cycles_pareto.png)

### 3. Tail Outliers and End-to-End Precision Budget

![Tail outlier error and input quantization breakdown](figures/tail_outlier_and_quantization_breakdown.png)

Key engineering takeaways from the plots:

1. **Why `FP16` beats `BF16` here (`9.29e-4` vs `7.59e-3`)**: `FP16` has 3 extra mantissa bits (`10` vs `7`), giving ~8x lower rounding noise when partial sums stay below `65,504` (`SmolLM2-135M` layer-15 max output is `8.88`).
2. **Why `OS M-N-K` drifts up to 2.71x worse**: Continuous `K=256` accumulation in a single PE register accumulates rounding error linearly (`os_seq_k`), while chunked reduction (`k=8` or `k=16`) bounds partial-sum growth and absorption.
3. **Where input quantization vs. accumulation dominates**: Under `FP8 E4M3` inputs, end-to-end error vs. unquantized `FP64` plateaus near `~0.038` for `FP32` and `FP16` accumulators (input quantization dominates). Switching to a `BF16` accumulator adds `0.01896` pure accumulation drift on `mlp_down_proj`.
4. **Pareto winner**: `16x16` `WS` (`15,872` cycles) avoids the `2.0x` to `2.71x` sequential `BF16` drift penalty of `OS 16x16 M-N-K` (`12,032` cycles, `34.04%` PE utilization) for only a `1.32x` cycle increase.

---

## Programmatic Usage

```python
from systolic_drift import (
    SystolicArraySimulator,
    SystolicConfig,
    compute_fp64_reference,
    evaluate_simulation,
    load_workloads,
)

workloads = load_workloads(source="auto", seq_len=16, max_k=256, max_n=256)
wl = workloads["attn_q_proj"]

cfg = SystolicConfig(
    array_rows=16,
    array_cols=16,
    dataflow="weight_stationary",
    loop_order="K-N-M",
    input_dtype="bf16",
    acc_dtype="bf16",
)
result = SystolicArraySimulator(cfg).matmul(wl.activations, wl.weights)
ref = compute_fp64_reference(wl.activations, wl.weights, input_dtype="bf16")
report = evaluate_simulation(result, ref)

print(f"rel_l2={report.rel_l2_error:.6f}, top1p_max={report.top1p_max_abs:.6f}, cycles={report.total_cycles}")
```

---

## Repository Structure

```text
systolic-drift/
├── README.md                 # Architecture overview, metrics, figures, and quickstart
├── docs/REPORT.md            # Full technical write-up, methodology, and limitations
├── pyproject.toml            # Dependencies (numpy, torch, transformers, pytest, matplotlib)
├── src/systolic_drift/
│   ├── dtypes.py             # IEEE RNE casting & MAC rounding (FP64/FP32/BF16/FP16/FP8)
│   ├── simulator.py          # Cycle-stepped 2D PE grid (weight-stationary & output-stationary)
│   ├── tiling.py             # Tile slicing, edge padding, and M/N/K traversal schedules
│   ├── tensors.py            # HuggingFace tensor extractor + deterministic synthetic fallback
│   └── metrics.py            # Cosine sim, rel L2, max abs, top-1% tail error, cycle stats
├── scripts/
│   ├── run_experiments.py    # Executes 144-run ablation sweep -> CSV/JSON
│   └── make_charts.py        # Generates comparison charts -> figures/
├── tests/
│   ├── test_simulator.py     # Rounding, single-tile equivalence, edge shapes, cancellation probes
│   └── test_metrics.py       # Analytical metric checks and non-finite edge cases
├── results/                  # Saved ablation_summary.csv and ablation_details.json
└── figures/                  # Generated PNG charts and README SVG badges
```

---

## Quickstart

### 1. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Run Unit Tests (`47` tests, offline safe)

Covers IEEE RNE bit-patterns, single-tile WS/OS bitwise equivalence, prime/rectangular shapes, deterministic cancellation probes, dataflow reuse invariants, and metric edge cases:

```bash
pytest -v
```

### 3. Run Experiments and Generate Figures

```bash
python scripts/run_experiments.py
python scripts/make_charts.py
```

For completely offline execution without downloading `HuggingFaceTB/SmolLM2-135M`, use `--source synthetic`:

```bash
python scripts/run_experiments.py --source synthetic
python scripts/make_charts.py
```
