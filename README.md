# 🏢 Thermal PDE Acceleration Pipeline for Floorplan Heat Diffusion
> **Numerical Laplace PDE solver & performance engineering across 4,500+ architectural floorplans via CPU Dynamic Parallelism, Numba JIT, Custom CUDA, and Fused CuPy Acceleration.**

[![DTU](https://img.shields.io/badge/DTU-02613_Python_&_HPC-red)](https://kurser.dtu.dk/course/02613)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-NVIDIA%20V100-76B900?style=flat&logo=nvidia&logoColor=white)
[![Numba](https://img.shields.io/badge/Numba-JIT_Compiled-orange.svg)](https://numba.pydata.org/)

---

## 📖 Table of Contents
- [Project Background](#project-background--problem-formulation)
- [Key Achievements & Impact](#key-achievements--impact)
- [Performance Benchmark & Scaling Roadmap](#performance-benchmark--scaling-roadmap)
- [Technical Deep-Dives & Profiling Discoveries](#technical-deep-dives--profiling-discoveries)
- [Physical Insights & Domain Results](#physical-insights--domain-results)
- [How I Used GenAI as a Multiplier](#how-i-used-genai-gemini-as-a-technical-multiplier)
- [Repository Structure](#-repository-structure--directory-layout)

## 🎯 Project Background & Problem Formulation

### 1. The Real-World Domain Problem
This project evaluates an experimental building heating solution—**Wall Heating**—across **4,571 building floorplans** from the [*Modified Swiss Dwellings*](/dtu/projects/02613_2025/data/modified_swiss_dwellings/) dataset. Heating elements are embedded into non-structural interior walls to act as continuous thermal radiators, while structural load-bearing walls remain cold.

The goal is to simulate steady-state room heat distribution and evaluate human thermal comfort ($\ge 18^\circ\text{C}$) and cold-spot risks ($< 15^\circ\text{C}$).

### 2. Mathematical Modeling (2D Thermal PDE)
Let $u(x,y)$ denote the steady-state temperature at coordinate $(x,y)$. The overall heat distribution across a floorplan is modeled by solving 2D **Laplace's Equation**:

$$\frac{\partial^{2}u}{\partial x^{2}} + \frac{\partial^{2}u}{\partial y^{2}} = 0$$

Subject to strict **Dirichlet Boundary Conditions**:
* **Interior Heated Walls**: $u(x,y) = 25^\circ\text{C}$ (fixed)
* **Load-bearing Walls**: $u(x,y) = 5^\circ\text{C}$ (fixed)
* **Interior Rooms**: Unconstrained, solved via stencil relaxation.

### 3. Numerical Discretization (Jacobi Iterative Solver)
Discretized on a $514 \times 514$ grid, the temperature $u_{i,j}$ at each interior grid point is iteratively updated as the average of its 4 orthogonal neighbors:

$$u_{i,j}^{(k+1)} = \frac{1}{4} \left( u_{i,j-1}^{(k)} + u_{i,j+1}^{(k)} + u_{i-1,j}^{(k)} + u_{i+1,j}^{(k)} \right)$$

Updates repeat until maximum iteration or convergence ($\Delta = \max|u^{(k+1)} - u^{(k)}| < \text{atol}$).

---

## 📌 Key Achievements & Impact

* **🚀 24.3x Peak Speedup**: Scaled processing speed **from ~13.9 hours down to ~24 minutes** for all 4,571 floorplans.
* **🔍 Bottleneck Profiling**: Identified that **88.7% of CUDA API duration** was lost to allocation overhead (`cudaMalloc`) via NVIDIA Nsys. Refactored dynamic checks, reducing launch calls by **98.4%**.
* **🤖 AI-Augmented Workflow**: Leveraged Gemini (LLM) as an engineering pair-programmer to profile memory bottlenecks, restructure memory alignment (row-major C-loops), and debug CUDA thread bounds.
* **📊 Thermal Findings**: Simulated 4,500+ buildings—revealing that only **17.6%** reach the $18^\circ\text{C}$ threshold over $>50\%$ of their room area with wall heating alone.

---

## ⚡ Performance Benchmark & Scaling Roadmap

Extrapolated from a benchmark subset of 20 floorplans ($514 \times 514$ grids) to the full dataset (4,571 floorplans):

| Optimization Stage | Tech Stack & Optimization Strategy | Time (20 Plans) | Speedup Factor | Full Dataset Est. (4,571 Plans) |
| :--- | :--- | :---: | :---: | :---: |
| **Task 2: Baseline** | Pure NumPy Slicing (Single Core) | 219.0 s| 1.0x | ~13.9 hours |
| **Task 5: Multi-Core CPU** | Static Scheduling (`multiprocessing`, 16 cores) | 125.6 s | 1.7x| ~3.19 hours |
| **Task 6: Multi-Core CPU** | Dynamic Scheduling (`chunksize=1`, 16 cores) | 84.6 s | 2.6x | ~2.15 hours |
| **Task 7: CPU Acceleration** | Numba `@njit` (Cache-friendly C-contiguous loops) | 63.0 s | 3.5x | ~4.00 hours |
| **Task 8: Custom GPU Kernel** | Numba `@cuda.jit` + Pre-allocated Static Arrays | 27.0 s | 8.1x | ~1.70 hours |
| **Task 9: Naive GPU** | High-level CuPy Array Operations| 55.0 s | 4.0x | ~3.49 hours |
| **🎯 Task 10: Fused GPU** | **CuPy `@cp.fuse()` + Pipelined Convergence (1/10 iter)** | **9.0 s** | **24.3x** | **~24 minutes** |

---

## 🛠️ Technical Deep-Dives & Profiling Discoveries

### 1. Dynamic Task Dispatch for Variable Geometries
* **Problem**: Different floorplans require vastly different iteration counts to reach numerical convergence. Static batch allocation caused core idling.
* **Solution**: Implemented dynamic worker pools (`multiprocessing.Pool(chunksize=1)`). Workers process the next available plan immediately upon completion, dropping execution time by **32.7%** compared to static scheduling at 16 cores.

### 2. Cache-Friendly JIT Compilation (`@njit`)
* **Strategy**: Replaced high-level NumPy array slicing with explicit nested loops structured in **row-major order** (outer loop `i` over rows, inner loop `j` over columns).
* **Impact**: Matches NumPy’s memory layout, minimizing CPU L1/L2 cache misses and delivering a **3.48x speedup** over the CPU baseline.

### 3. Profiling-Driven Kernel Fusion & Stream Pipelining
* **Nsys Profiling Discovery**: High-level CuPy code suffered from **2,177 `cuLaunchKernel` calls** and spending 88.7% of CUDA API time in dynamic allocation (`cudaMalloc`) due to checking convergence (`delta < atol`) every iteration.
* **Refactoring**:
  1. Consolidated 4-point stencil averaging and mask operations into a single unified GPU kernel via `@cp.fuse()`.
  2. Modified the host loop to evaluate convergence **only once every 10 iterations**, dropping launch calls from 2,177 to 35 (**98.4% reduction**).

---

## 📊 Physical Insights & Domain Results

* **Mean Temperature Distribution**: Normally distributed around **$14.69^\circ\text{C}$** ($\sigma \approx 6.80^\circ\text{C}$).
* **Comfort Threshold ($\ge 18^\circ\text{C}$)**: Only **17.6% (804 buildings)** satisfy the comfort threshold over $>50\%$ of their area.
* **Cold Deficit ($< 15^\circ\text{C}$)**: **54.1% (2,471 buildings)** remain below $15^\circ\text{C}$ over most of their layout, proving wall heating alone is structurally insufficient without supplementary exterior insulation.

---

## 💡 How I Used GenAI (Gemini) as a Technical Multiplier

Coming from a mathematics/statistics background, I strategically leveraged **Gemini (GenAI)** as an engineering pair-programmer:
1. **Profiling Log Analysis**: Converted raw `nsys` profiler outputs into explicit refactoring targets, uncovering asynchronous queue locks and dynamic dynamic allocations (`cudaMalloc`).
2. **CUDA Boundary Handling**: Restructured 2D thread index bounds (`cuda.grid(2)`) to eliminate memory segmentation faults during GPU stencil calls.
3. **Pipelining**: Designed async CPU-GPU execution pipelines to eliminate redundant PCIe data transfers

## 📁 Repository Structure & Directory Layout
```text
thermal-hpc-acceleration/
├── README.md                      # Project documentation and performance report
├── requirements.txt               # Dependency specs (NumPy, Numba, CuPy, etc.)
│
├── data/                          # Floorplan input datasets
│   ├── building_ids.txt           # List of 4,571 building IDs
│   └── sample_floorplans/         # Sample domain (*_domain.npy) & mask (*_interior.npy)
│
├── src/                           # Core Source Code Directory
│   ├── baseline/
│   │   └── simulate.py            # Task 2: Single-core NumPy reference implementation
│   │
│   ├── cpu_optimization/
│   │   ├── static_parallel.py     # Task 5: Multi-core CPU (Static scheduling)
│   │   ├── dynamic_parallel.py    # Task 6: Multi-core CPU (Dynamic scheduling, chunksize=1)
│   │   └── numba_jit.py           # Task 7: Numba @njit cache-friendly C-loop solver
│   │
│   └── gpu_acceleration/
│       ├── custom_cuda.py         # Task 8: Numba @cuda.jit custom 2D grid kernel
│       ├── naive_cupy.py          # Task 9: High-level CuPy array implementation
│       └── fused_cupy.py          # Task 10: Fused kernel (@cp.fuse) + pipelined convergence
│
├── profiling/                     # Performance Engineering & Profiling Artifacts
│   ├── kernprof_analysis/        # CPU line-by-line profiling logs
│   └── nsys_reports/              # NVIDIA Nsys GPU trace logs & CUDA API bottleneck data
│
└── hpc_scripts/                   # LSF 10 Cluster Batch Submission Scripts (DTU DCC)
    ├── submit_cpu_bench.sh        # LSF job script for CPU multi-core scaling
    └── submit_gpu_v100.sh         # LSF job script for NVIDIA V100 GPU tasks
```
