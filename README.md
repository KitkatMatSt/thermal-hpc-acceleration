# Thermal PDE Acceleration Pipeline for Floorplan Heat Diffusion

[![DTU](https://img.shields.io/badge/DTU-02613_Python_&_HPC-red)](https://kurser.dtu.dk/course/02613)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-NVIDIA%20V100-76B900?style=flat&logo=nvidia&logoColor=white)
[![Numba](https://img.shields.io/badge/Numba-JIT_Compiled-orange.svg)](https://numba.pydata.org/)

> Numerical 2D Laplace PDE solver and high-performance computing pipeline simulating heat diffusion across 4,500+ architectural floorplans. Engineered from a baseline Python stencil up to dynamic multi-core dispatch, Numba JIT, custom CUDA kernels, and fused CuPy operations.

*Developed for **DTU Course 02613: Python and High-Performance Computing** at the Technical University of Denmark (DTU). Baseline problem formulation and the Modified Swiss Dwellings dataset are credited to the course faculty.*


---

## Problem Context: Wall Heating Simulation

This project models an experimental building heating strategy—**Wall Heating**—evaluated across 4,571 floorplans from the *Modified Swiss Dwellings* dataset. 

* **The Goal:** Simulate 2D steady-state room heat diffusion where interior non-structural walls serve as active radiators ($25^\circ\text{C}$) while exterior load-bearing walls act as heat sinks ($5^\circ\text{C}$).
* **Key Metric:** Determine whether radiant wall heating alone can maintain human thermal comfort ($\ge 18^\circ\text{C}$) across diverse room layouts without auxiliary baseboard or air heating.

---

## Key Achievements & Technical Highlights

* **24.3x End-to-End Speedup:** Reduced total runtime across 4,571 floorplans from **~13.9 hours down to ~34 minutes**.
* **Profiling-Driven Optimization:** Discovered via NVIDIA Nsys that **88.7% of CUDA API time** was spent on dynamic allocation (`cudaMalloc`). Pipelining convergence checks reduced kernel launch calls by **98.4%**.
* **Workload-Aware Batching:** Divided the 4,571 floorplans into batch jobs to prevent HPC cluster timeouts and then merged into complete simulation metrics.
* **Domain Findings:** Simulated 4,500+ buildings, demonstrating that wall heating alone is structurally insufficient—only **17.6%** of floorplans meet thermal comfort ($\ge 18^\circ\text{C}$) over $>50\%$ of room area.

---

## Performance Benchmark & Scaling Roadmap

*Benchmark timings measured across 20 representative floorplans (514 X 514 grids) and extrapolated to the full dataset (4,571 floorplans):*

| Optimization Stage | Tech Stack & Optimization Strategy | Time (20 Plans) | Speedup Factor | Full Dataset Est. (4,571 Plans) |
| :--- | :--- | :---: | :---: | :---: |
| **CPU Baseline** | Pure NumPy Slicing (Single Core) | 219.0 s | 1.0x | ~13.9 hours |
| **CPU Static Multi-Core** | Static Scheduling (`multiprocessing`, 16 cores) | 125.6 s | 1.7x | ~3.19 hours |
| **CPU Dynamic Multi-Core** | Dynamic Scheduling (`chunksize=1`, 16 cores) | 84.6 s | 2.6x | ~2.15 hours |
| **CPU JIT** | Numba `@njit` (Cache-friendly C-contiguous loops) | 63.0 s | 3.5x | ~4.00 hours |
| **GPU Array Baseline** | High-level CuPy Array Operations | 55.0 s | 4.0x | ~3.49 hours |
| **GPU Custom Kernel** | Custom Numba `@cuda.jit` Kernel | 27.0 s | 8.1x | ~1.70 hours |
| **GPU Fused Kernel (Final)** | **CuPy `@cp.fuse()` + Pipelined Convergence (Every 10 iter)** | **9.0 s** | **24.3x** | **~34 minutes** |

---

## Technical Deep-Dives & Profiling Discoveries

### 1. Dynamic Task Dispatch for Variable Geometries
* **Problem**: Different floorplans require vastly different iteration counts to reach numerical convergence. Static batch allocation caused CPU core idling.
* **Solution**: Implemented dynamic worker pools (`multiprocessing.Pool(chunksize=1)`). Workers process the next available plan immediately upon completion, dropping execution time by **32.7%** compared to static scheduling at 16 cores.

### 2. Cache-Friendly JIT Compilation (`@njit`)
* **Strategy**: Replaced high-level NumPy array slicing with explicit nested loops structured in **row-major order** (outer loop `i` over rows, inner loop `j` over columns).
* **Impact**: Matches C-contiguous memory layout, minimizing CPU L1/L2 cache misses and delivering a **3.48x speedup** over the CPU baseline.

### 3. Profiling-Driven Kernel Fusion & Stream Pipelining
* **Nsys Profiling Discovery**: High-level CuPy code suffered from **2,177 `cuLaunchKernel` calls** and spent 88.7% of CUDA API time in dynamic allocation (`cudaMalloc`) due to checking convergence (`delta < atol`) every iteration.
* **Refactoring**:
  1. Consolidated 4-point stencil averaging and mask operations into a single unified GPU kernel via `@cp.fuse()`.
  2. Modified the host loop to evaluate convergence **only once every 10 iterations**, dropping launch calls from 2,177 to 35 (**98.4% reduction**).

---

## Quickstart & Reproducibility

### 1. Environment Setup
```bash
git clone https://github.com/KitkatMatSt/thermal-hpc-acceleration.git
cd thermal-hpc-acceleration
pip install -r requirements.txt
```

### 2. Run Single Simulation Benchmark
```bash
# Run the fused CuPy GPU solver on local sample data
python src/gpu_acceleration/simu_cupy_fused_mask.py --input data/sample_floorplans/

# Run the dynamic multi-core CPU solver
python src/cpu_optimization/simu_para_dynamic.py --cores 16 --chunksize 1
```

### 3. HPC Batch Submission
```bash
# Submit full GPU acceleration job on DTU DCC cluster
bsub < hpc_scripts/sub_cupy_fused_mask.sh
```

---

## Physical Insights & Domain Results

Simulating steady-state thermal diffusion across the Modified Swiss Dwellings corpus ($514 \times 514$ grids) revealed significant thermodynamic trade-offs:

<p align="center">
  <img src="docs/images/task3_heatmaps.jpg" width="60%" alt="Steady-State Thermal Equilibrium Solutions">
  <br>
  <em>Steady-state temperature distribution solved via 2D Jacobi relaxation across representative floorplans.</em>
</p>

* **Temperature Distribution**: The mean room temperature across all 4,571 buildings follows a bell-shaped distribution centered around **$14.69^\circ\text{C}$** ($\sigma \approx 6.80^\circ\text{C}$).
* **Thermal Comfort Deficit ($\ge 18^\circ\text{C}$)**: Only **17.6% (804 buildings)** satisfy the comfort threshold over > 50% of the room area.
* **Cold Deficit ($< 15^\circ\text{C}$)**: **54.1% (2,471 buildings)** remain below $15^\circ\text{C}$, demonstrating that wall heating alone is structurally insufficient without supplementary insulation.

---

## Mathematical Formulation & Background

### 1. Governing Equation

The 2D steady-state heat diffusion is modeled by Laplace's Equation:

$$\frac{\partial^{2}u}{\partial x^{2}} + \frac{\partial^{2}u}{\partial y^{2}} = 0$$

### 2. Dirichlet Boundary Conditions

- Interior Heated Walls: $u(x,y) = 25^\circ\text{C}$ (fixed)
- Load-bearing Walls: $u(x,y) = 5^\circ\text{C}$ (fixed)
- Interior Rooms: Unconstrained, solved via stencil relaxation.

### 3. Jacobi method

Discretized on a uniform grid, the temperature $u_{i,j}$ is updated iteratively via 4-point central differences:

$$u_{i,j}^{(k+1)} = \frac{1}{4} \left( u_{i,j-1}^{(k)} + u_{i,j+1}^{(k)} + u_{i-1,j}^{(k)} + u_{i+1,j}^{(k)} \right)$$

Iterations continue until $\max|u^{(k+1)} - u^{(k)}| < \text{atol}$ or maximum iterations are exceeded.

---

## Engineering Workflow & Profiling Methodology

* **Profiling-First Approach:** Instead of guessing where the code was slow, I used `line_profiler` on the CPU and `NVIDIA Nsys` on the GPU. This showed exactly where the bottlenecks were—especially the repeated memory allocations on the GPU.
* **AI-Assisted Development:** I used Gemini as a coding partner to help read complex GPU profiling logs, calculate 2D CUDA thread indexing (`cuda.grid(2)`), and debug boundary errors.

---

## Repository Structure

```text
thermal-hpc-acceleration/
├── data/sample_floorplans/     # Lightweight test floorplans for reproduction
├── docs/images/                # Heatmaps and simulation visualization assets
├── hpc_scripts/                # LSF cluster batch submission & profiling scripts
├── src/
│   ├── baseline/               # Pure NumPy single-core reference solver
│   ├── cpu_optimization/      # Multiprocessing & Numba JIT acceleration
│   ├── gpu_acceleration/      # Numba CUDA kernel & fused CuPy implementations
│   └── visualization/         # Floorplan plotting & result aggregation
├── .gitignore
├── requirements.txt            # Environment and package dependencies
└── README.md
