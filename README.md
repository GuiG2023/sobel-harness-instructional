# CSC 746 F25 – Coding Project #5  
## GPU Stencil Operations with CUDA and OpenMP Offload  
### Sobel Filter Implementations

---

### Overview

This project implements a Sobel edge detection filter in three different parallel computing paradigms:

1. **sobel_cpu** – CPU-only implementation using OpenMP parallelism  
2. **sobel_gpu** – GPU implementation using CUDA  
3. **sobel_cpu_omp_offload** – GPU implementation using OpenMP device offload  

All programs process the same input dataset:  
`data/zebra-gray-int8-4x.dat` (7112 × 5146 pixels)  
and generate their respective output files in the `data/` directory.

---

### Directory Structure
```
sobel-harness-instructional/
├── CMakeLists.txt
├── src/
│   ├── sobel_cpu.cpp
│   ├── sobel_gpu.cu
│   ├── sobel_cpu_omp_offload.cpp
│   └── sobel_utils.cpp/.h
│
├── data/
│   ├── zebra-gray-int8-4x.dat
│   ├── processed-raw-int8-4x-cpu.dat
│   ├── processed-raw-int8-4x-gpu.dat
│   ├── processed-raw-int8-4x-offload.dat
│   └── correct_results_4x.dat
│
├── scripts/
│   ├── heatmap_plot_hw5.py
│   ├── imshow.py
│   ├── plot_heatmaps.py
│   ├── run-cp5-cuda-configs-gpu-perlmutter.sh
│   └── skeleton-gpu-batch-script.sh
│
├── results.csv
├── cuda_performance_heatmaps.png
└── README.md
```

---

### Build Instructions (Perlmutter Environment)

Run these commands on NERSC Perlmutter before building:

```bash
module load PrgEnv-nvidia
export CC=cc
export CXX=CC
```

Then build the project:

```bash
mkdir build
cd build
cmake ../
make
```

This will produce three executables in the `build/` directory:
```
sobel_cpu
sobel_gpu
sobel_cpu_omp_offload
```

---

### Execution Instructions

#### Part 1: CPU (OpenMP Parallel)
**Executable:** `sobel_cpu`  
**Control:** `OMP_NUM_THREADS` environment variable  

Example:
```bash
export OMP_NUM_THREADS=8
./build/sobel_cpu
```

Performance Study: run with `OMP_NUM_THREADS = 1, 2, 4, 8, 16`

Output:  
`data/processed-raw-int8-4x-cpu.dat`

---

#### Part 2: GPU (CUDA)
**Executable:** `sobel_gpu`  
**Usage:**
```bash
./build/sobel_gpu <numBlocks> <numThreadsPerBlock>
```

Example:
```bash
./build/sobel_gpu 64 256
```

Performance Study:  
Test combinations:
- Threads per block = [32, 64, 128, 256, 512, 1024]  
- Number of blocks  = [1, 4, 16, 64, 256, 1024, 4096]

Output:  
`data/processed-raw-int8-4x-gpu.dat`

---

#### Part 3: GPU (OpenMP Device Offload)
**Executable:** `sobel_cpu_omp_offload`  

Example:
```bash
./build/sobel_cpu_omp_offload
```

Output:  
`data/processed-raw-int8-4x-offload.dat`

---

### Performance Measurement (Using NVIDIA Nsight Compute)

**Metrics Collected**
| Metric | Description |
|---------|--------------|
| `gpu__time_duration.avg` | GPU kernel runtime (ms) |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | Achieved Occupancy (%) |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | % of peak sustained memory bandwidth |

**Example Command (CUDA):**
```bash
ncu --set basic \
--metrics gpu__time_duration.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active \
./build/sobel_gpu 64 256
```

**Example Command (OpenMP Offload):**
```bash
ncu --set basic \
--metrics gpu__time_duration.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active \
./build/sobel_cpu_omp_offload
```

**Note:**  
If you see:
```
Profiling failed because a driver resource was unavailable.
```
Run:
```bash
dcgmi profile --pause
```
Then re-run the `ncu` command.

---

### Verification of Image Output
You can verify and visualize results using the provided Python script:

```bash
module load python

# Input image
python scripts/imshow.py data/zebra-gray-int8-4x.dat 7112 5146

# CPU result
python scripts/imshow.py data/processed-raw-int8-4x-cpu.dat 7112 5146

# CUDA result
python scripts/imshow.py data/processed-raw-int8-4x-gpu.dat 7112 5146

# OpenMP offload result
python scripts/imshow.py data/processed-raw-int8-4x-offload.dat 7112 5146
```

---

### GPU Node Usage Policy (NERSC Perlmutter)

**Rule 1: Compile & Plot on CPU Nodes**
```bash
salloc -N 1 -C cpu -t 10:00 -q interactive -A m3930
# (run cmake, make, python scripts)
exit
```

**Rule 2: Run GPU Workloads on GPU Nodes**
```bash
salloc -N 1 -C gpu -G 1 -t 10:00 -q interactive -A m3930
./build/sobel_gpu 64 256
exit
```

**Rule 3: Avoid Zombie Jobs**
- Always exit interactive shells manually with `exit`
- Check running jobs:
  ```bash
  squeue -u $USER
  ```
- Kill stray jobs:
  ```bash
  scancel <JOBID>
  ```

---

### Author Information
**Author:** Guiran Liu  
**Course:** CSC 746 – High Performance Computing (Fall 2025)  
**Instructor:** Prof. Wes Bethel  
**Due Date:** November 3, 2025, 23:59 PST  
**System Tested:** NERSC Perlmutter (A100 GPU node)