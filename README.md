# CSC 746 F25 – Coding Project #5  
## GPU Stencil Operations with CUDA and OpenMP Offload  
### Sobel Filter Implementations

---

### Overview

This project implements a **Sobel edge detection filter** in three different parallel computing paradigms:

1. **sobel_cpu** – CPU-only implementation using OpenMP parallelism  
2. **sobel_gpu** – GPU implementation using CUDA  
3. **sobel_cpu_omp_offload** – GPU implementation using OpenMP device offload  

All programs process the same input dataset:
data/zebra-gray-int8-4x.dat  (7112 × 5146 pixels)
and generate their respective output files in the `data/` directory.

---

### 📁 Directory Structure
sobel-harness-instructional/
├── CMakeLists.txt
├── src/
│   ├── sobel_cpu.cpp
│   ├── sobel_gpu.cu
│   ├── sobel_cpu_omp_offload.cpp
├── data/
│   ├── zebra-gray-int8-4x.dat
│   ├── processed-raw-int8-4x-cpu.dat
│   ├── processed-raw-int8-4x-gpu.dat
│   └── processed-raw-int8-4x-offload.dat
├── scripts/
│   ├── heatmap_plot_hw5.py
│   
└── README.md
---

###  Build Instructions (Perlmutter Environment)

Run these commands **on Perlmutter** before building:

```bash
module load PrgEnv-nvidia
export CC=cc
export CXX=CC

Then build the project:
mkdir build
cd build
cmake ../ 
make

This will produce three executables in the build/ directory:
	•	sobel_cpu
	•	sobel_gpu
	•	sobel_cpu_omp_offload

Execution Instructions

Part 1: CPU (OpenMP Parallel)
Executable: sobel_cpu
Input control: OMP_NUM_THREADS environment variable.
Example run:
export OMP_NUM_THREADS=8
./build/sobel_cpu
Performance Study:
Run with OMP_NUM_THREADS = 1, 2, 4, 8, 16

Output:
data/processed-raw-int8-4x-cpu.dat

Part 2: GPU (CUDA)
Executable: sobel_gpu
Usage:
./build/sobel_gpu <numBlocks> <numThreadsPerBlock>
Example:
./build/sobel_gpu 64 256
Performance Study:
Test combinations:
	•	Threads per block = [32, 64, 128, 256, 512, 1024]
	•	Number of blocks = [1, 4, 16, 64, 256, 1024, 4096]

Output:
data/processed-raw-int8-4x-gpu.dat

⸻

 Part 3: GPU (OpenMP Device Offload)
Executable: sobel_cpu_omp_offload

Example run:
./build/sobel_cpu_omp_offload
Output:
data/processed-raw-int8-4x-offload.dat

Performance Measurement (Using NVIDIA Nsight Compute)

Metric Name
Meaning
gpu__time_duration.avg
GPU kernel runtime (ms)
sm__warps_active.avg.pct_of_peak_sustained_active
Achieved Occupancy (%)
dram__throughput.avg.pct_of_peak_sustained_elapsed
% of peak sustained memory bandwidth

Example Command (CUDA):
ncu --set basic \
--metrics gpu__time_duration.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active \
./build/sobel_gpu 64 256

Example Command (OpenMP Offload):
ncu --set basic \
--metrics gpu__time_duration.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active \
./build/sobel_cpu_omp_offload

⚠️ If you see
Profiling failed because a driver resource was unavailable,
run:
dcgmi profile --pause
then re-run the ncu command.

Verification of Image Output
Verify output visually using the provided Python script:
module load python

# Input image
python scripts/imshow.py data/zebra-gray-int8-4x.dat 7112 5146

# CPU result
python scripts/imshow.py data/processed-raw-int8-4x-cpu.dat 7112 5146

# CUDA result
python scripts/imshow.py data/processed-raw-int8-4x-gpu.dat 7112 5146

# OpenMP offload result
python scripts/imshow.py data/processed-raw-int8-4x-offload.dat 7112 5146

GPU Node Usage Policy (NERSC Perlmutter)
Rule 1: Compile & Plot on CPU Nodes
salloc -N 1 -C cpu -t 10:00 -q interactive -A m3930
# (run cmake, make, python scripts)
exit
Rule 2: Run GPU Workloads on GPU Nodes
salloc -N 1 -C gpu -G 1 -t 10:00 -q interactive -A m3930
./build/sobel_gpu 64 256
exit
Rule 3: Avoid Zombie Jobs
	•	Always exit interactive shells manually with exit
	•	Check running jobs:
    squeue -u $USER
    Kill stray jobs:
    scancel <JOBID>

Author: Guiran Liu
Course: CSC 746 – High Performance Computing (Fall 2025)
Instructor: Prof. Bethel
Due Date: November 3, 2025, 23:59 PST
System Tested: NERSC Perlmutter (A100 GPU node)