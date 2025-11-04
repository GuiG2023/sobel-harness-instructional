# CSC 746 F25 CP5 - Sobel Filter Implementations

This submission is for Coding Project #5. It includes three implementations of a Sobel edge detection filter:

* **`sobel_cpu`**: CPU-only C++ with OpenMP parallelism.
* **`sobel_gpu`**: GPU implementation using CUDA.
* **`sobel_cpu_omp_offload`**: GPU implementation using C++ with OpenMP Device Offload.

All implementations read from the hard-coded data file `data/zebra-gray-int8-4x.dat` (7112x5146) and write their output to separate files in the `data/` directory.

# Build Instructions

These instructions must be run on Perlmutter.

1.  Set up the required environment:

    ```bash
    module load PrgEnv-nvidia
    export CC=cc
    export CXX=CC
    ```

2.  From the project's root directory, create a build directory and use CMake/make:

    ```bash
    mkdir build
    cd build
    cmake ../ -Wno-dev
    make
    ```

This will create three executables in the `build/` directory: `sobel_cpu`, `sobel_gpu`, and `sobel_cpu_omp_offload`.

# Execution Instructions

All execution should be done on a Perlmutter GPU node (e.g., via an interactive `salloc` session or a batch script).

### 1. Part 1: CPU (OpenMP Parallel)

This program (`sobel_cpu`) uses the `OMP_NUM_THREADS` environment variable to control concurrency, as required by the assignment.

**To run with N threads:**

```bash
# Example for 8 threads
export OMP_NUM_THREADS=8
./build/sobel_cpu
The performance test requires running with OMP_NUM_THREADS set to 1, 2, 4, 8, and 16. The output file is data/processed-raw-int8-4x-cpu.dat.

2. Part 2: GPU (CUDA)
This program (sobel_gpu) takes the number of thread blocks and threads per block as command-line arguments.

Usage: ./build/sobel_gpu <numBlocks> <numThreadsPerBlock>

Example (for 64 blocks and 256 threads):

Bash

./build/sobel_gpu 64 256
The output file is data/processed-raw-int8-4x-gpu.dat.

3. Part 3: GPU (OpenMP Offload)
This program (sobel_cpu_omp_offload) does not require any special runtime arguments.

To run:

Bash

./build/sobel_cpu_omp_offload
The output file is data/processed-raw-int8-4x-cpu.dat.

Performance Data Collection (NCU)
The following commands are used to gather the GPU performance metrics as specified in the assignment.

Note: If ncu fails with a "driver resource was unavailable" error, run dcgmi profile --pause and try the ncu command again.

1. GPU (CUDA)
This command collects all three required metrics (runtime, bandwidth, and occupancy) in one run.

Bash

# Example for 64 blocks, 256 threads
ncu --set basic --metrics gpu__time_duration.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active ./build/sobel_gpu 64 256
(Note: sm__warps_active.avg.pct_of_peak_sustained_active corresponds to the required "Achieved Occupancy %".)

2. GPU (OpenMP Offload)
This command collects the metrics for the single-run OpenMP offload version.

Bash

ncu --set basic --metrics gpu__time_duration.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active ./build/sobel_cpu_omp_offload
Testing and Verifying Computations
You can verify the output images using the provided Python script.

Prerequisites:

Log in to Perlmutter with X-tunneling enabled: ssh -Y user@perlmutter.nersc.gov

Load the Python module: module load python

Example (from the project's root directory):

Bash

# Display the source image
python scripts/imshow.py data/zebra-gray-int8-4x.dat 7112 5146

# Display the result from the CUDA code
python scripts/imshow.py data/processed-raw-int8-4x-gpu.dat 7112 5146

# Display the result from the CPU OpenMP code
python scripts/imshow.py data/processed-raw-int8-4x-cpu.dat 7112 5146

# Display the result from the OpenMP Offload code
python scripts/imshow.py data/processed-raw-int8-4x-cpu.dat 7112 5146