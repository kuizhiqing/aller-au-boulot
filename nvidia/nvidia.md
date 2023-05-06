# nvidia

| NAME | WHAT | SOURCE | HEADER | --- |
| --- | --- | --- | --- | --- |
| CUDA | Compute Unified Device Architecture | N | cuda.h | libcuda.so |
| cuBLAS | Basic Linear Algebra Subroutine | N | cublas_v2.h | cublas.so |
| CUTLASS | CUDA Templates for Linear Algebra Subroutines | Y | cutlass/cutlass.h | |
| CUB | Cooperative primitives for CUDA C++ | Y | cub/cub.cuh | | 
| Thrust | The C++ Parallel Algorithms Library | Y | thrust/.h | |
| cuSOLVER | LAPACK | N | cusolverDn.h | libcusolver.so |

## CUDA

一般 CUDA 指 CUDA Toolkit

```cpp
#include "cuda.h"
#include "cuda_runtime.h"
#include <iostream>

int main() {
  int rt_ver, dr_ver;
  auto err1 = cudaRuntimeGetVersion(&rt_ver);
  if (err1 != cudaSuccess) {
    std::cerr << cudaGetErrorString(err1) << std::endl;
    return -1;
  }
  std::cout << "Runtime version: " << rt_ver << std::endl;
  auto err2 = cudaDriverGetVersion(&dr_ver);
  if (err2 != cudaSuccess) {
    std::cerr << cudaGetErrorString(err2) << std::endl;
    return -1;
  }
  std::cout << "Driver version: " << dr_ver << std::endl;
  return 0;
}
```

## cuBLAS

**CUDA Basic Linear Algebra Subroutine library**

## CUTLASS

[examples](https://github.com/NVIDIA/cutlass/blob/master/media/docs/quickstart.md)

利用cutlass的warp和block层级的接口实现一个大的attention kernel为例，
难点主要是需要熟练使用cutlass，并能进行调优，而采用cutlass的开发有以下4个难点：

1. cutlass并不是一个如cublas那样使用简单的GEMM库。cutlass仅提供threads block, warp, instruction各个级别的原语，开发者需要熟悉cutlass源码，并根据需求组装原语实现kernel。
2. cutlass使用了大量的模版，在debug过程中需要阅读很长的log，且基本无法使用代码跳转工具，会给开发带来一定困难。
3. cutlass并不提供自动调优的功能，需要开发者手动选择最优的设置，或者接入框架的自动调优中。
4. 由于可能的组合非常多，基于cutlass进行开发时通常需要编写自动生成kernel的脚本。

从torch的实现上可以看出该工作需要的工作量还是比较大.

## CUB

```cpp
#include <cub/cub.cuh>

// Block-sorting CUDA kernel
__global__ void BlockSortKernel(int *d_in, int *d_out)
{
     using namespace cub;

     // Specialize BlockRadixSort, BlockLoad, and BlockStore for 128 threads
     // owning 16 integer items each
     typedef BlockRadixSort<int, 128, 16>                     BlockRadixSort;
     typedef BlockLoad<int, 128, 16, BLOCK_LOAD_TRANSPOSE>   BlockLoad;
     typedef BlockStore<int, 128, 16, BLOCK_STORE_TRANSPOSE> BlockStore;

     // Allocate shared memory
     __shared__ union {
         typename BlockRadixSort::TempStorage  sort;
         typename BlockLoad::TempStorage       load;
         typename BlockStore::TempStorage      store;
     } temp_storage;

     int block_offset = blockIdx.x * (128 * 16);	  // OffsetT for this block's ment

     // Obtain a segment of 2048 consecutive keys that are blocked across threads
     int thread_keys[16];
     BlockLoad(temp_storage.load).Load(d_in + block_offset, thread_keys);
     __syncthreads();

     // Collectively sort the keys
     BlockRadixSort(temp_storage.sort).Sort(thread_keys);
     __syncthreads();

     // Store the sorted segment
     BlockStore(temp_storage.store).Store(d_out + block_offset, thread_keys);
}
```

## cuSolver

**cuSolverDN**: Dense LAPACK

$ A \in \mathbf{R}^{n\times n}, Ax = b $, solve $ x $.

**cuSolverSP**: Sparse LAPACK

$ A \in \mathbf{R}^{n\times m}, Ax = b $, solve $ x = argmin || A x - b || $

**cuSolverRF**: Refactorization

$ A_i \in \mathbf{R}^{n\times n}, A_i x_i = f_i $, solve $ x_i\in \mathbf{R}^{n} $.

## Thrust 

```cpp
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>

int main() {
  // Generate 32M random numbers serially.
  thrust::default_random_engine rng(1337);
  thrust::uniform_int_distribution<int> dist;
  thrust::host_vector<int> h_vec(32 << 20);
  thrust::generate(h_vec.begin(), h_vec.end(), [&] { return dist(rng); });

  // Transfer data to the device.
  thrust::device_vector<int> d_vec = h_vec;

  // Sort data on the device.
  thrust::sort(d_vec.begin(), d_vec.end());

  // Transfer data back to host.
  thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
}
```

## Comparasion

NVIDIA HPC SDK vs CUDA Toolkit

cuda.h vs cuda_runtime.h

## Reference

* [CUDA](https://docs.nvidia.com/cuda/index.html)
* [cuBLAS](https://docs.nvidia.com/cuda/cublas/)
* [CUTLASS](https://github.com/NVIDIA/cutlass)
* [CUB doc](https://nvlabs.github.io/cub/index.html)
* [CUB github](https://github.com/NVIDIA/cub)
* [OpenBLAS](https://github.com/xianyi/OpenBLAS)
* [CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html)
* [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
* [Thrust](https://github.com/NVIDIA/thrust.git)
* [cuSolver](https://docs.nvidia.com/cuda/cusolver)
