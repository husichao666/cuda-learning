---
title: "2.1 CUDA C++ 入门"
description: "CUDA Programming Guide 13.4.2 · 第二章中文译文"
---

<a id="sec-2-1"></a>

# 2.1 CUDA C++ 入门

> 来源：《CUDA Programming Guide》13.4.2，书内第 21–39 页相关内容。正文为中文，常用 CUDA 与 memory 术语保留英文；代码保留原文，恢复 PDF 断行与缩进。


本节通过 C++ 中的具体用法介绍 CUDA 编程模型的基本概念。

本指南主要使用 CUDA Runtime API。它是 C++ 中最常用的 CUDA 接口，建立在更底层的 CUDA Driver API 之上。两者的区别见 [CUDA Runtime API 与 CUDA Driver API](/chapter-1/section-1-3#sec-1-3-2-1)；混合使用两套 API 的方法见 [CUDA Driver API](/chapter-3/section-3-3#sec-3-3)。

本节假定已安装 CUDA Toolkit 和 NVIDIA 驱动程序，并且系统中存在受支持的 NVIDIA GPU。安装方法参见 [CUDA Quickstart Guide](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)。

<a id="sec-2-1-1"></a>

## 2.1.1 使用 NVCC 编译

使用 C++ 编写的 GPU 代码由 NVIDIA CUDA 编译器 `nvcc` 编译。`nvcc` 是一个编译器驱动程序：它提供简单、常见的命令行选项，通过调用分别实现各个编译阶段的工具，简化 C++ 或 PTX 代码的编译过程。

本指南中的 `nvcc` 命令行可用于安装了 CUDA Toolkit 的 Linux、Windows 命令提示符或 PowerShell，以及安装了 CUDA Toolkit 的 Windows Subsystem for Linux。常用方法见 [NVCC：NVIDIA CUDA 编译器](/chapter-2/section-2-7#sec-2-7)，完整说明参见 [NVCC User Manual](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)。

<a id="sec-2-1-2"></a>

## 2.1.2 kernel

如 [编程模型](/chapter-1/section-1-2#sec-1-2) 所述，在 GPU 上执行、能从 host 调用的函数称为 kernel。kernel 通常由大量 thread 同时并行执行。

<a id="sec-2-1-2-1"></a>

### 2.1.2.1 声明 kernel

使用 `__global__` 声明说明符定义 kernel。这告诉编译器：该函数需要编译为 GPU 代码，并能通过 kernel launch 启动。kernel launch 通常由 CPU 发起，用于让 kernel 开始执行。kernel 的返回类型必须为 `void`。
```cpp
// Kernel definition
__global__ void vecAdd(float* A, float* B, float* C)
{
}
```

<a id="sec-2-1-2-2"></a>

### 2.1.2.2 启动 kernel

kernel launch 的参数包含并行执行该 kernel 的 thread 数量，这称为 execution configuration。同一个 kernel 的不同调用可以采用不同的配置，例如使用不同数量的 thread 或 block。

从 CPU 代码启动 kernel 有两种方法：三尖括号语法和 `cudaLaunchKernelEx`。这里介绍最常用的三尖括号语法；另一种方法的完整示例见 [cudaLaunchKernelEx](/chapter-3/section-3-1#sec-3-1-1)。

<a id="sec-2-1-2-2-1"></a>

#### 2.1.2.2.1 三尖括号语法

三尖括号语法是用于启动 kernel 的 CUDA C++ 语言扩展，因使用 `<<< >>>` 包围 execution configuration 而得名。配置参数在尖括号内用逗号分隔，形式类似函数实参。下面给出 `vecAdd` 的启动语法。
```cpp
__global__ void vecAdd(float* A, float* B, float* C)
 {
 }
int main()
{
    ...
    // Kernel invocation
    vecAdd<<<1, 256>>>(A, B, C);
    ...
}
```

前两个配置参数分别指定 grid 维度和 block 维度。对于一维 grid 或 block，可以直接使用整数。

上面的代码启动一个包含 256 个 thread 的 block，每个 thread 执行相同的 kernel 代码。[thread 与 grid 的索引内置变量](/chapter-2/section-2-1#sec-2-1-2-3) 将介绍如何用索引决定每个 thread 处理的数据。

由于同一 block 的所有 thread 都驻留在同一个 SM 上并共享其资源，因此每个 block 的 thread 数量存在上限。本版指南所述 GPU 每个 block 最多包含 1024 个 thread。资源允许时，一个 SM 可以同时调度多个 block。

kernel launch 对 host thread 是异步的。GPU 工作提交后，host 代码会继续执行，不会等待 kernel 完成，甚至不会等待 kernel 真正开始执行。因此，需要同步机制确认其执行完成。[同步 CPU 与 GPU](/chapter-2/section-2-1#sec-2-1-4) 介绍最基本的整个 GPU 同步，[异步执行](/chapter-2/section-2-5#sec-2-5) 介绍更精细的机制。

对于二维或三维 grid、block，使用 CUDA 类型 `dim3` 指定维度。下面使用 `16 × 16` 的 block grid，每个 block 为 `8 × 8`，启动 `MatAdd`。
```cpp
int main()
{
    ...
    dim3 grid(16,16);
    dim3 block(8,8);
    MatAdd<<<grid, block>>>(A, B, C);
    ...
}
```

<a id="sec-2-1-2-3"></a>

### 2.1.2.3 thread 与 grid 的索引内置变量

kernel 可以通过以下内置变量读取 execution configuration，以及当前 thread 或 block 的索引。

| 内置变量 | 含义 |
| --- | --- |
| `threadIdx` | 当前 thread 在 block 内的索引；同一 block 中各 thread 的索引不同。 |
| `blockDim` | 启动时指定的 block 维度。 |
| `blockIdx` | 当前 block 在 grid 内的索引；各 block 的索引不同。 |
| `gridDim` | 启动时指定的 grid 维度。 |

这些变量都是带 `.x`、`.y`、`.z` 成员的三分量向量。未指定的维度默认为 1。`threadIdx` 和 `blockIdx` 从零开始：`threadIdx.x` 的范围为 `0` 到 `blockDim.x - 1`，`.y`、`.z` 同理；`blockIdx.x` 的范围为 `0` 到 `gridDim.x - 1`，其他维度同理。

因此，thread 可以确定自己需要完成的工作。回到 `vecAdd`：三个参数都是浮点向量，kernel 将 `A`、`B` 逐元素相加，结果写入 `C`。每个 thread 完成一次加法，使用 thread 和 block 索引确定处理哪个元素。
```cpp
__global__ void vecAdd(float* A, float* B, float* C)
{
   // calculate which element this thread is responsible for computing
   int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
   // Perform computation
   C[workIndex] = A[workIndex] + B[workIndex];
}
int main()
{
    ...
    // A, B, and C are vectors of 1024 elements
    vecAdd<<<4, 256>>>(A, B, C);
    ...
}
```

本例使用 4 个 block，每个 block 256 个 thread，处理 1024 个元素。第一个 block 的 `blockIdx.x` 为 0，`workIndex` 就是 `threadIdx.x`；第二个 block 的 `blockIdx.x` 为 1，因此 `workIndex` 为 `threadIdx.x + 256`；第三个 block 则为 `threadIdx.x + 512`。

这种 `workIndex` 计算在一维并行化中很常见。推广到二维或三维时，通常对每个维度采用相同模式。

<a id="sec-2-1-2-3-1"></a>

#### 2.1.2.3.1 边界检查

前面的示例假设向量长度是 block 大小的整数倍，这里为 256。要支持任意长度，可以加入边界判断，让最后一个 block 中超出有效范围的 thread 不执行数组访问。
```cpp
__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
     // calculate which element this thread is responsible for computing
     int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
     if(workIndex < vectorLength)
     {
         // Perform computation
         C[workIndex] = A[workIndex] + B[workIndex];
     }
}
```

这样，即使启动的 thread 比实际需要的多，也不会发生数组越界。`workIndex >= vectorLength` 的 thread 不执行计算便退出。在一个 block 中启动少量不工作的额外 thread，开销通常不大；但应避免启动整个 block 都不工作的情况。

所需 block 数量为“所需 thread 数量除以每个 block 的 thread 数量，再向上取整”。下面在整数除法前加上 `threads - 1`，只有向量长度不能整除 block 大小时才增加一个 block。
```cpp
// vectorLength is an integer storing number of elements in the vector
int threads = 256;
int blocks = (vectorLength + threads-1)/threads;
vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
```

CUDA Core Compute Libraries（CCCL）提供 `cuda::ceil_div`，可以更方便地完成向上取整除法。包含 `<cuda/cmath>` 即可使用。
```cpp
// vectorLength is an integer storing number of elements in the vector
int threads = 256;
int blocks = cuda::ceil_div(vectorLength, threads);
vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
```

这里选择每个 block 256 个 thread 并非硬性要求，但它经常是一个不错的初始取值。

<a id="sec-2-1-3"></a>

## 2.1.3 GPU 计算中的 memory

要使用上面的 `vecAdd`，数组 `A`、`B`、`C` 必须存放在 GPU 可访问的 memory 中。这里演示两种方法，其他方法见 [Unified Memory 与 system memory](/chapter-2/section-2-6#sec-2-6)。GPU 代码可使用的 memory space 已在 [GPU memory](/chapter-1/section-1-2#sec-1-2-3) 中介绍，更详细的说明见 [GPU device memory space](/chapter-2/section-2-3#sec-2-3-3)。

<a id="sec-2-1-3-1"></a>

### 2.1.3.1 Unified Memory

Unified Memory 由 CUDA runtime 提供，让 NVIDIA 驱动程序管理 host 和 device 之间的数据移动。使用 `cudaMallocManaged` 分配，或用 `__managed__` 声明变量。CPU 或 GPU 访问时，驱动程序负责使相应 memory 可用。

下面的完整函数使用 Unified Memory 保存输入和输出向量，再启动 `vecAdd`。`cudaMallocManaged` 分配的 buffer 可以由 CPU 或 GPU 访问，使用 `cudaFree` 释放。
```cpp
void unifiedMemExample(int vectorLength)
{
    // Pointers to memory vectors
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));
    // Use unified memory to allocate buffers
    cudaMallocManaged(&A, vectorLength*sizeof(float));
    cudaMallocManaged(&B, vectorLength*sizeof(float));
    cudaMallocManaged(&C, vectorLength*sizeof(float));
    // Initialize vectors on the host
    initArray(A, vectorLength);
    initArray(B, vectorLength);
    // Launch the kernel. Unified memory will make sure A, B, and C are
    // accessible to the GPU
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
    // Wait for the kernel to complete execution
    cudaDeviceSynchronize();
    // Perform computation serially on CPU for comparison
    serialVecAdd(A, B, comparisonResult, vectorLength);
    // Confirm that CPU and GPU got the same answer
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Unified Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Unified Memory: Error - CPU and GPU answers do not match\n");
    }
    // Clean Up
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(comparisonResult);
}
```

所有受 CUDA 支持的操作系统和 GPU 都支持 Unified Memory，但底层机制和性能依系统架构而异，见 [Unified Memory](/chapter-2/section-2-6#sec-2-6-2)。某些 Linux 系统具有 Address Translation Services（ATS）或 Heterogeneous Memory Management（HMM），其中所有 system memory 都自动具备 Unified Memory 能力，无需使用 `cudaMallocManaged` 或 `__managed__`。

<a id="sec-2-1-3-2"></a>

### 2.1.3.2 显式 memory 管理

显式管理 memory allocation 及不同 memory space 之间的数据迁移，可以帮助提高性能，但代码会更长。下面使用 `cudaMalloc` 显式分配 GPU memory，并使用与 Unified Memory 相同的 `cudaFree` 释放。
```cpp
void explicitMemExample(int vectorLength)
{
    // Pointers for host memory
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));
    // Pointers for device memory
    float* devA = nullptr;
    float* devB = nullptr;
    float* devC = nullptr;
    //Allocate Host Memory using cudaMallocHost API. This is best practice
    // when buffers will be used for copies between CPU and GPU memory
    cudaMallocHost(&A, vectorLength*sizeof(float));
    cudaMallocHost(&B, vectorLength*sizeof(float));
    cudaMallocHost(&C, vectorLength*sizeof(float));
    // Initialize vectors on the host
    initArray(A, vectorLength);
    initArray(B, vectorLength);
    // start-allocate-and-copy
    // Allocate memory on the GPU
    cudaMalloc(&devA, vectorLength*sizeof(float));
    cudaMalloc(&devB, vectorLength*sizeof(float));
    cudaMalloc(&devC, vectorLength*sizeof(float));
    // Copy data to the GPU
    cudaMemcpy(devA, A, vectorLength*sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(devB, B, vectorLength*sizeof(float), cudaMemcpyDefault);
    cudaMemset(devC, 0, vectorLength*sizeof(float));
    // end-allocate-and-copy
    // Launch the kernel
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
    // wait for kernel execution to complete
    cudaDeviceSynchronize();
    // Copy results back to host
    cudaMemcpy(C, devC, vectorLength*sizeof(float), cudaMemcpyDefault);
    // Perform computation serially on CPU for comparison
    serialVecAdd(A, B, comparisonResult, vectorLength);
    // Confirm that CPU and GPU got the same answer
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Explicit Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Explicit Memory: Error - CPU and GPU answers to not match\n");
    }
    // clean up
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    free(comparisonResult);
}
```

`cudaMemcpy` 用于将 CPU buffer 的数据复制到 GPU buffer。它接收目标指针、源指针、字节数，以及类型为 `cudaMemcpyKind_t` 的方向参数，常用值如下。

| 参数值 | 复制方向 |
| --- | --- |
| `cudaMemcpyHostToDevice` | CPU 到 GPU |
| `cudaMemcpyDeviceToHost` | GPU 到 CPU |
| `cudaMemcpyDeviceToDevice` | 同一 GPU 内部或不同 GPU 之间 |

本例传入 `cudaMemcpyDefault`，让 CUDA 根据源、目标指针的值判断复制类型。

原文将这里的 `cudaMemcpy` 描述为同步调用，即复制完成后才返回。异步复制见 [在 CUDA stream 中启动 memory transfer](/chapter-2/section-2-5#sec-2-5-2-3)。

代码使用 `cudaMallocHost` 分配 host 上的 page-locked memory（页锁定内存）。这有利于提高复制性能，也是异步 memory transfer 的必要条件。通常，用于 CPU 与 GPU 数据传输的 CPU buffer 应采用 page-locked memory；但锁定过多 host memory 可能降低部分系统的性能，因此只对需要向 GPU 发送或接收数据的 buffer 进行页锁定。

<a id="sec-2-1-3-3"></a>

### 2.1.3.3 memory 管理与应用性能

显式 memory 管理要求程序员写出 host 与 device 之间的复制，因此代码更长。这既是缺点，也是优点：程序可以准确控制何时复制、数据驻留在哪里，以及各处实际分配哪些 memory。通过控制传输时机并让传输与计算重叠，可以获得性能收益。

使用 Unified Memory 时，也可以通过 [Memory Advise 与 Prefetch](/chapter-2/section-2-6#sec-2-6-2-4) 中介绍的 API，向管理 memory 的驱动程序提供提示，从而获得显式 memory 管理的部分性能优势。

<a id="sec-2-1-4"></a>

## 2.1.4 同步 CPU 与 GPU

如 [启动 kernel](/chapter-2/section-2-1#sec-2-1-2-2) 所述，kernel launch 相对于调用它的 CPU thread 是异步的：kernel 完成之前，甚至启动之前，CPU 控制流就可能继续执行。要确保 host 代码继续前 kernel 已完成，必须使用同步机制。

最简单的方法是 `cudaDeviceSynchronize`。它阻塞 host thread，直到先前提交到当前 GPU 的全部工作完成。本节示例只在 GPU 上执行单个操作，因此足够使用。

较大应用程序可能有多个 stream 同时执行工作，`cudaDeviceSynchronize` 会等待所有 stream。此时推荐使用 [stream 同步 API](/chapter-2/section-2-5#sec-2-5-2-4)，只等待特定 stream，或使用 [CUDA event](/chapter-2/section-2-5#sec-2-5-3)。这些方法详见 [异步执行](/chapter-2/section-2-5#sec-2-5)。

<a id="sec-2-1-5"></a>

## 2.1.5 完整示例

下面给出向量加法 kernel、host 代码以及校验计算结果所需的辅助函数。默认向量长度为 1024，也可以通过命令行参数指定长度。

<strong>Unified Memory 版本：</strong>
```cpp
#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <cuda/cmath>
__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
    int workIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if(workIndex < vectorLength)
    {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}
void initArray(float* A, int length)
{
     std::srand(std::time({}));
    for(int i=0; i<length; i++)
    {
        A[i] = rand() / (float)RAND_MAX;
    }
}
void serialVecAdd(float* A, float* B, float* C, int length)
{
    for(int i=0; i<length; i++)
    {
        C[i] = A[i] + B[i];
    }
}
bool vectorApproximatelyEqual(float* A, float* B, int length, float epsilon=0.00001)
{
    for(int i=0; i<length; i++)
    {
        if(fabs(A[i] -B[i]) > epsilon)
        {
            printf("Index %d mismatch: %f != %f", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}
//unified-memory-begin
void unifiedMemExample(int vectorLength)
{
    // Pointers to memory vectors
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));
    // Use unified memory to allocate buffers
    cudaMallocManaged(&A, vectorLength*sizeof(float));
    cudaMallocManaged(&B, vectorLength*sizeof(float));
    cudaMallocManaged(&C, vectorLength*sizeof(float));
    // Initialize vectors on the host
    initArray(A, vectorLength);
    initArray(B, vectorLength);
    // Launch the kernel. Unified memory will make sure A, B, and C are
    // accessible to the GPU
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
    // Wait for the kernel to complete execution
    cudaDeviceSynchronize();
    // Perform computation serially on CPU for comparison
    serialVecAdd(A, B, comparisonResult, vectorLength);
    // Confirm that CPU and GPU got the same answer
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Unified Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Unified Memory: Error - CPU and GPU answers do not match\n");
    }
    // Clean Up
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(comparisonResult);
}
//unified-memory-end
int main(int argc, char** argv)
{
    int vectorLength = 1024;
    if(argc >=2)
    {
        vectorLength = std::atoi(argv[1]);
    }
    unifiedMemExample(vectorLength);
    return 0;
}
```

<strong>显式 memory 管理版本：</strong>
```cpp
#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <cuda/cmath>
__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
    int workIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if(workIndex < vectorLength)
    {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}
void initArray(float* A, int length)
{
     std::srand(std::time({}));
    for(int i=0; i<length; i++)
    {
        A[i] = rand() / (float)RAND_MAX;
    }
}
void serialVecAdd(float* A, float* B, float* C, int length)
{
    for(int i=0; i<length; i++)
    {
        C[i] = A[i] + B[i];
    }
}
bool vectorApproximatelyEqual(float* A, float* B, int length, float epsilon=0.00001)
{
    for(int i=0; i<length; i++)
    {
        if(fabs(A[i] -B[i]) > epsilon)
        {
            printf("Index %d mismatch: %f != %f", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}
//explicit-memory-begin
void explicitMemExample(int vectorLength)
{
    // Pointers for host memory
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));
    // Pointers for device memory
    float* devA = nullptr;
    float* devB = nullptr;
    float* devC = nullptr;
    //Allocate Host Memory using cudaMallocHost API. This is best practice
    // when buffers will be used for copies between CPU and GPU memory
    cudaMallocHost(&A, vectorLength*sizeof(float));
    cudaMallocHost(&B, vectorLength*sizeof(float));
    cudaMallocHost(&C, vectorLength*sizeof(float));
    // Initialize vectors on the host
    initArray(A, vectorLength);
    initArray(B, vectorLength);
    // start-allocate-and-copy
    // Allocate memory on the GPU
    cudaMalloc(&devA, vectorLength*sizeof(float));
    cudaMalloc(&devB, vectorLength*sizeof(float));
    cudaMalloc(&devC, vectorLength*sizeof(float));
    // Copy data to the GPU
    cudaMemcpy(devA, A, vectorLength*sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(devB, B, vectorLength*sizeof(float), cudaMemcpyDefault);
    cudaMemset(devC, 0, vectorLength*sizeof(float));
    // end-allocate-and-copy
    // Launch the kernel
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
    // wait for kernel execution to complete
    cudaDeviceSynchronize();
    // Copy results back to host
    cudaMemcpy(C, devC, vectorLength*sizeof(float), cudaMemcpyDefault);
    // Perform computation serially on CPU for comparison
    serialVecAdd(A, B, comparisonResult, vectorLength);
    // Confirm that CPU and GPU got the same answer
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Explicit Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Explicit Memory: Error - CPU and GPU answers to not match\n");
    }
    // clean up
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    free(comparisonResult);
}
//explicit-memory-end
int main(int argc, char** argv)
{
    int vectorLength = 1024;
    if(argc >=2)
    {
        vectorLength = std::atoi(argv[1]);
    }
    explicitMemExample(vectorLength);
    return 0;
}
```

使用 `nvcc` 构建和运行：
```sh
$ nvcc vecAdd_unifiedMemory.cu -o vecAdd_unifiedMemory
$ ./vecAdd_unifiedMemory
Unified Memory: CPU and GPU answers match
$ ./vecAdd_unifiedMemory 4096
Unified Memory: CPU and GPU answers match
$ nvcc vecAdd_explicitMemory.cu -o vecAdd_explicitMemory
$ ./vecAdd_explicitMemory
Explicit Memory: CPU and GPU answers match
$ ./vecAdd_explicitMemory 4096
Explicit Memory: CPU and GPU answers match
```

在这些示例中，各 thread 独立工作，无需彼此协调或同步。其他任务中，thread 往往需要合作与通信；同一 block 的 thread 可以通过 shared memory 共享数据，并通过同步协调 memory access。

最基本的 block 级同步是 `__syncthreads()`。它充当 barrier：block 的所有 thread 到达前，任何 thread 都不能越过该位置继续执行。使用示例见 [Shared Memory](/chapter-2/section-2-3#sec-2-3-3-2)。

shared memory 通常是靠近处理器核心的低延迟 memory，类似 L1 cache；`__syncthreads()` 也设计得比较轻量。它只同步同一个 block 内的 thread。

不同 block 之间的同步仅在特定条件下受支持。例如，thread block cluster 允许同一 cluster 中的 block 同步；Cooperative Groups 提供创建跨 block 同步域的机制。

通常，将同步限制在 block 内能取得更好的性能。不同 block 仍可通过 atomic memory 函数共同更新结果，后续会介绍。更精细的同步原语见 [高级同步原语](/chapter-3/section-3-2#sec-3-2-4)，它们可以帮助充分利用性能与资源。

<a id="sec-2-1-6"></a>

## 2.1.6 runtime 初始化

CUDA runtime 为系统中的 device 创建 CUDA context。该 context 称为对应 device 的 primary context，在首次调用需要该 device 活跃 context 的 runtime 函数时初始化，并由应用程序的所有 host thread 共享。创建过程中，必要时会对 device code 进行 JIT 编译并加载到 device memory；这些步骤对应用程序透明。Driver API 也能访问 runtime 创建的 primary context，互操作方法见 [Runtime API 与 Driver API 的互操作](/chapter-3/section-3-3#sec-3-3-4)。

从 CUDA 12.0 开始，`cudaInitDevice` 和 `cudaSetDevice` 会初始化 runtime 及指定 device 的 primary context。如果在调用它们之前就发起 runtime API 请求，runtime 会按需自初始化，并隐式使用 device 0。测量 runtime 函数耗时、解释首次调用的错误码时，应考虑初始化开销。CUDA 12.0 之前，`cudaSetDevice` 不初始化 runtime。

`cudaDeviceReset` 销毁当前 device 的 primary context；之后再次调用 CUDA runtime API，会为该 device 创建新的 primary context。

::: warning 注意
CUDA 接口依赖全局状态，它在 host 程序启动时初始化，在程序终止时销毁。在程序初始化阶段，或 `main` 结束后的程序终止阶段，隐式或显式调用 CUDA 接口可能导致未定义行为。

从 CUDA 12.0 开始，`cudaSetDevice` 在为 host thread 切换当前 device 后，会显式初始化尚未初始化的 runtime。旧版本会推迟到切换后的首次 runtime 调用。因此，应检查 `cudaSetDevice` 返回值，以捕获初始化错误。
:::

参考手册中错误处理与版本管理部分的 runtime 函数不会初始化 runtime。

<a id="sec-2-1-7"></a>

## 2.1.7 CUDA 错误检查

CUDA API 返回枚举类型 `cudaError_t`。示例代码常省略检查，但生产应用程序应始终检查和处理每次调用的返回值。没有错误时返回 `cudaSuccess`。

许多应用程序使用类似下面的辅助宏：
```cpp
#define CUDA_CHECK(expr_to_check) do { \
    cudaError_t result = expr_to_check; \
    if(result != cudaSuccess) \
    { \
        fprintf(stderr, \
                "CUDA Runtime Error: %s:%i:%d = %s\n", \
                __FILE__, \
                __LINE__, \
                result,\
                cudaGetErrorString(result)); \
    } \
} while(0)
```

宏调用 `cudaGetErrorString`，将 `cudaError_t` 转为可读的错误描述。调用 runtime API 时，将表达式包在 `CUDA_CHECK` 中：
```cpp
CUDA_CHECK(cudaMalloc(&devA, vectorLength*sizeof(float)));
CUDA_CHECK(cudaMalloc(&devB, vectorLength*sizeof(float)));
CUDA_CHECK(cudaMalloc(&devC, vectorLength*sizeof(float)));
```

如果发生错误，宏会将错误打印到 `stderr`。小项目常采用这种方式；大型程序可以将它改为接入日志系统或其他错误处理机制。

::: warning 注意
某次 CUDA API 返回的错误，也可能来自之前提交的异步操作，详见 [异步错误处理](/chapter-2/section-2-5#sec-2-5-4-2)。
:::

<a id="sec-2-1-7-1"></a>

### 2.1.7.1 错误状态

CUDA runtime 为每个 host thread 维护一个 `cudaError_t` 错误状态，初始为 `cudaSuccess`，发生错误时被覆盖。`cudaGetLastError` 返回当前状态并将其重置为 `cudaSuccess`；`cudaPeekAtLastError` 读取状态但不清除。

三尖括号形式的 kernel launch 不返回 `cudaError_t`，因此应在启动后立即检查错误状态，以捕获启动时的即时错误或此前的异步错误。

此时得到 `cudaSuccess`，不代表 kernel 已成功完成，甚至不代表已开始执行。它只能说明传给 runtime 的 launch 参数与 execution configuration 没有触发错误，且当前状态中没有先前操作留下的错误。

<a id="sec-2-1-7-2"></a>

### 2.1.7.2 异步错误

kernel launch 和许多 runtime API 都是异步的，详见 [异步执行](/chapter-2/section-2-5#sec-2-5)。错误发生时会设置并覆盖 CUDA 错误状态，因此异步操作执行期间的错误，只有在下一次读取状态时才会被报告。读取可能来自 `cudaGetLastError`、`cudaPeekAtLastError`，也可能来自其他返回 `cudaError_t` 的 CUDA API。

runtime API 返回错误时，不会因此自动清除错误状态。例如 kernel 非法 memory access 产生的异步错误，可能由后续多个 runtime 调用重复报告；原文要求使用 `cudaGetLastError` 清除所保存的错误状态。
```cpp
vecAdd<<<blocks, threads>>>(devA, devB, devC);
// check error state after kernel launch
CUDA_CHECK(cudaGetLastError());
// wait for kernel execution to complete
// The CUDA_CHECK will report errors that occurred during execution of the kernel
CUDA_CHECK(cudaDeviceSynchronize());
```

::: tip 注意
`cudaStreamQuery` 和 `cudaEventQuery` 可能返回 `cudaErrorNotReady`。这表示操作尚未完成，不被视为上述错误，也不会由 `cudaPeekAtLastError` 或 `cudaGetLastError` 报告。
:::

<a id="sec-2-1-7-3"></a>

### 2.1.7.3 CUDA_LOG_FILE

环境变量 `CUDA_LOG_FILE` 也能帮助定位错误。设置后，CUDA driver 会将遇到的错误消息写入该变量指定的文件。下面的错误示例尝试启动一个超过所有受支持架构 block 大小上限的 block。
```cpp
__global__ void k()
{ }
int main()
{
        k<<<8192, 4096>>>(); // Invalid block size
        CUDA_CHECK(cudaGetLastError());
        return 0;
}
```

构建并运行后，[错误检查宏](/chapter-2/section-2-1#sec-2-1-7) 会报告：
```sh
$ nvcc errorLogIllustration.cu -o errlog
$ ./errlog
CUDA Runtime Error: /home/cuda/intro-cpp/errorLogIllustration.cu:24:1 = invalid argument
```

如果设置 `CUDA_LOG_FILE`，日志文件会提供更详细的信息。
```sh
$ env CUDA_LOG_FILE=cudaLog.txt ./errlog
CUDA Runtime Error: /home/cuda/intro-cpp/errorLogIllustration.cu:24:1 = invalid argument
$ cat cudaLog.txt
[12:46:23.854][137216133754880][CUDA][E] One or more of block dimensions of (4096,1,1) exceeds corresponding maximum value of (1024,1024,64)
[12:46:23.854][137216133754880][CUDA][E] Returning 1 (CUDA_ERROR_INVALID_VALUE) from cuLaunchKernel
```

将 `CUDA_LOG_FILE` 设置为 `stdout` 或 `stderr`，可分别输出到标准输出或标准错误。即使应用程序没有正确检查返回值，也能借此收集、定位 CUDA 错误。

这对调试很有用，但单靠环境变量不能让应用程序在运行时处理错误并恢复。CUDA 错误日志功能还允许向 driver 注册 callback，错误发生时调用它，从而接入已有日志系统并执行运行时错误处理。

更多示例见 [错误日志管理](/chapter-4/section-4-9#sec-4-9)。该功能与 `CUDA_LOG_FILE` 要求 NVIDIA driver r570 或更新版本。

<a id="sec-2-1-8"></a>

## 2.1.8 device 函数与 host 函数

`__global__` 指定 kernel 的入口，即将在 GPU 上并行执行的函数。kernel 通常由 host 启动，也可以使用 dynamic parallelism 从另一个 kernel 内启动，见 [CUDA Dynamic Parallelism](/chapter-4/section-4-20#sec-4-20)。

`__device__` 表示函数编译为 GPU 代码，并可由其他 `__device__` 或 `__global__` 函数调用。函数，包括类成员函数、functor 和 lambda，可以同时指定 `__device__` 和 `__host__`。

<a id="sec-2-1-9"></a>

## 2.1.9 变量说明符

CUDA 说明符可控制静态变量存放位置。

| 说明符 | 存放位置 |
| --- | --- |
| `__device__` | global memory |
| `__constant__` | constant memory |
| `__managed__` | Unified Memory |
| `__shared__` | shared memory |

在 `__device__` 或 `__global__` 函数内部，不带说明符的变量优先放入 register，必要时放入 local memory。在这些函数之外，不带说明符的变量分配在 system memory 中。

<a id="sec-2-1-9-1"></a>

### 2.1.9.1 检测 device 编译

给函数标记 `__host__ __device__`，编译器会分别生成 GPU 和 CPU 版本。在这种函数中，可以通过预处理器为两个版本选择不同代码。常用方法是检查 `__CUDA_ARCH__` 是否已定义。

<a id="sec-2-1-10"></a>

## 2.1.10 thread block cluster

从 Compute Capability 9.0 开始，CUDA 增加了可选的 thread block cluster 层级。与同一个 block 的 thread 保证共同调度到一个 SM 类似，同一个 cluster 的 block 保证共同调度到同一个 GPC。

cluster 也可按一维、二维或三维组织，参见 [图 5](/chapter-1/section-1-2#figure-05)。cluster 中 block 数量可以自定义；CUDA 保证可移植的 cluster 大小上限为 8 个 block。GPU 或 MIG 配置过小、不足以支持 8 个 SM 时，上限会相应减少。部分更大配置可以支持超过 8 个 block，但依赖具体架构，可通过 `cudaOccupancyMaxPotentialClusterSize` 查询。

同一 cluster 的所有 block 保证在同一 GPC 上共同调度、同时执行，并能使用 Cooperative Groups 的 `cluster.sync()` 进行硬件支持的同步。`num_threads()`、`num_blocks()` 分别查询 group 的 thread 数量和 block 数量；原文还列出 `dim_threads()`、`dim_blocks()` 作为相关 thread、block 维度信息的查询接口。

cluster 中的 block 能访问 distributed shared memory，即全部成员 block 的 shared memory 的组合，并能对其中任意地址读取、写入或执行 atomic 操作。直方图示例见 [Distributed Shared Memory](/chapter-2/section-2-3#sec-2-3-3-8)。

::: tip 注意
为了保持兼容性，采用 cluster 的 kernel 中，`gridDim` 仍以 block 数量表示 grid 大小。block 在 cluster 中的 rank 可以通过 Cooperative Groups API 查询。
:::

<a id="sec-2-1-10-1"></a>

### 2.1.10.1 使用三尖括号启动 cluster

可以用编译时 kernel 属性 `__cluster_dims__(X,Y,Z)`，或用 `cudaLaunchKernelEx` 启用 cluster。下面在编译时将 cluster 大小固定为 `(2, 1, 1)`，随后仍使用三尖括号启动 kernel。如果 kernel 已声明编译时 cluster 大小，launch 时不能再修改它。
```cpp
// Kernel definition
// Compile time cluster size 2 in X-dimension and 1 in Y and Z dimension
__global__ void __cluster_dims__(2, 1, 1) cluster_kernel(float *input, float* output)
{
}
int main()
{
    float *input, *output;
    // Kernel invocation with compile time cluster size
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    // The grid dimension is not affected by cluster launch, and is still enumerated
    // using number of blocks.
    // The grid dimension must be a multiple of cluster size.
    cluster_kernel<<<numBlocks, threadsPerBlock>>>(input, output);
}
```
