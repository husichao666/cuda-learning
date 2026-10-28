---
title: "2.7 NVCC：NVIDIA CUDA 编译器"
description: "CUDA Programming Guide 13.4.2 · 第二章中文译文"
---

<a id="sec-2-7"></a>

# 2.7 NVCC：NVIDIA CUDA 编译器

> 来源：《CUDA Programming Guide》13.4.2，书内第 128–136 页相关内容。正文为中文，常用 CUDA 与 memory 术语保留英文；代码保留原文，恢复 PDF 断行与缩进。


`nvcc` 是 NVIDIA 提供、用于编译 CUDA C/C++ 与 PTX 的工具链，属于 CUDA Toolkit，包含编译器、linker、PTX 和 cubin 汇编相关工具。顶层 `nvcc` 协调各阶段，并调用所需工具。

`nvcc` 负责离线编译，与 CUDA runtime compiler `nvrtc` 的运行时或 JIT 编译相对。本节介绍构建应用程序最常用的方法，完整参考见 [NVCC Documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)。

<a id="sec-2-7-1"></a>

## 2.7.1 CUDA 源文件与头文件

输入可以包含 CPU 上的 host code、GPU 上的 device code，或二者混合。仅有 host code 时常用 `.c`、`.cpp`、`.cc`、`.cxx`；包含 device code 时使用 `.cu`。包含 device code 的头文件通常用 `.cuh` 区分。

| 扩展名 | 描述 | 内容 |
| --- | --- | --- |
| `.c` | C 源文件 | host-only code |
| `.cpp`、`.cc`、`.cxx` | C++ 源文件 | host-only code |
| `.h`、`.hpp`、`.hh`、`.hxx` | C/C++ 头文件 | 可含 host、device 或混合代码 |
| `.cu` | CUDA 源文件 | 可含 host、device 或混合代码 |
| `.cuh` | CUDA 头文件 | 可含 host、device 或混合代码 |

<a id="sec-2-7-2"></a>

## 2.7.2 NVCC 编译流程

首先，`nvcc` 将 host code 与 device code 分离，交给相应编译器处理。

编译 host code 需要系统具有兼容的 host compiler，支持范围由 CUDA Toolkit 的 Linux、Windows 平台策略规定。纯 host 文件可以直接用 host compiler 编译，其 object file 可在链接时与 `nvcc` 生成的 object file 组合。

GPU compiler 将 C/C++ device code 编译为 PTX。对命令行指定的每个虚拟架构，例如 `compute_90`，分别执行相应编译。随后 `ptxas` 将 PTX 转为面向实际 SM 架构的 cubin。

多个 PTX、cubin 目标可以装入同一个 fatbin，再嵌入可执行文件或库，使单一程序支持多个虚拟和真实硬件架构。

这些调用由 `nvcc` 自动协调。使用 `-v` 查看完整流程与命令；`-keep` 保留中间文件，默认写入当前目录，也可用 `--keep-dir` 指定位置。

以下是示例 `example.cu`：
```cpp
// ----- example.cu -----
#include <stdio.h>
__global__ void kernel() {
    printf("Hello from kernel\n");
}
void kernel_launcher() {
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}
int main() {
    kernel_launcher();
    return 0;
}
```

<strong>NVCC 基本编译流程：</strong>

![NVCC 基本编译流程](/images/chapter-02/nvcc-whole-program.png)

<strong>包含多个 PTX/cubin 目标的编译流程：</strong>

![NVCC 多架构 PTX 与 cubin 编译流程](/images/chapter-02/nvcc-multi-architecture.png)

两图均直接提取自 PDF。更详细的阶段说明见 [NVCC Documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)。

<a id="sec-2-7-3"></a>

## 2.7.3 NVCC 基本用法

编译 CUDA 源文件的基本命令：
```sh
nvcc <source_file>.cu -o <output_file>
```

常用选项包括 `-I <path>` 指定头文件目录、`-L <path>` 指定库目录、`-l<library>` 链接库，以及 `-D<macro>=<value>` 定义宏。
```sh
nvcc example.cu -I path_to_include/ -L path_to_library/ -lcublas -o <output_file>
```

<a id="sec-2-7-3-1"></a>

### 2.7.3.1 生成 PTX 与 cubin

默认情况下，`nvcc` 为 Toolkit 支持的较早架构生成 PTX 与 cubin，以扩大兼容范围。使用 `-arch` 指定目标，或用 `-gencode` 为多个架构分别配置。

`--list-gpu-code` 列出真实 GPU 目标，`--list-gpu-arch` 列出虚拟架构；也可查阅 NVCC 文档中的 Virtual Architecture List 与 GPU Architecture List。
```sh
nvcc --list-gpu-code # list all supported real GPU architectures
nvcc --list-gpu-arch # list all supported virtual GPU architectures
nvcc example.cu -arch=compute_<XY> # e.g. -arch=compute_80 for NVIDIA Ampere GPUs and later
                                   # PTX-only, GPU forward compatible
nvcc example.cu -arch=sm_<XY> # e.g. -arch=sm_80 for NVIDIA Ampere GPUs and later
                                   # PTX and Cubin, GPU forward compatible
nvcc example.cu -arch=native # automatically detects and generates Cubin for the current GPU
                                   # no PTX, no GPU forward compatibility
nvcc example.cu -arch=all # generate Cubin for all supported GPU architectures
                                   # also includes the latest PTX for GPU forward compatibility
nvcc example.cu -arch=all-major # generate Cubin for all major supported GPU architectures, e.g. sm_80, sm_90,
                                   # also includes the latest PTX for GPU forward compatibility
```

更高级的用法可以独立指定 PTX 和 cubin 目标：
```sh
# generate PTX for virtual architecture compute_80 and compile it to Cubin for real architecture sm_86, keep compute_80 PTX
nvcc example.cu -arch=compute_80 -gpu-code=sm_86,compute_80 # (PTX and Cubin)
# generate PTX for virtual architecture compute_80 and compile it to Cubin for real architecture sm_86, sm_89
nvcc example.cu -arch=compute_80 -gpu-code=sm_86,sm_89 # (no PTX)
nvcc example.cu -gencode=arch=compute_80,code=sm_86,sm_89 # same as above
# (1) generate PTX for virtual architecture compute_80 and compile it to Cubin for real architecture sm_86, sm_89
# (2) generate PTX for virtual architecture compute_90 and compile it to Cubin for real architecture sm_90
nvcc example.cu -gencode=arch=compute_80,code=sm_86,sm_89 -gencode=arch=compute_90,code=sm_90
```

完整 code generation 选项见 [NVCC Documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)。

<a id="sec-2-7-3-2"></a>

### 2.7.3.2 Host code 编译说明

不包含 device code 或 device 符号的 compilation unit，即源文件及其头文件，可直接交给 host compiler。

只要使用 CUDA Runtime API，最终程序就需要链接 CUDA runtime library。它有静态 `libcudart_static` 和动态 `libcudart` 两种形式；`nvcc` 默认链接静态版本，使用 `--cudart=shared` 可选择动态版本。

`-ccbin <compiler>` 指定 host compiler；也可用环境变量 `NVCC_CCBIN`。`-Xcompiler` 将参数转交 host compiler，下面将 `-O3` 转交给它。
```sh
nvcc example.cu -ccbin=clang++
export NVCC_CCBIN='gcc'
nvcc example.cu -Xcompiler=-O3
```

<a id="sec-2-7-3-3"></a>

### 2.7.3.3 GPU code 分离编译

默认 whole-program compilation 要求某个 compilation unit 使用的 GPU code 和符号都可在该单元内获得。若 device 函数需要调用其他单元的函数或访问其变量，必须通过 `-rdc=true` 或相关 `-dc` 编译模式启用 device linking，这称为 separate compilation。

这种方式允许更灵活地组织代码，可以改善编译时间和二进制大小，但增加构建流程复杂度，也可能因为跨文件优化受限而降低性能，因此默认不开启。LTO 可减轻这一开销。

使用时需满足：

- 在其他 compilation unit 中引用非 const device 变量时，使用 `extern`。
- Const device 变量的定义和引用均按要求使用 `extern`。
- 相关 `.cu` 文件采用 `-dc` 或 `-rdc=true` 模式编译。

普通 host、device 函数默认具有 external linkage，通常不需要 `extern`。但从 CUDA 13 开始，`__global__` 函数，以及 `__managed__`、`__device__`、`__constant__` 变量默认具有 internal linkage，应结合声明规则处理。

下面 `definition.cu` 定义变量、函数，`example.cu` 引用，再分别编译并链接。
```cpp
// ----- definition.cu -----
extern __device__ int device_variable = 5;
__device__ int device_function() { return 10; }
// ----- example.cu -----
extern __device__ int device_variable;
__device__ int device_function();
__global__ void kernel(int* ptr) {
    device_variable = 0;
    *ptr = device_function();
}
```
```sh
nvcc -dc definition.cu -o definition.o
nvcc -dc example.cu -o example.o
nvcc definition.o example.o -o program
```

<a id="sec-2-7-4"></a>

## 2.7.4 常用编译器选项

下面列出语言功能、调试、优化、profiling 和构建方面的常见选项。完整说明见 [NVCC Documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)。

<a id="sec-2-7-4-1"></a>

### 2.7.4.1 语言功能

NVCC 支持 C++03 到 C++23 的核心语言功能，可使用 `--std={c++03|c++11|c++14|c++17|c++20|c++23}` 选择标准版本。

其他扩展选项如下：

| 选项 | 含义 |
| --- | --- |
| `-restrict` | 承诺所有 kernel 指针参数满足 restrict 条件。 |
| `-extended-lambda` | 允许 lambda 使用 `__host__`、`__device__` 注解。 |
| `-expt-relaxed-constexpr` | 实验性允许 host 调用 device constexpr 函数，以及 device 调用 host constexpr 函数。 |

详见 [Extended Lambdas](/chapter-5/section-5-3#sec-5-3-8-2) 和 [constexpr 函数](/chapter-5/section-5-3#sec-5-3-11-3)。

<a id="sec-2-7-4-2"></a>

### 2.7.4.2 调试选项

| 选项 | 含义 |
| --- | --- |
| `-g` | 生成 host 调试信息，供 GDB、LLDB 等使用。 |
| `-G` | 生成 device 调试信息，供 CUDA-GDB 使用，并定义 `__CUDACC_DEBUG__`。 |
| `-lineinfo` | 生成 device 行号信息，不改变执行性能，方便 Compute Sanitizer 等工具定位源码。 |

默认 GPU 优化级别为 `-O3`。`-G` 会禁用部分优化，因此调试构建通常更慢。运行时 assertion 也可能降低性能，可通过 `-DNDEBUG` 禁用。

<a id="sec-2-7-4-3"></a>

### 2.7.4.3 优化选项

常用代码生成控制如下：

| 选项 | 用途 |
| --- | --- |
| `-Xptxas` | 向 PTX assembler 传参，例如 `-Xptxas=-maxrregcount=N` 限制每个 thread 的 register 数。 |
| `-extra-device-vectorization` | 更积极的 device code vectorization。 |
| `--apply-controls=/path/to/file` | 向 NVCC、PTXAS 提供 Advanced Controls File（ACF），针对 workload 改变默认编译行为。 |

原文提醒，ACF 可能导致编译失败或运行结果错误，生成方法参见 [CompileIQ GitHub 文档](https://github.com/NVIDIA/CompileIQ)。浮点行为的细粒度控制见 [浮点计算](/chapter-5/section-5-5#sec-5-5) 与 [NVCC Documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)。

以下选项用于获取优化诊断信息：

| 选项 | 输出 |
| --- | --- |
| `-res-usage` | 各 kernel 的 register、shared、constant、local memory 使用报告。 |
| `-opt-info=inline` | 函数内联信息。 |
| `-Xptxas=-warn-lmem-usage` | 使用 local memory 时警告。 |
| `-Xptxas=-warn-spills` | register spill 到 local memory 时警告。 |

<a id="sec-2-7-4-4"></a>

### 2.7.4.4 Link-Time Optimization（LTO）

Separate compilation 限制跨文件优化，可能降低性能。LTO 在链接时优化分离编译的文件，以增加编译时间为代价，在保留组织灵活性的同时恢复较多 whole-program compilation 性能。

使用 `-dlto` 或 `lto_<SM version>` 目标启用：
```sh
nvcc -dc -dlto -arch=sm_100 definition.cu -o definition.o
nvcc -dc -dlto -arch=sm_100 example.cu -o example.o
nvcc -dlto definition.o example.o -o program
nvcc -dc -arch=lto_100 definition.cu -o definition.o
nvcc -dc -arch=lto_100 example.cu -o example.o
nvcc -dlto definition.o example.o -o program
```

<a id="sec-2-7-4-5"></a>

### 2.7.4.5 Profiling 选项

Nsight Compute、Nsight Systems 不要求额外编译参数就能分析 CUDA 应用程序，但以下信息有助于将源码与生成代码关联。

- `-lineinfo`：生成 device 行号，profiling 工具据此显示源码；通常要求原文件仍位于编译时路径。
- `-src-in-ptx`：将原始源码保留在 PTX 中，缓解上述路径限制；需要同时启用 `-lineinfo`。

<a id="sec-2-7-4-6"></a>

### 2.7.4.6 Fatbin 压缩

NVCC 默认压缩可执行文件或库中的 fatbin。

- `-no-compress`：禁用压缩。
- `--compress-mode={default|size|speed|balance|none}`：选择模式。`speed` 优先解压速度，`size` 优先缩小体积，`balance` 折中，默认模式为 `speed`，`none` 禁用压缩。

<a id="sec-2-7-4-7"></a>

### 2.7.4.7 编译性能控制

下列选项用于分析或加速编译过程本身。

| 选项 | 作用 |
| --- | --- |
| `-t <N>` | 并行编译一个 compilation unit 的多个架构目标时，使用的 CPU thread 数。 |
| `-split-compile <N>` | 并行优化阶段使用的 CPU thread 数。 |
| `-split-compile-extended <N>` | 更积极的 split compilation，需要 LTO。 |
| `-Ofc <N>` | 指定 device code 编译速度级别。 |
| `-time <filename>` | 将各编译阶段耗时写为 CSV。 |
| `-fdevice-time-trace` | 生成 device code 编译的时间 trace。 |
