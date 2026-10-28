---
title: "2.2 CUDA Python 入门"
description: "CUDA Programming Guide 13.4.2 · 第二章中文译文"
---

<a id="sec-2-2"></a>

# 2.2 CUDA Python 入门

> 来源：《CUDA Programming Guide》13.4.2，书内第 39–47 页相关内容。正文为中文，常用 CUDA 与 memory 术语保留英文；代码保留原文，恢复 PDF 断行与缩进。


本节介绍如何使用 Python 编写 CUDA kernel。CUDA Python 包含一系列仍在积极发展的工具和库。下面先介绍其中一些组件，再使用它们演示如何编写、运行 GPU 代码。

Python 中利用 GPU 的方法很多，不少方法不要求直接编写 kernel。CUDA Python 生态中的部分组件可以直接在 GPU 上完成运算，无需开发者显式控制 GPU。NVIDIA Accelerated Computing Hub 的 [Accelerated Python User’s Guide](https://github.com/NVIDIA/accelerated-computing-hub/tree/main/Accelerated_Python_User_Guide/notebooks) 介绍了相关库和工具，适合希望尽快使用 GPU、而不一定亲自编写 GPU 代码的读者。

本节则聚焦于直接控制 GPU，并用 Python 编写在 GPU 上运行的 kernel，采用 SIMT 编程模型。

<a id="sec-2-2-1"></a>

## 2.2.1 CUDA Python 生态

CUDA Python 是支持 Python GPU 计算的工具和库生态。以下内容根据 [CUDA Python GitHub 仓库](https://github.com/NVIDIA/cuda-python)中的完整列表整理；本节不要求使用全部组件。

| 类别 | 组件 | 用途 |
| --- | --- | --- |
| GPU 控制与库调用 | `cuda.core` | 提供 Python 风格的 memory、device 管理接口，相当于 CUDA runtime 在 C++ 中承担的角色。 |
| GPU 控制与库调用 | `cuda.compute` | 将 CCCL 的 GPU 加速函数提供给 Python。 |
| GPU 控制与库调用 | CuPy | 提供 GPU 加速的 NumPy 例程，以及存放在 GPU 上的 `ndarray` 容器。 |
| kernel 编写 | `cuda.lang` | 使用 Python 语言子集，以 SIMT 模型编写 kernel 和 device 函数的 DSL。 |
| kernel 编写 | `cuda.coop` | 提供 CCCL 中可在 device 调用的原语，可配合 `cuda.lang` 使用。 |
| kernel 编写 | `cuda.tile` | 使用 tile 编程模型编写 kernel 和 device 函数的 Python DSL。 |
| 其他组件 | `cuda.pathfinder` | 查找 Python 环境中已安装的 CUDA 组件。 |
| 其他组件 | `cuda.bindings` | CUDA Driver API、Runtime API、NVRTC、NVVM 等底层库的 Python binding。它通过 C API 的封装提供相关能力，而非 `cuda.core` 那样的原生 Python 风格接口。 |

<a id="sec-2-2-1-1"></a>

### 2.2.1.1 在 Python 中使用 CUDA 库

CUDA C++ 有丰富的库生态，许多 GPU 加速任务不需要直接编写 kernel。2006 年 CUDA C++ 刚推出时，可用库很少，开发者基本需要自行编写 GPU 代码；之后大量库逐步建立，使开发者无需编写或仅需编写很少的 GPU 代码就能使用 GPU。

CUDA Python 的发展方向相反：在能够直接用 Python 语法编写自定义 kernel 之前，CuPy 等库就已经提供 GPU 加速的计算与算法。这些库中的许多接口，是对 CUDA C++ GPU 代码的 Python 封装。

如今，只要 GPU 加速库能充分表达需求，通常应优先使用它们，因为其实现往往经过 GPU 专家调优。当库不可用或功能不足时，也可以像 C++ 一样，直接用 Python 编写 kernel 与 device 函数。

<a id="sec-2-2-1-2"></a>

### 2.2.1.2 本节范围

虽然应尽可能优先使用现成库，本节其余内容仍将介绍需要自定义 GPU 代码时的做法。与 [CUDA C++ 入门](/chapter-2/section-2-1#sec-2-1) 类似，先介绍如何定义 GPU kernel，再使用 CuPy 提供的 GPU 加速 `ndarray` 分配 GPU memory，并在 CPU 与 GPU 之间传输数据。

<a id="sec-2-2-1-3"></a>

### 2.2.1.3 环境准备

CUDA Python 生态的大多数组件可从 PyPI 获得，通过 `pip` 或常见 Python 包管理器安装。这些包要求系统安装适当版本的 NVIDIA driver；编写或运行 CUDA Python 应用程序通常不要求单独安装 CUDA Toolkit。

不同平台的安装和配置方法，参见 [NVIDIA Developer Zone 中的 CUDA Python 文档](https://developer.nvidia.com/how-to-cuda-python)。

<a id="sec-2-2-1-4"></a>

### 2.2.1.4 运行 CUDA Python 应用程序

无论使用 GPU 加速库还是自定义 GPU 代码，运行方式都与普通 Python 应用程序相同。本节统一使用命令行，通过 `python3` 执行 Python 文件：
```sh
$ python3 cuda-python-app.py
```

<a id="sec-2-2-2"></a>

## 2.2.2 Python 中的 SIMT kernel

如 [编程模型](/chapter-1/section-1-2#sec-1-2) 所述，GPU 上执行、可由 host 调用的函数称为 kernel。CUDA 支持 SIMT 和 tile 两种模型。SIMT kernel 由大量 thread 并行执行，这一概念在 Python 与 C++ 中相同。本节通过 SIMT kernel 介绍 CUDA Python。

<a id="sec-2-2-2-1"></a>

### 2.2.2.1 声明 kernel

首先导入 `numba.cuda`：
```python
from numba import cuda
```

这会导入包，并使用它提供的 `cuda` 命名空间。在函数定义上方放置 `@cuda.jit` decorator，即可将函数声明为 kernel。
```python
from numba import cuda
@cuda.jit
def function(input_array, output_array):
    ...
```

首次启动时，kernel 会针对当前 GPU 进行 JIT 编译。若没有显式选择其他 GPU，则使用默认 CUDA device；本节示例均采用这种方式。

<a id="sec-2-2-2-2"></a>

### 2.2.2.2 启动 kernel

启动时指定执行 kernel 的 thread 数量，也就是 execution configuration。不同调用可以采用不同的 block 大小或 block 数量。

<a id="sec-2-2-2-2-1"></a>

#### 2.2.2.2.1 kernel launch

在 kernel 名称后、函数实参前使用方括号 `[]` 放置 execution configuration。参数顺序与 [C++ 三尖括号语法](/chapter-2/section-2-1#sec-2-1-2-2-1) 相同：
```python
kernel_name[number_of_thread_blocks, threads_per_block](arguments, ...)
```

以下示例在一个 Python 文件中定义并调用 kernel。
```python
from numba import cuda
@cuda.jit
def my_kernel(input, output):
    ...
## launch the kernel
my_kernel[num_thread_blocks, threads_per_block](in_array, out_array)
```

同一 block 的所有 thread 都驻留在同一个 SM 并共享其资源，因此 block 大小有上限。本版指南所述 GPU 每个 block 最多支持 1024 个 thread。资源允许时，同一 SM 可以同时调度多个 block。

<a id="sec-2-2-2-2-2"></a>

#### 2.2.2.2.2 多维 grid 与 block

grid 和 block 都可以是一维、二维或三维。一维时用整数指定；二维或三维时用 tuple。下面启动一个二维 grid，其中 `gridX`、`gridY` 为 grid 的两个维度，`blockX`、`blockY` 为每个 block 的两个维度。
```python
from numba import cuda
@cuda.jit
def function(input, output):
    ...
## launch the kernel
function[(gridX, gridY), (blockX, blockY)](in_array, out_array)
```

<a id="sec-2-2-2-3"></a>

### 2.2.2.3 thread 与 grid 的索引内置变量

[block 与 grid](/chapter-1/section-1-2#sec-1-2-2-1) 介绍了组织方式，[启动 kernel](/chapter-2/section-2-2#sec-2-2-2-2) 介绍了如何配置维度。kernel 内每个 thread 可以读取以下变量，确定自身身份。

| 变量 | 含义 |
| --- | --- |
| `cuda.threadIdx.x/y/z` | 当前 thread 在 block 内的索引。 |
| `cuda.blockDim.x/y/z` | 启动时指定的 block 维度。 |
| `cuda.blockIdx.x/y/z` | 当前 block 在 grid 内的索引。 |
| `cuda.gridDim.x/y/z` | 启动时指定的 grid 维度。 |

未指定的维度默认大小为 1，对应索引为 0。`cuda.threadIdx` 与 `cuda.blockIdx` 从 0 开始，因此 `cuda.threadIdx.x` 范围为 `0` 到 `cuda.blockDim.x - 1`，其他维度同理。

下面的 kernel 将三个数组作为参数，执行逐元素向量加法 `C = A + B`。
```python
# C = A + B vector addition
@cuda.jit
def vecadd(A, B, C):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    C[idx] = A[idx] + B[idx]
```

kernel 首先计算 thread 在 grid 中的唯一索引。这里假设使用一维 block 和一维 grid，`idx` 范围为 `0` 到 `N - 1`，其中 `N = cuda.gridDim.x * cuda.blockDim.x`，即整个 grid 的 thread 总数。

这种计算非常常见，因此 Numba 提供 `cuda.grid(n)` 简写，其中 `n` 为维数。下面这行：
```python
idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
```

可替换为：
```python
idx = cuda.grid(1)
```

这个 kernel 没有显式检查 `A`、`B`、`C` 的访问是否越界。原文在这里假定参数是 CuPy 创建的 `ndarray`，并将边界检查归于数组类型本身；该说法的译注与边界检查说明见 [ndarray 对象类型](/chapter-2/section-2-2#sec-2-2-3-3)。

<a id="sec-2-2-3"></a>

## 2.2.3 GPU 计算中的 memory

::: tip 注意
CuPy 等 Python 包通过 CUDA C++ API 管理 GPU memory，例如 [显式 memory 管理](/chapter-2/section-2-1#sec-2-1-3-2) 中的 API。多个 Python 包都提供相关封装，本指南只使用 CuPy。除特别说明外，基本概念及行为与相应 C++ 接口类似。
:::

如 [GPU memory](/chapter-1/section-1-2#sec-1-2-3) 所述，GPU 有直接连接的 DRAM。kernel 使用的数据数组通常应在访问前放入 GPU DRAM。Python 中控制数据位置，以及在 CPU、GPU 之间移动数据，仍由程序员负责，与 C++ 中的显式 memory 管理相同。

<a id="sec-2-2-3-1"></a>

### 2.2.3.1 在 GPU 上创建数组

CuPy 可以在 GPU 上创建指定类型、维度的 `ndarray`，并在 CPU、GPU 之间复制。许多函数的签名与 NumPy 相近。以下示例展示在 GPU memory 中创建和填充数组。
```python
import cupy as cp
import numpy as np
## create a matrix of zeros on the GPU
## when a datatype is not specified, float32 is used by default
A_device = cp.zeros((1024, 1024))
## create an array of 2^20 random doubles on the GPU
B_device = cp.rand.random((2**20), dtype=np.double)
## create an array of zeroes with the same shape and datatype as an existing array
C_device = cp.zeros_like(A)
```

<a id="sec-2-2-3-2"></a>

### 2.2.3.2 在 host memory 与 GPU memory 之间复制数组

CuPy 可以将 CPU memory 中的 NumPy `ndarray` 复制为 GPU memory 中的 CuPy 数组，反之亦然。
```python
import cupy as cp
import numpy as np
## Create an array in host memory
A_host = np.zeros((1024, 1024))
## Copy the array to the GPU
A_device = cp.array(A_host)
## Create an array in GPU memory
B_device = cp.rand.random((1024, 1024))
## copy the array to host memory
B_host = cp.asnumpy(B_device)
```

<a id="sec-2-2-3-3"></a>

### 2.2.3.3 ndarray 对象类型

前面的数组分别位于 host memory 或 GPU memory，并不会同时存在于两处。原文指出，把 host 数组直接交给 GPU kernel，或把 GPU 数组直接交给要求 host 数组的普通 Python 函数，会导致错误。

CuPy 不会隐式地在 CPU、GPU 之间复制，因为复制代价可能很高，过多复制会损害性能。程序员需要明确决定何时传输数据。

`ndarray` 的一个优势是携带自身维度大小。原文据此声称，即使所需 thread 数略少于启动数量，也不必显式检查越界。

<a id="python-bounds-note"></a>

::: warning 译注：原文的边界检查描述需要修正
数组携带 shape，并不意味着 Numba CUDA kernel 会自动检查索引。对于这里的 `@cuda.jit` kernel，若 launch 产生的 thread 可能超出数组长度，应显式检查，例如 `if idx < A.size:`，再访问数组。[Numba CUDA 官方 kernel 指南](https://nvidia.github.io/numba-cuda/user/kernels.html) 也要求在不能保证索引有效时进行边界判断。本页代码保留 PDF 原例；其中完整向量示例选择了能被 block 大小整除的长度。
:::


<a id="sec-2-2-4"></a>

## 2.2.4 同步 CPU 与 GPU

与 C++ 一样，Python kernel launch 对 host thread 也是异步的。启动后 CPU 继续执行，不能保证 GPU 已完成甚至已开始工作。要确认 kernel 已完成，host 必须与 GPU 同步。

最简单的方法是整个 device 同步。CUDA driver 提供这一操作，CuPy 和 `numba.cuda` 都将它封装为 `synchronize()`：
```python
import cupy as cp
from numba import cuda
...
## Wait on host thread for all pending GPU work to complete
## this uses the interface provided by cupy
cp.synchronize()
## Wait on host thread for all pending GPU work to complete
## this uses the interface provided by numba.cuda
cuda.synchronize()
```

device 范围的同步会等待 GPU 上先前提交的全部工作。更精细的方法是 [CUDA stream](/chapter-2/section-2-5#sec-2-5)。Python 中使用 stream 时，推荐通过 `cuda.core` 创建 stream，并只在必要时同步特定 stream。

<a id="sec-2-2-5"></a>

## 2.2.5 完整示例

下面给出 Python 版本的并行向量加法完整示例。
```python
import numpy as np
from numba import cuda
import cupy as cp
## Defines a CUDA kernel to perform C = A + B vector addition
@cuda.jit
def vecadd(A, B, C):
    work_index = cuda.grid(1)
    C[work_index] = A[work_index] + B[work_index]
# note that vector size is not a power of 2 nor a multiple of the block_size defined below
vector_size = 2**24 + 11
device = cp.cuda.Device()
## Create device arrays of uniform random float32 values as input, and an array of zeros
## as the result vector
a = cp.random.uniform(-1, 1, vector_size)
b = cp.random.uniform(-1, 1, vector_size)
c = cp.zeros_like(a)
block_size = 256
grid_size = int(np.ceil(vector_size/block_size))
vecadd[grid_size, block_size](a, b, c)
## synchronize the CPU thread and the GPU to ensure that the kernel has completed
## this is included to illustrate good practices, even though the copy below would implicitly wait for
## the kernel to complete
device.synchronize()
## Copy all 3 arrays to the CPU as ndarrays
a_np = cp.asnumpy(a)
b_np = cp.asnumpy(b)
c_np = cp.asnumpy(c)
## Perform the copy on the CPU to verify the answer
expected = a_np + b_np
## Test that the answer is correct, within floating point epsilon
np.testing.assert_array_almost_equal(c_np, expected)
## The assert will print diagnostics and abort
## so this only prints if the assertion passes
print("Test succeeded")
```

输入数组 `A`、`B` 由 CuPy 在 GPU 上创建并用随机数初始化。代码末尾将它们复制回 CPU，只是为了在 CPU 上也执行一次加法，以验证 CPU 与 GPU 结果是否一致。

<a id="sec-2-2-6"></a>

## 2.2.6 CUDA Python 错误检查

任何影响 GPU 的操作，包括 memory allocation、复制和 kernel launch，都可能发生错误。与 [C++ 错误检查](/chapter-2/section-2-1#sec-2-1-7) 一样，应检查与 GPU 交互时是否出现异常。

Python 中的 CUDA 错误通常表现为 exception，未捕获时会终止程序，可用标准 Python 语法捕获。下面故意将每个 block 的 thread 数设为 2048，超过本版指南所述 GPU 的上限，导致 launch 失败并抛出异常。
```python
import numpy as np
from numba import cuda
import cupy as cp
## Defines a CUDA kernel to perform C = A + B vector addition
@cuda.jit
def vecadd(A, B, C):
    work_index = cuda.grid(1)
    C[work_index] = A[work_index] + B[work_index]
try:
    vector_size = 2**24 + 11
    device = cp.cuda.Device()
    a = cp.random.uniform(-1, 1, vector_size)
    b = cp.random.uniform(-1, 1, vector_size)
    c = cp.zeros_like(a)
    ## this block size is too large for any current GPUs
    block_size = 2048
    grid_size = int(np.ceil(vector_size/block_size))
    # Error: launching kernel with invalid block size
    vecadd[grid_size, block_size](a, b, c)
    device.synchronize()
    print("Test did not encounter any errors")
except Exception as e:
    print(f"Exception occurred: {e}")
```

运行时，错误被捕获并显示：
```sh
$ python3 vecadd_error.py
Exception occurred: CUDA_ERROR_INVALID_VALUE: This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values.
```

捕获异常后，程序正常结束。如果没有 `try` 和 `except`，程序将异常退出，并向控制台打印包含相同错误信息的 traceback。
