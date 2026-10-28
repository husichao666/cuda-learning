---
title: "2.4 编写 tile kernel"
description: "CUDA Programming Guide 13.4.2 · 第二章中文译文"
---

<a id="sec-2-4"></a>

# 2.4 编写 tile kernel

> 来源：《CUDA Programming Guide》13.4.2，书内第 74–104 页相关内容。正文为中文，常用 CUDA 与 memory 术语保留英文；代码保留原文，恢复 PDF 断行与缩进。


CUDA Tile 提供不同于 SIMT 的 GPU kernel 编程方式。程序员以 tile 为单位表达并行性，将底层 thread 并行化交给编译器和内置操作，从而更容易使用 Tensor Memory Accelerator（TMA）、Tensor Core 等 GPU 性能特性。

- Python 通过 cuTile Python 包 `cuda.tile` 使用 CUDA Tile。
- CUDA Tile C++ 从 CUDA Toolkit 13.3 开始提供。

kernel 外围的应用程序代码，例如分配 device memory、host/device 传输及安排 launch 顺序，与前面 SIMT 示例相同。tile kernel 使用标准 CUDA API 分配的 global memory，结果也以相同方式复制回 host。改变的是 kernel 内部的编程方式。

SIMT 中，程序员计算每个 thread 的全局索引，加载其数据、计算并写回。Tile 中，程序员站在整个 block 的层面：加载包含许多元素的 tile，对整个 tile 执行操作，再存储结果。编译器负责将这些操作映射到 block 内的硬件 thread。

本节只讨论这部分区别，即入口函数和内部 tile 操作。示例同时给出 cuTile Python（`cuda.tile`）与 CUDA Tile C++（`cuda::tiles`）；它们共享 CUDA Tile IR 编译后端，因此具有相同的执行语义。

两种语言都约定将 tile API 简写为 `ct`：Python 使用 `import cuda.tile as ct`，C++ 使用 `namespace ct = cuda::tiles`。C++ 接口由 `cuda_tile.h` 提供。
```cpp
#include "cuda_tile.h"
namespace ct = cuda::tiles;
```

下文的 `ct.` 或 `ct::` 分别表示对应语言中的 tile API。完整参考见 [CUDA Tile C++ API Reference](https://docs.nvidia.com/cuda/cuda-tile-cpp-api-reference/api_reference.html) 和 [cuTile Python API Reference](https://docs.nvidia.com/cuda/cutile-python/operations.html)。

<a id="sec-2-4-1"></a>

## 2.4.1 kernel 与函数声明

Tile kernel 是 GPU 入口，在 launch grid 的每个 block 上执行一次。Tile function 则由 tile kernel 或其他 tile function 调用，本身不是入口。与 SIMT kernel 一样，不能把 tile kernel 当作普通 host 函数直接调用，必须 launch。

C++ 中：

- `__tile_global__` 对应 SIMT 的 `__global__`，标记 tile kernel 入口。
- `__tile__` 对应 `__device__`，标记编译为 GPU 代码、可由其他 tile 函数调用的函数。

数组与标量参数的传递方式与 SIMT kernel 相同。同一个 `.cu` 文件可以同时定义 `__tile_global__` 与 `__global__` kernel，同一 host 程序也可以启动二者。

::: tip 注意
本版中，`__global__`、`__device__` 函数不能调用 `__tile__` 函数；`__tile_global__`、`__tile__` 函数也不能调用 `__device__` 函数。未来版本可能放宽此限制。
:::

Python 中：

- `@ct.kernel` 标记 tile kernel 入口。
- `@ct.function` 标记可从 tile kernel 或 tile function 调用的函数。

实际上，kernel 调用的函数会自动编译为 tile code，因此 `@ct.function` 可省略。数组参数可以是实现 DLPack 或 CUDA Array Interface 的 device-resident 数组，例如 PyTorch tensor、CuPy array；标量直接传递。

C++：
```cpp
#include "cuda_tile.h"
// Tile kernel entry point. Cannot be called directly; must be launched.
__tile_global__ void my_kernel(float* a, float* b, float* c) {
    ...
}
// Tile function. Callable from tile kernels and tile functions.
__tile__ float helper(float x, float y) {
    return x + y;
}
```

Python：
```python
import cuda.tile as ct
# Tile kernel entry point. Cannot be called directly; must be launched.
@ct.kernel
def my_kernel(a, b, c):
    ...
# Tile function. Callable from tile kernels and tile functions.
# @ct.function is optional, any function called from tile code
# is automatically compiled as tile code.
@ct.function
def helper(x, y):
    return x + y
```

<a id="sec-2-4-2"></a>

## 2.4.2 启动 kernel

与 SIMT 一样，tile kernel 在 block grid 上启动。程序员指定最多三维的 grid 形状。从编程视角看，每个 tile block 由一个逻辑 thread 执行，block 内硬件并行由编译器管理。

C++ 复用三尖括号：第一个参数指定 tile block 数量；第二个参数必须为 `1`，实际硬件 thread 数量由编译器决定。也可以通过现有 `cudaLaunchKernel`、`cudaLaunchKernelEx` API，以相同的 `grid, 1` 配置启动，方便接入已有 launch 代码。

Python 的 `ct.launch` 有四个位置参数：CUDA stream、grid tuple、kernel 对象、kernel 实参 tuple。

C++：
```cpp
my_kernel<<<dim3(num_blocks_x, num_blocks_y), 1>>>(a, b, c); // second arg must be 1
```

Python：
```python
import torch
stream = torch.cuda.current_stream() # CUDA stream object
grid = (num_blocks_x, num_blocks_y, 1) # tile-block grid (x, y, z)
ct.launch(stream, grid, my_kernel, (a, b, c))
```

<a id="sec-2-4-2-1"></a>

### 2.4.2.1 grid 大小的计算模式

通常启动足够多的 block 覆盖整个数组，允许最后一个 block 在某些维度超出数组边界。

C++：
```cpp
int num_blocks = (N + tile_size - 1) / tile_size; // ceil division -> covers partial tail
kernel<<<num_blocks, 1>>>(in, out, N);
```

Python：
```python
import math
grid = (math.ceil(N / TILE),) # ceil division -> covers partial tail
ct.launch(stream, grid, my_kernel, (arr_in, arr_out, TILE))
```

数组大小不能整除 tile 大小时，边缘 tile 的处理见 [tile 的 load 与 store](/chapter-2/section-2-4#sec-2-4-6)。

<a id="sec-2-4-3"></a>

## 2.4.3 查询 block 位置

每个 block 需要知道自己在 grid 中的位置，以确定处理哪部分数据。SIMT 组合 `blockIdx` 和 `threadIdx` 计算全局 thread 索引；tile code 只需要 block 索引，block 内索引由编译器管理。

C++ 的 `ct::bid()` 返回包含三维 block 索引的 `uint3`；`ct::num_blocks()` 返回各维 block 总数的 `dim3`，使用 `.x`、`.y`、`.z` 访问。

Python 的 `ct.bid(axis)` 返回指定轴 `0`、`1`、`2` 上的 block 索引，为 `int32` 标量；`ct.num_blocks(axis)` 返回该轴的 block 总数，可用于边界判断与循环次数计算。

C++：
```cpp
#include "cuda_tile.h"
__tile_global__ void my_kernel(float* a, float* b, float* c) {
    namespace ct = cuda::tiles;
    int bid_x = ct::bid().x; // block index along .x
    int bid_y = ct::bid().y; // block index along .y
    int num_x = ct::num_blocks().x; // total blocks along .x
}
```

Python：
```python
@ct.kernel
def my_kernel(a, b, c):
    bid_x = ct.bid(0) # block index along axis 0
    bid_y = ct.bid(1) # block index along axis 1
    num_x = ct.num_blocks(0) # total blocks along axis 0
```

<a id="sec-2-4-4"></a>

## 2.4.4 创建 tile

Tile 是固定大小的多维标量集合，形状和元素类型在编译时已知，每个维度必须是 2 的幂。Tile 具有值语义：复制后两份数据在语义上互相独立。由于内部硬件表示由编译器控制，复制通常很便宜。程序员无需为 tile 显式分配、释放 memory。

Tile 可通过 [从数组加载](/chapter-2/section-2-4#sec-2-4-6-1) 创建，也可用 factory 函数生成指定模式。

C++ 类型写作 `ct::tile<T, ct::shape<dims...>>`，其中 `T` 是元素类型，维度以模板参数编码。例如 `ct::tile<float, ct::shape<8>>` 是 8 元素的一维 tile，`ct::tile<float, ct::shape<4, 4>>` 是 `4 × 4` tile。形状是类型的一部分，因此编译时必然已知。

Factory 函数将完整 tile 类型作为模板参数：

| C++ | 含义 |
| --- | --- |
| `ct::zeros<Tile>()` / `ct::ones<Tile>()` | 全 0 / 全 1 |
| `ct::full<Tile>(val)` | 每个元素均为 `val` |
| `ct::iota<Tile>()` | 生成 `0, 1, ..., N-1` |

为简化调用，本节 C++ 常用 `using` 定义类型别名，例如 `using f32x4x4 = ct::tile<float, ct::shape<4, 4>>`。

Python factory 的 shape tuple 和 `dtype` 也必须在编译时确定。`(64, 64)`、`ct.float32` 这样的字面量自然符合，也可通过 [Constant 参数](/chapter-2/section-2-4#sec-2-4-5-1) 传入。结果 tile 的 `.shape`、`.dtype`、`.ndim` 反映这些编译时属性。

| Python | 含义 |
| --- | --- |
| `ct.zeros(shape, dtype)` / `ct.ones(shape, dtype)` | 全 0 / 全 1 |
| `ct.full(shape, fill_value, dtype)` | 填充指定常量 |
| `ct.arange(size, dtype=...)` | 一维序列 `0, 1, ..., size-1` |

C++：
```cpp
#include "cuda_tile.h"
__tile__ void factories() {
    namespace ct = cuda::tiles;
    using i32x8 = ct::tile<int, ct::shape<8>>; // 1-D: 8 ints
    using f32x4x4 = ct::tile<float, ct::shape<4, 4>>; // 2-D: 4x4 floats
    auto z = ct::zeros<f32x4x4>(); // all zeros
    auto o = ct::ones<f32x4x4>(); // all ones
    auto filled = ct::full<f32x4x4>(3.14f); // all 3.14
    auto seq = ct::iota<i32x8>(); // {0, 1, 2, 3, 4, 5, 6, 7}
}
```

Python：
```python
import cuda.tile as ct
@ct.function
def factories():
    zeros = ct.zeros((64, 64), dtype=ct.float32) # 64x64 tile of 0.0
    ones = ct.ones((128,), dtype=ct.float16) # 128-element tile of 1.0
    filled = ct.full((32, 32), 3.14, dtype=ct.float32) # 32x32 tile of 3.14
    seq = ct.arange(8, dtype=ct.int32) # [0, 1, 2, 3, 4, 5, 6, 7]
```

<a id="sec-2-4-5"></a>

## 2.4.5 编译时常量

Tile 编译器会针对不同形状、类型及结构参数组合生成特化机器码。影响代码结构的值必须在编译时确定，包括 tile 的形状和元素类型。

前面使用字面量，例如 `ct.zeros((64, 64), dtype=ct.float32)` 与 `ct::tile<int, ct::shape<8>>`；也可以通过 kernel 参数接口传入编译时已知值。

<a id="sec-2-4-5-1"></a>

### 2.4.5.1 Python Constant[T]

在参数上使用 `ct.Constant[T]`，表示将它作为常量嵌入：kernel 中使用该参数，等价于在对应位置直接写入字面量。类型参数 `T` 可省略，`ct.Constant` 可以嵌入任意类型常量。

最常见的形式是 `ct.Constant[int]`，用于 tile 形状或循环边界。
```python
import cuda.tile as ct
@ct.kernel
def my_kernel(TILE: ct.Constant[int]):
    # TILE is constant-embedded: wherever TILE appears, the compiler sees its
    # literal value (e.g., 128) and generates specialized code. Here TILE drives
    # the shape of a factory-built tile.
    zeros = ct.zeros((TILE,), dtype=ct.float32)
```

<a id="sec-2-4-5-2"></a>

### 2.4.5.2 C++ integral_constant 与 _ic 字面量

C++ 使用 `ct::integral_constant`，将数值编码在类型中。`ct::literals` 中的 `_ic` 字面量提供简写，例如 `0_ic` 产生 `ct::integral_constant<0>`。

接受编译时值的 API，通常同时支持 non-type template parameter（NTTP）和 `_ic` 参数形式。例如 `ct::cat` 的拼接轴必须编译时已知，以下两行仅写法不同。
```cpp
#include "cuda_tile.h"
__tile__ void concat_demo() {
    namespace ct = cuda::tiles;
    using namespace ct::literals;
    using T = ct::tile<int, ct::shape<4, 8>>;
    T lhs = ct::full<T>(0);
    T rhs = ct::full<T>(1);
    auto a = ct::cat<0>(lhs, rhs); // NTTP form
    auto b = ct::cat(lhs, rhs, 0_ic); // _ic form
}
```

`ct::extents` 与 `ct::shape` 都有 NTTP 形式，例如 `ct::extents<std::uint32_t, 4, 8>`，也有花括号形式。后者允许混合运行时值和编译时值：已知维度用 `_ic`，launch 时才知道的维度用普通变量。`ct::tensor_span`、`ct::partition_view` 可以用这种方式描述数组。
```cpp
auto shape2d = ct::extents{8_ic, length}; // 8 is compile-time; length is runtime
```

因此，只要某个值形式的 API 参数要求编译时常量，例如拼接轴或 extents、shape 的分量，都可以使用 `_ic` 简写。

<a id="sec-2-4-6"></a>

## 2.4.6 tile 的 load 与 store

如 [数组与 tile](/chapter-1/section-1-2#sec-1-2-2-3-1) 所述，array 位于 global memory，对全部 tile block 可见；tile 只存在于单个 block 的 tile code 中，通常保存 array 的一个子集。本节讨论从 array 加载到 tile，以及将 tile 写回 array。

两类访问方式如下：

- [Tile-space load/store](/chapter-2/section-2-4#sec-2-4-6-1)：通过 view 对象规定 array 到 tile 的规则映射，再用 tile-space 索引访问。
- [Gather/scatter](/chapter-2/section-2-4#sec-2-4-6-2)：用索引 tile 或指针 tile，分别指定每个元素的来源或目标位置。

在受支持硬件上，编译器可将 tile-space load 映射到 TMA，通常比逐元素 gather 更快。C++ 的相关优化见 [性能建议](/chapter-2/section-2-4#sec-2-4-12)。

加载边缘 tile 时，程序员需要确定越界元素的填充值。Python store 默认丢弃越界写入；C++ 使用 masked 版本时也如此。

<a id="sec-2-4-6-1"></a>

### 2.4.6.1 Tile-space load/store

首先用 view 对象定义数组如何划分为 tile 大小的区域网格，这个映射就是 tile space。kernel 每次按 tile-space 索引加载或存储一个区域。

[图 19](/chapter-2/section-2-4#figure-19) 的 partition view 将数组划分为固定大小、不重叠且没有间隙的 tile。

<a id="figure-19"></a>

![图 19：PDF 原图 19](/images/chapter-02/figure-19.png)

*图 19：partition view 的 tile-space 索引。形状 `(10, 16)` 的数组按 `(2, 4)` tile 划分，得到 `(5, 4)` 的 tile grid。每格标注索引 `(i, j)`；高亮的 `(1, 2)` 覆盖数组元素 `(2, 8)` 到 `(3, 11)`。*

数组维度不能被 tile 大小整除时，边缘 tile 只有部分元素有效，见 [边界处理](/chapter-2/section-2-4#sec-2-4-6-1-3)。

Partition view 只是 tile-space 的一种；两种语言还支持：

- 固定形状、按指定步长排列的块，块间可有间隙或重叠。C++ 为 `ct::strided_view`，Python 为 `Array.tiled_view` 的 `traversal_steps` 参数。
- 按某个维度上的索引 tile 选择数据块，用于不规则访问。C++ 为 `ct::gather_scatter_view`，Python 为 `ct.load_advanced_indexing`、`ct.store_advanced_indexing`。

本节只演示 partition view，它也是 `Array.tiled_view` 的默认结果。其他 view 详见各语言 API Reference。

<a id="sec-2-4-6-1-1"></a>

#### 2.4.6.1.1 Partition view 的 load/store

结构化 tile-space 访问是 global memory 与 tile 之间移动数据的首选方式。先构造 view，再用 tile-space 索引逐 tile 访问。

C++ 分为两步：

1. `ct::tensor_span` 将原始指针与 `ct::extents` 结合，赋予多维结构。
2. `ct::partition_view` 将 span 划分为固定大小的 tile grid，提供 `.load(idx...)` 与 `.store(tile, idx...)`。

Python 的 `Array.tiled_view(tile_shape)` 返回 `TiledView`，提供 `.load(index)`、`.store(index, tile)`，与 C++ 的 partition view 对应。

::: tip 注意
本节 C++ 指针使用 `__restrict__`，并在 kernel 前部调用 `ct::assume_aligned(ptr, 16_ic)`，它们是重要的性能注解，见 [C++ 性能建议](/chapter-2/section-2-4#sec-2-4-12)。数字后的 `_ic` 表示 [编译时常量](/chapter-2/section-2-4#sec-2-4-5)。
:::

C++：
```cpp
__tile_global__ void vec_add(float* __restrict__ a, float* __restrict__ b, float* __restrict__ out) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;
    a = ct::assume_aligned(a, 16_ic);
    b = ct::assume_aligned(b, 16_ic);
    out = ct::assume_aligned(out, 16_ic);
    // Step 1: attach a shape to each raw pointer. 128_ic marks 128 as a compile-time constant.
    auto aSpan = ct::tensor_span{a, ct::extents{128_ic}};
    auto bSpan = ct::tensor_span{b, ct::extents{128_ic}};
    auto oSpan = ct::tensor_span{out, ct::extents{128_ic}};
    // Step 2: partition each span into a tile space of fixed 8-element tiles.
    auto aView = ct::partition_view{aSpan, ct::shape{8_ic}};
    auto bView = ct::partition_view{bSpan, ct::shape{8_ic}};
    auto oView = ct::partition_view{oSpan, ct::shape{8_ic}};
    int bx = ct::bid().x; // this block's tile-space index along .x
    auto aTile = aView.load(bx); // pick the bx-th tile of a
    auto bTile = bView.load(bx);
    oView.store(aTile + bTile, bx); // write the tile back at the bx-th position of out
}
```

Python：
```python
@ct.kernel
def vec_add(a, b, c, TILE: ct.Constant[int]):
    a_view = a.tiled_view((TILE,))
    b_view = b.tiled_view((TILE,))
    c_view = c.tiled_view((TILE,))
    bid = ct.bid(0)
    a_tile = a_view.load((bid,))
    b_tile = b_view.load((bid,))
    c_view.store((bid,), a_tile + b_tile)
```

<a id="sec-2-4-6-1-2"></a>

#### 2.4.6.1.2 Python 单次调用形式的 load/store

Python 还可以不显式创建 view，直接在每次操作中指定形状：`ct.load(array, index, shape)` 读取 tile，`ct.store(array, index, tile)` 写回。

它与 `Array.tiled_view` 表达相同的访问模式，差别只是 tile 形状绑定的位置：view 在创建时绑定一次，直接调用则每次指定。多次复用同一划分时优先使用 view；一次性的简单加载可直接调用。
```python
@ct.kernel
def vec_add(a, b, c, TILE: ct.Constant[int]):
    bid = ct.bid(0) # this block's tile-space index along axis 0
    a_tile = ct.load(a, index=(bid,), shape=(TILE,)) # (index, shape) = pick the bid-th TILE-sized region of a
    b_tile = ct.load(b, index=(bid,), shape=(TILE,))
    ct.store(c, index=(bid,), tile=a_tile + b_tile) # write the tile back to the bid-th region of c
```

<a id="sec-2-4-6-1-3"></a>

#### 2.4.6.1.3 Tile-space 边界处理

C++ `partition_view` 提供 unmasked 与 masked 两种版本。

| 操作 | 边界行为 |
| --- | --- |
| `.load(idx...)` / `.store(tile, idx...)` | 假定整个 tile 均在边界内；部分越界是未定义行为。 |
| `.load_masked(idx...)` | 安全处理部分越界，默认补零，也可选择其他 padding，例如浮点 NaN。 |
| `.store_masked(tile, idx...)` | 丢弃越界位置的写入。 |

数组能被 tile 完全整除时，优先使用 unmasked；需要处理边界时，masked 也可以用于完全有效的 tile。

这里的 C++ 示例首次使用运行时数组维度。`ct::extents{N}` 接受运行时值，也可以与 `_ic` 常量混合，使 span、partition view 描述 launch 时才确定大小的数组。

Python 的 `ct.load` 通过 `padding_mode` 决定越界位置的值：

- `PaddingMode.UNDETERMINED`：默认值。仍执行边界检查，不会读取数组外 memory，但越界位置返回何值由实现决定。适用于这些位置永远不会被使用的情况。
- `PaddingMode.ZERO`：越界位置填零。

`ct.store` 无需 `padding_mode`，默认检查边界并丢弃越界写入。`tiled_view` 采用相同规则，并在创建 view 时固定 padding 模式。

`ct.load`、`ct.store` 还接受默认值为 `True` 的 `check_bounds`。Padding 决定越界位置的值，而 `check_bounds` 决定是否检查边界。设置为 `False` 后，越界可能读取或覆盖相邻 memory，属于未定义行为，相当于 C++ unmasked 形式。`TiledView.load/store` 也接受该参数。

C++：
```cpp
__tile_global__ void edge_safe(float* __restrict__ in, float* __restrict__ out, int N) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;
    in = ct::assume_aligned(in, 16_ic);
    out = ct::assume_aligned(out, 16_ic);
    // ct::extents{N} uses a runtime dimension; 128_ic stays compile-time.
    auto inView = ct::partition_view{ct::tensor_span{in, ct::extents{N}}, ct::shape{128_ic}};
    auto outView = ct::partition_view{ct::tensor_span{out, ct::extents{N}}, ct::shape{128_ic}};
    int bx = ct::bid().x;
    auto tile = inView.load_masked(bx); // masked load: OOB lanes default to 0
    outView.store_masked(tile, bx); // masked store: OOB writes silently discarded
}
```

Python：
```python
@ct.kernel
def edge_safe(arr_in, arr_out, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tile = ct.load(arr_in, index=(bid,), shape=(TILE,),
                   padding_mode=ct.PaddingMode.ZERO) # OOB lanes of a partial edge tile become 0
    ct.store(arr_out, index=(bid,), tile=tile) # OOB writes are silently discarded
```

C++ 使用 `.load_masked()`、`.store_masked()` 处理边缘；Python load 使用零填充，store 默认丢弃越界写入。完整模式见各语言 API Reference。

::: warning 边界范围
这里讨论的是部分越界的边缘 tile。加载或存储一个完全位于数组之外的 tile，原文规定为未定义行为。
:::

<a id="sec-2-4-6-2"></a>

### 2.4.6.2 Gather 与 scatter

Partition view 表达规则、对齐的数据划分。当访问取决于数据或不规则索引，例如查表、置换时，可用 gather/scatter 访问数组中非均匀、非连续的位置。

Python 使用索引 tile，传给 `ct.gather()`、`ct.scatter()`，默认带边界检查。C++ 使用指针 tile，传给 `ct::load()`、`ct::store()`；masked 版本接收布尔 mask tile。

C++ 中，标量指针加整数 tile 会逐元素生成指针 tile，每个元素指向一个目标地址，是构造 gather/scatter 的常用方式。

Python 的 `ct.gather` 对各索引逐元素读取，越界默认补零，可通过 `padding_value=` 修改；`ct.scatter` 对各索引逐元素写入，丢弃越界写入。二者均可用 `check_bounds=False` 关闭检查。

C++：
```cpp
__tile_global__ void vec_add_gather(int* __restrict__ a, int* __restrict__ b, int* __restrict__ out) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;
    using i32x8 = ct::tile<int, ct::shape<8>>;
    a = ct::assume_aligned(a, 16_ic);
    b = ct::assume_aligned(b, 16_ic);
    out = ct::assume_aligned(out, 16_ic);
    int bx = ct::bid().x;
    auto offsets = 8 * bx + ct::iota<i32x8>(); // element-level offsets, one per lane
    // scalar pointer + int tile = tile of pointers (one pointer per offset).
    auto aPtrs = a + offsets;
    auto bPtrs = b + offsets;
    auto aTile = ct::load(aPtrs); // gather: one load per pointer
    auto bTile = ct::load(bPtrs);
    ct::store(out + offsets, aTile + bTile); // scatter: one store per pointer
}
```

Python：
```python
@ct.kernel
def vec_add_gather(a, b, c, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    indices = bid * TILE + ct.arange(TILE, dtype=ct.int32) # one element index per lane
    a_tile = ct.gather(a, indices) # load a[indices[i]] per lane
    b_tile = ct.gather(b, indices)
    ct.scatter(c, indices, a_tile + b_tile) # store one value per index into c
```

<a id="sec-2-4-6-2-1"></a>

#### 2.4.6.2.1 Gather/scatter 边界处理

Python 默认保证边界安全：越界读返回 padding，默认零；越界写丢弃。只有能证明全部索引有效时才应关闭检查，否则越界属于未定义行为。其他 mask、padding 参数见对应 API Reference。

C++ 不自动检查边界。程序员需要构造布尔 mask，例如比较 offset 与数组长度，再传给 `ct::load_masked`、`ct::store_masked`。
```cpp
__tile_global__ void gather_safe(int* __restrict__ arr, int* __restrict__ out, int N) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;
    using i32x8 = ct::tile<int, ct::shape<8>>;
    arr = ct::assume_aligned(arr, 16_ic);
    out = ct::assume_aligned(out, 16_ic);
    int bx = ct::bid().x;
    auto offsets = 8 * bx + ct::iota<i32x8>(); // element-level offsets, one per lane
    auto mask = offsets < N; // boolean tile: true where the offset is in-bounds
    auto ptrs = arr + offsets; // tile of pointers, one per offset
    auto tile = ct::load_masked(ptrs, mask, 0); // masked lanes get the pad value 0
    ct::store_masked(out + offsets, tile, mask); // masked lanes are skipped on the store
}
```

<a id="sec-2-4-7"></a>

## 2.4.7 控制流

从程序员视角，每个 tile block 只有一条控制流路径。条件和循环边界由标量决定，循环体中的 tile 操作由编译器分配到硬件 thread。

并非所有控制流结构都受支持，例如不能从循环体内直接 `return`。完整限制见 [CUDA Tile C++ general principles](https://docs.nvidia.com/cuda/cuda-tile-cpp-api-reference/general_principles.html) 和 [cuTile Python control flow](https://docs.nvidia.com/cuda/cutile-python/execution.html#control-flow) 文档。

<a id="sec-2-4-7-1"></a>

### 2.4.7.1 循环

常见模式是依次处理数组中的多个 tile。C++ 的 `ct::irange` 表示从下界开始、到上界之前结束、具有可选步长的递增整数范围。通过 range-for 遍历它，可以将结构化边界信息提供给编译器以便优化；需要将循环变量直接绑定到该 range-for，相关优化才能生效。

Python 支持 `range()`、`for`、`while` 及嵌套循环。Range 的步长必须严格为正，不支持负步长。

以下 kernel 仅用一个 block，遍历并求和一维数组中的所有 tile。

C++：
```cpp
__tile_global__ void tile_sum(float* __restrict__ arr, float* __restrict__ out, int num_tiles) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;
    using f32x8 = ct::tile<float, ct::shape<8>>;
    arr = ct::assume_aligned(arr, 16_ic);
    out = ct::assume_aligned(out, 16_ic);
    auto inView = ct::partition_view{ct::tensor_span{arr, ct::extents{8 * num_tiles}},
                                      ct::shape{8_ic}};
    auto outView = ct::partition_view{ct::tensor_span{out, ct::extents{8_ic}},
                                      ct::shape{8_ic}};
    auto acc = ct::full<f32x8>(0.0f);
    // range-for over ct::irange gives the compiler structured iteration bounds.
    for (auto k : ct::irange(0, num_tiles)) {
        auto tile = inView.load(k);
        acc = acc + tile; // accumulate the k-th tile into acc
    }
    outView.store(acc, 0); // write the final result as the 0-th tile of out
}
```

Python：
```python
@ct.kernel
def tile_sum(arr, out, TILE: ct.Constant[int], N_TILES: ct.Constant[int]):
    # Intended grid: (1,) -- a single block sums all tiles of arr.
    acc = ct.zeros((TILE,), dtype=ct.float32)
    for k in range(N_TILES): # range() works natively in tile code
        tile = ct.load(arr, index=(k,), shape=(TILE,))
        acc = acc + tile # accumulate the k-th tile into acc
    ct.store(out, index=(0,), tile=acc) # write the final result as the 0-th tile of out
```

<a id="sec-2-4-7-2"></a>

### 2.4.7.2 条件语句

标准 `if/else` 可以正常使用。由于一个 block 只有单一控制流路径，不需要考虑 SIMT 中 warp 内的分支分歧。

C++：
```cpp
__tile_global__ void conditional_load(float* __restrict__ arr, float* __restrict__ out, int N) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;
    using f32x8 = ct::tile<float, ct::shape<8>>;
    arr = ct::assume_aligned(arr, 16_ic);
    out = ct::assume_aligned(out, 16_ic);
    auto inView = ct::partition_view{ct::tensor_span{arr, ct::extents{N}}, ct::shape{8_ic}};
    auto outView = ct::partition_view{ct::tensor_span{out, ct::extents{N}}, ct::shape{8_ic}};
    int bx = ct::bid().x;
    int nb_x = ct::num_blocks().x;
    auto tile = ct::full<f32x8>(0.0f); // default for the last-block branch
    // Scalar condition -> one control-flow path per block; no divergence to reason about.
    if (bx < nb_x - 1) {
        tile = inView.load(bx); // all blocks except the last
    }
    outView.store_masked(tile, bx); // masked to handle a potentially partial final tile
}
```

Python：
```python
@ct.kernel
def conditional_load(arr, out, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    # Scalar condition -> one control-flow path per block; no divergence to reason about.
    if bid < ct.num_blocks(0) - 1:
        tile = ct.load(arr, index=(bid,), shape=(TILE,)) # all blocks except the last
    else:
        tile = ct.zeros((TILE,), dtype=ct.float32) # last block: emit zeros
    ct.store(out, index=(bid,), tile=tile)
```

<a id="sec-2-4-8"></a>

## 2.4.8 逐元素运算与 broadcasting

Tile 支持标准逐元素算术。两个操作数形状不同但兼容时，先 broadcast 到共同形状，再执行操作。

<a id="sec-2-4-8-1"></a>

### 2.4.8.1 Broadcasting

规则与 NumPy 相同：标量复制到全部元素；长度为 1 的维度可扩展；低秩操作数按尾部维度对齐，缺少的前导维度视为 1。如果对应维度都不为 1 且大小不同，则操作无效。

下面同时演示维度提升与 singleton 扩展：`8 × 2` 先视作 `1 × 8 × 2`，再与 `4 × 1 × 2` broadcast 到 `4 × 8 × 2`。

C++：
```cpp
auto x = ct::iota<ct::tile<int, ct::shape<8, 2>>>(); // 8x2 (rank 2)
auto y = ct::iota<ct::tile<int, ct::shape<4, 1, 2>>>(); // 4x1x2 (rank 3)
auto z = x + y; // x promoted to 1x8x2, then broadcasts to 4x8x2
```

Python：
```python
x = ct.full((8, 2), 3, dtype=ct.int32) # 8x2 (rank 2)
y = ct.full((4, 1, 2), 5, dtype=ct.int32) # 4x1x2 (rank 3)
z = x + y # x promoted to 1x8x2, then broadcasts to 4x8x2
```

<a id="sec-2-4-8-2"></a>

### 2.4.8.2 算术运算符

支持的运算符均逐元素执行，产生 broadcast 后形状的新 tile。标量与 tile 组合时，标量扩展到每个元素。不同类型通常优先选择能保留更多信息的类型。

- Tile 与 tile：选择更高精度或更大范围的类型，例如 `int + float` 得到 `float`，`int16 + int32` 得到 `int32`。
- 标量与 tile：如果标量可直接由 tile 元素类型表示，例如整数 `2` 与 int tile、`2.0f` 与 float tile，则使用 tile 的类型。
- 如果标量需要窄化才能放入 tile 类型，例如 `2.5` 与 int tile，Python 会提升到能同时容纳二者的类型，而 C++ 会将表达式判为无效。

C++：
```cpp
using i32x8 = ct::tile<int, ct::shape<8>>;
i32x8 x = ct::full<i32x8>(3);
x + 2; // OK - int literal matches int tile element type
x + 2.5; // ill-formed - 2.5 would narrow to int
```

Python：
```python
x = ct.full((8,), 3, dtype=ct.int32)
x + 2 # int32 - int literal matches int32 tile dtype
x + 2.5 # float32 - result promoted to hold both
```

实际编程时，尽量让标量字面量与 tile 元素类型一致，需要改变精度时显式转换。对加载得到的 tile 也采用相同规则。

C++：
```cpp
__tile_global__ void elementwise(float* __restrict__ a, float* __restrict__ b, float* __restrict__ out, int N) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;
    a = ct::assume_aligned(a, 16_ic);
    b = ct::assume_aligned(b, 16_ic);
    out = ct::assume_aligned(out, 16_ic);
    auto aView = ct::partition_view{ct::tensor_span{a, ct::extents{N}}, ct::shape{8_ic}};
    auto bView = ct::partition_view{ct::tensor_span{b, ct::extents{N}}, ct::shape{8_ic}};
    auto cView = ct::partition_view{ct::tensor_span{out, ct::extents{N}}, ct::shape{8_ic}};
    int bx = ct::bid().x;
    auto x = aView.load(bx);
    auto y = bView.load(bx);
    // 2.0f matches the float tiles' element type, so no narrowing conversion is required.
    // The scalar is broadcast across every element; + then runs elementwise.
    auto z = 2.0f * x + y;
    cView.store(z, bx);
}
```

Python：
```python
@ct.kernel
def elementwise(a, b, c, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    x = ct.load(a, index=(bid,), shape=(TILE,))
    y = ct.load(b, index=(bid,), shape=(TILE,))
    # 2.0 is a loosely typed float constant; with float tiles, the result stays float.
    # Scalars are broadcast across every element of the tile, then + runs elementwise.
    z = 2.0 * x + y
    ct.store(c, index=(bid,), tile=z)
```

需要显式控制舍入或 subnormal 行为时，可使用 tile API 提供的数学函数，包括逐元素 `ct.add`、`ct::add`，以及 `exp`、`tanh` 等超越函数。

<a id="sec-2-4-9"></a>

## 2.4.9 Tile 原语

Factory、load/store、逐元素运算都是 tile 原语，即语言内置操作。程序员以 tile 粒度表达，编译器映射到硬件，适用时可使用 Tensor Core。本节介绍其他原语。

<a id="sec-2-4-9-1"></a>

### 2.4.9.1 矩阵乘法

Tile 间矩阵乘法是实现数组矩阵乘法的基础。CUDA Tile 提供纯矩阵乘法 `matmul`，即 `a @ b`，以及乘加 `mma`，即 `a @ b + acc`。`mma` 的 accumulator 将前一个 K-tile 的部分结果传给下一个，适合分块矩阵乘法内循环。

两种操作支持二维矩阵乘法、三维 batched multiply，以及输入和 accumulator 采用不同精度。秩与类型限制见 API Reference。

下面采用常见模式：不论输入精度，使用 FP32 accumulator，store 时再转为输出类型。Python 使用 `ct.mma(a, b, acc)`，C++ 使用 `ct::mma(a, b, acc)`，且 `acc` 为 FP32。

K 循环次数为 `ceil(K / tk)`，覆盖 A 的右边缘、B 的下边缘。部分有效的 K-tile 加载时补零：Python 使用 `PaddingMode.ZERO`，C++ 使用 `.load_masked()`；C 的 M/N 边缘在 store 时丢弃越界写入。

C++：
```cpp
__tile_global__ void gemm(const __half* __restrict__ A, const __half* __restrict__ B, float* __restrict__ C,
                          std::size_t M, std::size_t K, std::size_t N) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;
    using f32_acc = ct::tile<float, ct::shape<32, 32>>;
    A = ct::assume_aligned(A, 16_ic);
    B = ct::assume_aligned(B, 16_ic);
    C = ct::assume_aligned(C, 16_ic);
    constexpr auto tm = 32_ic;
    constexpr auto tn = 32_ic;
    constexpr auto tk = 16_ic;
    auto aView = ct::partition_view{ct::tensor_span{A, ct::extents{M, K}}, ct::shape{tm, tk}};
    auto bView = ct::partition_view{ct::tensor_span{B, ct::extents{K, N}}, ct::shape{tk, tn}};
    auto cView = ct::partition_view{ct::tensor_span{C, ct::extents{M, N}}, ct::shape{tm, tn}};
    auto [bx, by, bz] = ct::bid();
    auto acc = ct::full<f32_acc>(0.0f); // FP32 accumulator
    std::size_t num_k = (K + tk - 1) / tk;
    for (auto k : ct::irange(std::size_t{0}, num_k)) {
        acc = ct::mma(aView.load_masked(bx, k), // zero-pad partial K-tile
                      bView.load_masked(k, by),
                      acc); // acc += a @ b
    }
    cView.store_masked(acc, bx, by); // drop OOB edge lanes
}
```

Python：
```python
@ct.kernel
def gemm(A, B, C,
         tm: ct.Constant[int], tn: ct.Constant[int], tk: ct.Constant[int]):
    bx, by = ct.bid(0), ct.bid(1)
    num_k = ct.num_tiles(A, axis=1, shape=(tm, tk)) # number of K-tiles
    acc = ct.full((tm, tn), 0, dtype=ct.float32) # FP32 accumulator
    for k in range(num_k):
        a = ct.load(A, index=(bx, k), shape=(tm, tk),
                    padding_mode=ct.PaddingMode.ZERO) # zero-pad partial K-tile
        b = ct.load(B, index=(k, by), shape=(tk, tn),
                    padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(a, b, acc) # acc += a @ b
    ct.store(C, index=(bx, by), tile=acc.astype(C.dtype)) # cast + store
```

<a id="sec-2-4-9-2"></a>

### 2.4.9.2 Reduction 与 scan

Reduction 将 tile 沿一个或多个轴归约为标量或一组标量，例如 softmax 分母、LayerNorm 的均值方差、attention score 中的最大值。

需要注意输出形状：Python 默认删除被归约轴，可用 `keepdims=True` 保留为长度 1；C++ 总是保留该轴，因此 tile 秩不变。下面都对 `2 × 4` tile 沿 axis 1 归约，但输出形状不同。

C++：
```cpp
using namespace ct::literals;
using i32x2x4 = ct::tile<int, ct::shape<2, 4>>;
auto x = ct::iota<i32x2x4>(); // [[0,1,2,3],[4,5,6,7]]
auto row_sums = ct::sum(x, 1_ic); // shape (2, 1) - axis kept
// row_sums == [[6], [22]]
```

Python：
```python
x = ct.arange(8, dtype=ct.int32).reshape((2, 4)) # [[0,1,2,3],[4,5,6,7]]
s = ct.sum(x, axis=1) # shape (2,) - axis dropped
s_k = ct.sum(x, axis=1, keepdims=True) # shape (2, 1) - axis kept
# s == [6, 22]; s_k == [[6], [22]]
```

Scan 是沿轴累积的对应操作。例如 prefix sum（`cumsum`）输出与输入维度相同，每个位置保存到该位置为止的前缀和。完整操作见各语言 reduction 与 scan API 文档。

<a id="sec-2-4-9-3"></a>

### 2.4.9.3 Transpose 与 permutation

这两类原语改变 tile 的轴排列：`transpose` 交换轴，`permute` 指定任意排列。它们用于矩阵乘操作数转置、attention block 的行列交换，或在 broadcast 前对齐各轴。

Python 对二维 tile 调用 `ct.transpose(x)` 交换两轴，更高维时显式指定 `axis0`、`axis1`；`ct.permute(x, axes)` 接受轴索引 tuple。

C++ 的 `ct::transpose(x)` 交换前两维，保留后续维度；`ct::permute(x, map)` 用 `ct::dimension_map` 描述新顺序。

C++：
```cpp
using namespace ct::literals;
using t2d = ct::tile<int, ct::shape<2, 4>>;
using t3d = ct::tile<int, ct::shape<2, 2, 2>>;
auto tx = ct::iota<t2d>();
auto ty = ct::transpose(tx); // shape (4, 2)
auto tz = ct::iota<t3d>();
auto tw = ct::permute(tz, ct::dimension_map{2_ic, 0_ic, 1_ic}); // axes (0,1,2) -> (2,0,1)
```

Python：
```python
tx = ct.arange(8, dtype=ct.int32).reshape((2, 4))
ty = ct.transpose(tx) # shape (4, 2)
tz = ct.arange(8, dtype=ct.int32).reshape((2, 2, 2))
tw = ct.permute(tz, (2, 0, 1)) # axes (0,1,2) -> (2,0,1)
```

<a id="sec-2-4-9-4"></a>

### 2.4.9.4 逐元素选择

给定布尔 tile 和两个数据 tile，根据对应布尔值逐元素选择结果。条件可 broadcast 到操作数形状，两个数据操作数的类型须兼容。

Python 使用 `ct.where(cond, x, y)`；C++ 使用 `ct::select(cond, lhs, rhs)`。

C++：
```cpp
using namespace ct::literals;
auto cond = ct::iota<ct::tile<int, ct::shape<4>>>() < 2; // {T, T, F, F}
auto t = ct::full<ct::tile<float, ct::shape<4>>>( 1.0f);
auto f = ct::full<ct::tile<float, ct::shape<4>>>(-1.0f);
auto r = ct::select(cond, t, f); // {1, 1, -1, -1}
```

Python：
```python
cond = ct.arange(4, dtype=ct.int32) < 2 # [T, T, F, F]
x_true = ct.full((4,), 1.0, dtype=ct.float32)
x_false = ct.full((4,), -1.0, dtype=ct.float32)
result = ct.where(cond, x_true, x_false) # [1, 1, -1, -1]
```

<a id="sec-2-4-9-5"></a>

### 2.4.9.5 数学函数

`ct` 命名空间提供以下常见逐元素操作：

| 类别 | 函数 |
| --- | --- |
| 基本算术 | `add`、`sub`、`mul` |
| 除法与取模 | `truediv`、`floordiv`、`cdiv`、`mod` |
| 幂、指数、对数 | `pow`、`exp`、`exp2`、`log`、`log2` |
| 平方根 | `sqrt`、`rsqrt` |
| 三角函数 | `sin`、`cos`、`tan` |
| 双曲函数 | `sinh`、`cosh`、`tanh` |
| 比较与取负 | `minimum`、`maximum`、`negative` |
| 取整 | `floor`、`ceil` |

函数逐元素处理输入 tile 并返回相应形状的 tile，也可在 tile code 中处理标量。精确规则和完整列表见 [cuTile Python Math Operations](https://docs.nvidia.com/cuda/cutile-python/operations.html#math)、[CUDA Tile C++ Math Operations](https://docs.nvidia.com/cuda/cuda-tile-cpp-api-reference/math_operations.html)。

<a id="sec-2-4-10"></a>

## 2.4.10 Atomic memory 操作

Tile code 中，常见的 atomic 需求有两种：不同 block 将部分结果合并到同一 global memory 地址；或同一 tile 的多个元素写入同一地址。

对 tile 执行一次 atomic 调用，会为每个元素执行一次 atomic update。<strong>单个元素更新是原子的，整个 tile 调用并不是一个整体原子操作。</strong> 各元素操作顺序未指定。

Python 通过 array 索引寻址，规则与 gather/scatter 相同。可选参数控制边界检查、memory order、thread scope，默认启用边界检查，采用 `ACQ_REL` 和 device scope。

`TiledView` 还提供 `atomic_store_` 开头的方法，例如 `atomic_store_add(index, update)`，使用 tile-space 索引，不返回旧值，并可生成 PTX atomic reduction。不需要旧值时，优先使用它以提升性能。

C++ 自由函数接收标量指针与标量值，或指针 tile 与数值 tile。Memory order 通过编译时类型标签指定，例如 `ct::memory_order_relaxed_t{}`；thread scope 也使用类型标签，省略时默认为 system scope。

`ct::partition_view`、`ct::strided_view` 还提供 `atomic_add`、`atomic_sub`、`atomic_and`、`atomic_or`、`atomic_xor`、`atomic_max`、`atomic_min` 成员函数。与 `.store()` 一样，value tile 在前，tile-space 索引在后。它们不返回旧值，可生成硬件 reduction；不需要旧值时应优先使用。`ct::gather_scatter_view` 不提供这些成员。

<a id="sec-2-4-10-1"></a>

### 2.4.10.1 跨 block 竞争

下面不同 block 写入同一 `out` 地址，若不使用 atomic，并行更新将产生错误。C++ 显式使用 `ct::thread_scope_device_t{}`，Python 默认 device scope，使更新对所有 block 可见。

Python 使用 `TiledView.atomic_store_add`，因为只需要将每个 block 的部分和累加到 `out[0]`，不需要返回旧值。

C++：
```cpp
__tile_global__ void block_sum(int* __restrict__ arr, int* __restrict__ out, std::size_t N) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;
    constexpr auto TILE = 16_ic;
    arr = ct::assume_aligned(arr, 16_ic);
    out = ct::assume_aligned(out, 16_ic);
    auto aView = ct::partition_view{ct::tensor_span{arr, ct::extents{N}},
                                    ct::shape{TILE}};
    int bid = ct::bid().x;
    auto tile = aView.load_masked(bid); // partial final tile -> OOB lanes default to 0
    auto partial = ct::sum(tile, 0_ic); // reduce to a 1-element tile
    ct::atomic_add(out, (int)partial, // accumulate the scalar into out[0]
                   ct::memory_order_relaxed_t{}, // single-location accumulator -> relaxed suffices
                   ct::thread_scope_device_t{}); // visible across the device
}
```

Python：
```python
@ct.kernel
def block_sum(arr, out, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    # partial final tile -> OOB lanes default to 0
    tile = ct.load(arr, index=(bid,), shape=(TILE,),
                      padding_mode=ct.PaddingMode.ZERO)
    partial = ct.sum(tile) # reduce to a scalar
    out.tiled_view((1,)).atomic_store_add((0,), partial) # atomically accumulate into out[0]
```

<a id="sec-2-4-10-2"></a>

### 2.4.10.2 block 内竞争

下面一个 tile 的所有元素原子累加到同一个 `slot` 地址。`ptrs` 中每个指针都指向同一位置，`ct::iota<i32x16>()` 的每个元素分别执行 atomic add，顺序未指定。

使用 `ct::thread_scope_block_t{}`，表示只需要 block 内的原子可见性范围。
```cpp
using i32x16 = ct::tile<int, ct::shape<16>>;
int* slot = /* pointer to the contended location */;
// 16 lanes all aim at the same address. Add is commutative, so the
// unspecified ordering doesn't affect this sum; block scope suffices
// since contention stays within one block.
auto ptrs = ct::full<ct::tile<int*, ct::shape<16>>>(slot);
ct::atomic_add(ptrs, ct::iota<i32x16>(),
            ct::memory_order_relaxed_t{},
            ct::thread_scope_block_t{});
```

::: tip 注意
这只是说明性示例。若只是将 block 内一个 tile 求和为标量，应优先使用 [reduction](/chapter-2/section-2-4#sec-2-4-9-2)。
:::

<a id="sec-2-4-10-3"></a>

### 2.4.10.3 支持的 atomic 操作

| 操作 | 作用 |
| --- | --- |
| `atomic_and` / `atomic_or` / `atomic_xor` | 与 memory 旧值逐元素执行按位与、或、异或。 |
| `atomic_max` / `atomic_min` | 比较传入值与旧值，写入较大值或较小值。 |
| `atomic_add` | 累加并写回。 |
| `atomic_sub` | 减去传入值并写回，仅 C++ 提供此名称。 |
| `atomic_xchg` | 写入新值并返回旧值。 |
| `atomic_cas` | 比较旧值与 expected，相等时替换为 desired；C++ 名称为 `atomic_compare_exchange`。 |

完整说明见两种语言 API Reference 的 memory operation 部分。

<a id="sec-2-4-11"></a>

## 2.4.11 优化 hint

Hint 是附着在 kernel 或特定 load/store 调用处的元数据，用于引导编译器生成代码。它不改变程序语义，可以添加、移除或调节而不影响正确性，编译器也可以忽略它。

Hint 具有两个共同特征：只作用于所附着的特定结构，不作用于周围所有代码；可以按 GPU 架构分别配置，也可以为全部架构指定统一值。

C++ 使用 attribute，Python 使用 kernel decorator 和 memory 调用上的关键字参数。两种语言的 hint 种类及含义相同，见 [Hint 类型](/chapter-2/section-2-4#sec-2-4-11-3)。

<a id="sec-2-4-11-1"></a>

### 2.4.11.1 C++ 的 cutile::hint attribute

语法如下：
```cpp
[[ cutile::hint(arch, kind1=value1, kind2=value2, ...) ]]
```

第一个参数是目标架构整数，采用 `__CUDA_ARCH__` 的编码规则，例如 `900` 表示 `sm_90`、`1000` 表示 `sm_100`；`0` 表示所有架构。其余参数为 `kind=value`。

Kernel hint 放在函数声明上；`ct::load`、`ct::store`、partition view 等 memory 操作的 hint，放在包含调用的表达式语句前。其他位置的规则见 API 文档。

下面为 `sm_90`、`sm_100` 指定不同 `num_cta_in_cga`，并将某次 load 标记为较重的 memory 流量。
```cpp
[[ cutile::hint(900, num_cta_in_cga=4), // sm_90: prefer 4 CTAs per cluster
   cutile::hint(1000, num_cta_in_cga=8) ]] // sm_100: prefer 8 CTAs per cluster
__tile_global__ void optimization_hints(float* __restrict__ in,
                                        float* __restrict__ out) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;
    in = ct::assume_aligned(in, 16_ic);
    out = ct::assume_aligned(out, 16_ic);
    auto inSpan = ct::tensor_span{in, ct::extents{128_ic}};
    auto outSpan = ct::tensor_span{out, ct::extents{128_ic}};
    auto inView = ct::partition_view{inSpan, ct::shape{8_ic}};
    auto outView = ct::partition_view{outSpan, ct::shape{8_ic}};
    int bx = ct::bid().x;
    // Expression-statement hint: tag this particular load as bandwidth-heavy.
    ct::tile<float, ct::shape<8>> tile;
    [[ cutile::hint(0, latency=8) ]]
    tile = inView.load(bx);
    outView.store(tile, bx);
}
```

同一结构上同类 hint 同时存在时，特定架构配置优先于通用配置。

<a id="sec-2-4-11-2"></a>

### 2.4.11.2 Python decorator 参数与调用关键字

Python 的 kernel 级 hint 是 `@ct.kernel(...)` 的关键字参数。已编译 kernel 还提供 `.replace_hints(**hints)`，返回采用新 hint 的 kernel，并具有自己的 JIT cache，适合作为 autotuning 循环的基础。

单次调用 hint 则放在 `ct.load/store`、`TiledView.load/store`、`ct.gather/scatter` 等调用处。

按架构指定时使用 `cuda.tile.ByTarget(default=..., sm_XXX=..., sm_YYY=...)`，架构名如 `sm_100`、`sm_120`。普通值则适用于所有目标，相当于 C++ 的 `arch=0`。

下面对应前面的 C++ 示例：`ByTarget` 承载 kernel hint，`latency=8` 设置特定调用，`replace_hints` 无需修改源代码即可生成另一配置。
```python
@ct.kernel(num_ctas=ByTarget(sm_90=4, sm_100=8))
def optimization_hints(in_, out, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    # Per-call hint: this particular load is bandwidth-heavy.
    tile = ct.load(in_, index=(bid,), shape=(TILE,), latency=8)
    ct.store(out, index=(bid,), tile=tile)
# Autotuning: produce a new kernel with overridden hints without editing the
# source. The new kernel has its own JIT cache.
tuned_kernel = optimization_hints.replace_hints(num_ctas=8)
```

<a id="sec-2-4-11-3"></a>

### 2.4.11.3 Hint 类型

以下 hint 在两种语言中含义和取值一致，只是部分名称不同。

<a id="sec-2-4-11-3-1"></a>

#### 2.4.11.3.1 每个 cluster 的 CTA 数

- C++：kernel attribute `num_cta_in_cga`。
- Python：`@ct.kernel` 的 `num_ctas`。
- 取值：`1`、`2`、`4`、`8`、`16`；`sm_80` 仅可使用 `1`。
- 含义：建议 kernel 的每个 Cooperative Group Array（CGA）包含多少 Cooperative Thread Array（CTA）。

<a id="sec-2-4-11-3-2"></a>

#### 2.4.11.3.2 Occupancy

- 两种语言都使用 `occupancy`，位于 kernel 级。
- 取值：闭区间 `[1, 32]` 的整数。
- 含义：建议每个 SM 的活跃 CTA 数量。编译器在代码生成时尽量满足，属于建议而非保证。

<a id="sec-2-4-11-3-3"></a>

#### 2.4.11.3.3 Memory access latency

- 两种语言均为 `latency`。C++ 放在调用所在表达式语句上，Python 放在调用关键字参数中。
- 适用于 tile-space load/store 与 gather/scatter。
- 取值为 `[1, 10]` 的整数，1 表示轻 DRAM 流量，10 表示重流量。较大值通常使编译器采用更深的 prefetch。

<a id="sec-2-4-11-3-4"></a>

#### 2.4.11.3.4 允许 TMA

- 两种语言均为 `allow_tma`。
- 仅适用于 tile-space load/store，不适用于 gather/scatter。
- 取值为布尔值。默认允许 TMA；设为 `false` 或 `False`，可要求编译器不要把该次访问映射到 TMA。

<a id="sec-2-4-12"></a>

## 2.4.12 C++ 性能建议

本节说明前面 C++ 示例常用的注解与编程模式，以及它们对性能的影响。

<a id="sec-2-4-12-1"></a>

### 2.4.12.1 对 memory 中的数组使用 __restrict__ 指针

`__restrict__` 告诉编译器，在该指针有效期间，通过它访问的区域不会通过其他独立指针别名访问，详见 [__restrict__ 指针](/chapter-5/section-5-4#sec-5-4-1-4)。满足条件并正确标记，对 tile C++ 的 memory 操作性能很重要。

考虑没有 `__restrict__` 的逐元素复制：
```cpp
__tile_global__ void tile_elementwise_copy(float* out, float const* in) {
    namespace ct = cuda::tiles;
    using f32x64 = ct::tile<float, ct::shape<64>>;
    using i32x64 = ct::tile<int, ct::shape<64>>;
    auto inPtrs = in + 64 * ct::bid().x + ct::iota<i32x64>();
    auto outPtrs = out + 64 * ct::bid().x + ct::iota<i32x64>();
    auto data = ct::load(inPtrs); // (1)
    ct::store(outPtrs, data); // (2)
}
```

通常无需关心编译器如何并行化 tile，但这里可以看出不重叠数组的优势：输入、输出互不重叠时，load 可拆为独立读取，store 只依赖自己使用的数据已经读取完成，因此读写可交错。

若输入、输出可能重叠，编译器必须先完成整个 tile 的所有 load，再发出任何 store，以免提前覆盖尚未读取的数据。这会限制调度优化。

因此，使用不重叠数组，并通过 `__restrict__` 告知编译器，有助于生成更高效代码；错误地对具有别名访问的区域作此承诺，会导致未定义行为。

<a id="sec-2-4-12-2"></a>

### 2.4.12.2 将数组指针标记为 16 字节对齐

使用 `ct::assume_aligned`：
```cpp
__tile_global__ void foo(float* __restrict__ in) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;
    in = ct::assume_aligned(in, 16_ic);
    ct::tensor_span t{in, ct::extents{256_ic, 256_ic}};
    ct::partition_view{t, ct::shape{4_ic, 4_ic}};
    // ...
}
```

这是 `ct::partition_view` 使用 TMA 的必要对齐保证。运行时必须确实提供 16 字节对齐的指针，否则是未定义行为。`cudaMalloc` 等 CUDA allocator 返回的指针保证至少满足这一对齐。

<a id="sec-2-4-12-3"></a>

### 2.4.12.3 优先使用 ct::partition_view 访问 memory

对于结构化访问，优先使用 `ct::partition_view`，而非 gather/scatter 形式的 `ct::load`、`ct::store`。View 形式可在支持的硬件上映射到 TMA，通常比逐元素 gather 更快。相关背景见 [Gather 与 scatter](/chapter-2/section-2-4#sec-2-4-6-2)。

<a id="sec-2-4-12-4"></a>

### 2.4.12.4 有界循环使用 ct::irange

遍历确定范围时，优先使用 `ct::irange` 而不是普通整数 `for`。结构化范围让编译器更容易进行 pipelining、vectorization 等优化；边界与步长只是难以分析的整数表达式时，无法获得同样的信息，见 [控制流](/chapter-2/section-2-4#sec-2-4-7)。
```cpp
for (auto idx : ct::irange(lowerBound, upperBound, step)) {
    // ...
}
```
