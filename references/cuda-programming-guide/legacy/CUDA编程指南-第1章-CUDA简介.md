# 第 1 章 CUDA 简介

> 来源：本目录中的《CUDA Programming Guide》，Release 13.4.2，NVIDIA Corporation，2026 年 9 月 10 日。
> 范围：第 1 章 Introduction to CUDA，书内第 3–19 页（PDF 第 21–37 页）。
> 本文按原文章节顺序完整翻译正文、图注、提示及脚注，移除重复页眉和页脚。图片直接提取自 PDF，保留图内英文，中文说明见图注。文内章节与图号引用均可点击跳转；第 2 章及之后的章节目前仅有标题框架，尚未翻译正文。

> 术语约定：常用 CUDA 术语及内存术语保留英文，首次介绍时附中文释义；block 指执行单元，tile 指数据单元。

<a id="sec-1-1"></a>

## 1.1 引言

<a id="sec-1-1-1"></a>

### 1.1.1 图形处理器

图形处理器（Graphics Processing Unit，GPU）最初是面向三维图形的专用处理器，采用固定功能硬件，加速实时三维渲染中的并行操作。随着一代代产品的发展，GPU 的可编程能力不断增强。到 2003 年，图形流水线中的部分阶段已经能够完全编程，可以针对三维场景或图像中的各个组成部分并行运行自定义代码。

2006 年，NVIDIA 推出了计算统一设备架构（Compute Unified Device Architecture，CUDA），使任何计算工作负载都能独立于图形 API，利用 GPU 的高吞吐能力。

此后，CUDA 和 GPU 计算被用于加速几乎所有类型的计算任务，从流体动力学、能量传输等科学仿真，到数据库、数据分析等商业应用。此外，GPU 的计算能力和可编程性也为新算法与新技术的发展奠定了基础，涵盖图像分类，以及扩散模型、大语言模型等生成式人工智能。

<a id="sec-1-1-2"></a>

### 1.1.2 使用 GPU 的优势

在相近的价格和功耗范围内，GPU 能提供远高于 CPU 的指令吞吐量和 memory bandwidth。许多应用利用这些能力，在 GPU 上获得了明显快于 CPU 的运行速度（参见原书中的 [GPU Applications](https://www.nvidia.com/en-us/accelerated-applications/) 链接）。FPGA 等其他计算设备同样具有很高的能效，但其编程灵活性远低于 GPU。

GPU 与 CPU 的设计目标不同。CPU 致力于尽可能快地执行一串串行操作，即一个 thread（线程），并能并行执行几十个这样的 thread。GPU 则致力于同时执行数千个 thread，通过牺牲部分单个 thread 的性能，获得高得多的总体吞吐量。

GPU 专门面向高度并行的计算，因此将更多晶体管用于数据处理单元；CPU 则将更多晶体管用于数据 cache 和流程控制。[图 1](#figure-01) 展示了 CPU 与 GPU 芯片资源分配的一个示例。

<a id="figure-01"></a>

![图 1：GPU 将更多晶体管用于数据处理](images/chapter-01/figure-01.png)

*图 1：GPU 将更多晶体管用于数据处理。*

<a id="sec-1-1-3"></a>

### 1.1.3 快速入门

利用 GPU 的计算能力有很多途径。本指南介绍如何使用 C++ 等高级语言为 CUDA GPU 平台编程。不过，应用程序也有许多无需直接编写 GPU 代码就能使用 GPU 的方法。

各个领域中越来越多的算法和例程，都可以通过专用库获得。如果某个库已经实现了所需功能，尤其是 NVIDIA 提供的库，那么使用它通常比从头重新实现算法具有更高的开发效率和更好的性能。cuBLAS、cuFFT、cuDNN 和 CUTLASS 只是其中几个例子，它们帮助开发者避免重复实现成熟算法。这些库还针对不同 GPU 架构进行了优化，能够兼顾开发效率、性能和可移植性。

此外，一些框架，特别是人工智能领域使用的框架，也提供了 GPU 加速的基础组件。许多框架正是通过调用上述 GPU 加速库来实现加速。

NVIDIA 的 Warp、OpenAI 的 Triton 等领域专用语言（Domain-Specific Language，DSL），也可以编译为直接在 CUDA 平台上运行的代码。与本指南介绍的高级语言相比，它们提供了抽象层次更高的 GPU 编程方式。

[NVIDIA Accelerated Computing Hub](https://github.com/NVIDIA/accelerated-computing-hub) 提供了用于学习 GPU 和 CUDA 计算的资源、示例及教程。

<a id="sec-1-2"></a>

## 1.2 编程模型

本节从较高层次介绍 CUDA 编程模型，不依赖任何具体语言。这里引入的术语和概念，适用于所有受支持编程语言中的 CUDA。后续章节将使用 C++ 演示这些概念。

<a id="sec-1-2-1"></a>

### 1.2.1 异构系统

CUDA 编程模型假定运行环境是异构计算系统，即同时包含 GPU 和 CPU 的系统。CPU 及其直接连接的内存，分别称为**host（主机）**和**host memory（主机内存）**。GPU 及其直接连接的内存，分别称为**device（设备）**和**device memory（设备内存）**。在某些片上系统（SoC）中，它们可能位于同一个封装内；在更大的系统中，则可能包含多个 CPU 或 GPU。

CUDA 应用程序会将一部分代码放在 GPU 上执行，但应用程序总是从 CPU 开始运行。在 CPU 上运行的代码称为**host code（主机代码）**。host code 可以使用 CUDA API 在 host memory 与 device memory 之间复制数据、启动 GPU 上的代码，并等待数据复制或 GPU 代码执行完成。CPU 和 GPU 可以同时执行代码；通常，让二者都得到尽可能充分的利用，才能获得最佳性能。

应用程序在 GPU 上执行的代码称为**device code（设备代码）**。出于历史原因，被调用以在 GPU 上执行的函数称为**kernel（核函数）**。开始运行 kernel 的动作称为**kernel launch（启动核函数）**。可以将 kernel launch 理解为：让许多 thread 在 GPU 上并行执行同一份 kernel 代码。GPU thread 与 CPU thread 的行为类似，但二者也有一些影响正确性和性能的重要差异，后续小节将详细介绍，参见[第 3.2.2.1.1 节 Independent Thread Scheduling](CUDA%E7%BC%96%E7%A8%8B%E6%8C%87%E5%8D%97-%E7%AC%AC3%E7%AB%A0-CUDA%E8%BF%9B%E9%98%B6.md#sec-3-2-2-1-1)。

<a id="sec-1-2-2"></a>

### 1.2.2 GPU 硬件模型

与其他编程模型一样，CUDA 依赖于对底层硬件的概念性描述。对于 CUDA 编程，可以将 GPU 看作一组**SM（Streaming Multiprocessor，流式多处理器）**，这些 SM 又组成若干**GPC（Graphics Processing Cluster，图形处理簇）**。每个 SM 都包含本地 register file（寄存器文件）、unified data cache（统一数据缓存），以及若干执行计算的功能单元。unified data cache 为 shared memory（共享内存）和 L1 cache（缓存）提供物理资源，可以在运行时配置分配给两者的容量比例。不同 GPU 架构中，各类内存的大小及 SM 内功能单元的数量可能不同。

> **注意**：GPU 的实际硬件布局，或其在物理层面执行编程模型的方式，可能有所不同。这些差异不会影响遵循 CUDA 编程模型编写的软件的正确性。

<a id="figure-02"></a>

![图 2：CPU、GPU、SM 与 GPC 的硬件组织](images/chapter-01/figure-02.png)

*图 2：GPU 包含许多 SM，每个 SM 又包含许多功能单元。GPC 是 SM 的集合，GPU 则由连接到 GPU memory 的一组 GPC 构成。CPU 通常包含多个核心，以及连接 system memory 的 memory controller。CPU 与 GPU 通过 PCIe 或 NVLINK 等互连连接。*

<a id="sec-1-2-2-1"></a>

#### 1.2.2.1 block 与 grid

应用程序启动 kernel 时，会启动许多 thread，通常可达数百万个。这些 thread 被组织成块，称为**thread block（线程块，后文简称 block）**。block 进一步组成**grid（网格）**。同一个 grid 内，所有 block 的大小和维度都相同。[图 3](#figure-03) 展示了由 block 组成的 grid。

block 和 grid 都可以是一维、二维或三维。这些维度有助于简化单个 thread 与工作单元或数据元素之间的映射。

启动 kernel 时，需要指定一个**execution configuration（执行配置）**，其中包含 grid 和 block 的维度。execution configuration 还可以包含可选参数，例如 cluster 大小、stream（流）以及 SM 配置设置，后续小节将介绍这些内容。

执行 kernel 的每个 thread，都可以通过内置变量确定自己在 block 中的位置，以及所属 block 在 grid 中的位置。thread 还可以通过这些内置变量，获得启动 kernel 时所指定的 block 和 grid 的维度。这样，每个 thread 在执行该 kernel 的所有 thread 中都有了唯一身份。这个身份经常用于确定该 thread 负责处理的数据或操作。

<a id="figure-03"></a>

![图 3：block 组成的 grid](images/chapter-01/figure-03.png)

*图 3：block 组成的 grid。每个箭头代表一个 thread，图中的箭头数量不代表实际 thread 数量。*

一个 block 中的所有 thread 都在同一个 SM 上执行。这使得 block 内 thread 能够高效通信与同步。block 中的所有 thread 都可以访问片上 shared memory，并用它在 block 内交换信息。

一个 grid 可能包含数百万个 block，而执行它的 GPU 可能只有几十或几百个 SM。一个 block 的所有 thread 都由同一个 SM 执行，并且在大多数情况下，会一直在该 SM 上运行直至完成。[^1] block 之间没有调度顺序保证，因此一个 block 不能依赖其他 block 的结果，因为其他 block 可能只有等当前 block 完成后才有机会被调度。[图 4](#figure-04) 展示了 grid 中的 block 如何分配给 SM。

CUDA 编程模型允许任意规模的 grid 运行在任意规模的 GPU 上，无论 GPU 只有一个 SM，还是拥有数千个 SM。为此，除某些例外情况之外，CUDA 编程模型要求不同 block 的 thread 之间不能有数据依赖。也就是说，同一 grid 中，一个 thread 不应依赖另一个 block 中的 thread 产生的结果，也不应与其同步。block 内的所有 thread 在同一个 SM 上同时运行；grid 内不同 block 会被调度到可用的 SM 上，并且可以按任意顺序执行。换言之，CUDA 编程模型要求 block 能够以任意顺序、并行或串行地执行。

<a id="figure-04"></a>

![图 4：block 调度到 SM](images/chapter-01/figure-04.png)

*图 4：每个 SM 都有一个或多个活跃 block。本例中，每个 SM 同时调度了三个 block。grid 中的 block 分配给 SM 的顺序没有保证。*

<a id="sec-1-2-2-1-1"></a>

##### 1.2.2.1.1 thread block cluster

除了 block 之外，Compute Capability 为 9.0 及以上的 GPU 还提供一个可选的分组层级，称为**thread block cluster（线程块簇，后文简称 cluster）**。cluster 由一组 block 组成，与 block 和 grid 类似，也可以按照一维、二维或三维布局。[图 5](#figure-05) 展示了按 cluster 进一步分组的 block grid。指定 cluster 不会改变 grid 维度，也不会改变 block 在 grid 中的索引。

指定 cluster 会将相邻 block 划分为一组，并提供额外的 cluster 级同步与通信能力。具体来说，同一个 cluster 中的所有 block 都在同一个 GPC 内执行。[图 6](#figure-06) 展示了指定 cluster 时，block 如何调度到 GPC 内的 SM 上。由于这些 block 会在同一个 GPC 内同时被调度，同一 cluster 内不同 block 中的 thread 可以使用 Cooperative Groups 提供的软件接口进行通信与同步。cluster 内 thread 可以访问该 cluster 中所有 block 的 shared memory，这称为**distributed shared memory（分布式共享内存）**。cluster 的最大大小依赖于硬件，不同设备可能不同。

[图 6](#figure-06) 还展示了 cluster 内 block 如何同时调度到同一个 GPC 的 SM 上。一个 cluster 内的 block 在 grid 中总是彼此相邻。

<a id="figure-05"></a>

![图 5：包含 thread block cluster 的 grid](images/chapter-01/figure-05.png)

*图 5：指定 cluster 后，block 在 grid 中的位置保持不变，同时也具有在所属 cluster 中的位置。*

<a id="figure-06"></a>

![图 6：cluster 内 block 在同一个 GPC 中调度](images/chapter-01/figure-06.png)

*图 6：指定 cluster 后，cluster 内 block 在 grid 中按照 cluster 的形状排列。同一 cluster 内的 block 会被同时调度到单个 GPC 内的 SM 上。*

<a id="sec-1-2-2-2"></a>

#### 1.2.2.2 warp 与 SIMT

在 block 内部，thread 以每组 32 个 thread 组织为**warp（线程束）**。warp 使用**SIMT（Single-Instruction Multiple-Threads，单指令多线程）**范式执行 kernel 代码。在 SIMT 中，同一个 warp 中的所有 thread 执行相同的 kernel 代码，但每个 thread 可以沿代码中的不同分支运行。也就是说，虽然所有 thread 执行相同的代码，却不必遵循相同的执行路径。

thread 由 warp 执行时，会被分配到一个**lane（线程束通道）**。lane 编号为 0–31。block 中的 thread 按照可预测的方式分配给 warp，详见[第 3.2.2.2 节 Hardware Multithreading（硬件多线程）](CUDA%E7%BC%96%E7%A8%8B%E6%8C%87%E5%8D%97-%E7%AC%AC3%E7%AB%A0-CUDA%E8%BF%9B%E9%98%B6.md#sec-3-2-2-2)。

同一 warp 中的所有 thread 同时执行同一条指令。如果其中一些 thread 进入某个控制流分支，而其他 thread 不进入，那么执行该分支时，不进入该分支的 thread 会被屏蔽。例如，如果某个条件只对一个 warp 中一半的 thread 成立，那么活跃 thread 执行这些指令时，另一半 thread 将被屏蔽。[图 7](#figure-07) 展示了这种情况。同一 warp 中的 thread 走向不同代码路径时，这种情况有时称为**warp divergence（线程束分歧）**。因此，当 warp 内的 thread 遵循相同的控制流路径时，GPU 利用率最高。

<a id="figure-07"></a>

![图 7：warp 中的条件分支与 thread 屏蔽](images/chapter-01/figure-07.png)

*图 7：本例中，只有 thread 索引为偶数的 thread 执行 `if` 语句体；执行语句体时，其他 thread 会被屏蔽。*

在 SIMT 模型中，同一个 warp 中的所有 thread 以同步步进的方式执行 kernel。实际硬件执行方式可能不同，关于这种差别在哪些情况下具有重要影响，参见[第 3.2.2.1.1 节 Independent Thread Scheduling（独立线程调度）](CUDA%E7%BC%96%E7%A8%8B%E6%8C%87%E5%8D%97-%E7%AC%AC3%E7%AB%A0-CUDA%E8%BF%9B%E9%98%B6.md#sec-3-2-2-1-1)。不建议依赖 warp 执行如何映射到真实硬件的具体细节。CUDA 编程模型和 SIMT 规定，同一 warp 的所有 thread 一起推进代码执行。只要程序遵循编程模型，硬件就可以采用对程序透明的方式优化被屏蔽的 lane。如果程序违反该模型，则可能导致未定义行为，而且在不同 GPU 硬件上表现不同。

编写 CUDA 代码时并不一定需要考虑 warp，但理解 warp 执行模型，有助于理解 global memory 合并访问、shared memory bank（存储体）访问模式等概念。一些高级编程技术会在 block 内对不同 warp 进行专门分工，以减少 thread 分歧并提高利用率。这些优化利用了 thread 在执行时按 warp 分组这一事实。

warp 执行方式带来的一个结果是：最好将 block 的 thread 总数设置为 32 的倍数。使用其他 thread 数也是合法的，但如果总数不是 32 的倍数，最后一个 warp 中的部分 lane 会在整个执行过程中始终闲置。这通常会使该 warp 的功能单元利用率和内存访问表现不够理想。

SIMT 经常与**单指令多数据（Single Instruction Multiple Data，SIMD）**并行进行比较，但两者存在重要差别。在 SIMD 中，执行遵循单一控制流路径；在 SIMT 中，每个 thread 都可以遵循自己的控制流路径。因此，SIMT 不像 SIMD 那样具有固定的数据宽度。更详细的讨论参见[第 3.2.2.1 节 SIMT Execution Model](CUDA%E7%BC%96%E7%A8%8B%E6%8C%87%E5%8D%97-%E7%AC%AC3%E7%AB%A0-CUDA%E8%BF%9B%E9%98%B6.md#sec-3-2-2-1)。

<a id="sec-1-2-2-3"></a>

#### 1.2.2.3 CUDA 中的 tile programming

除了前面介绍的 SIMT 模型，CUDA 还支持**tile programming model（分块编程模型）**。在 tile programming 中，程序员以整个 block 为单位编写代码，描述对称为**tile（数据块）**的多维数据集合进行的操作。编译器负责将这些操作映射到 block 内的各个 thread。

与[第 1.2.2.1 节 block 与 grid](#sec-1-2-2-1)介绍的一样，tile kernel 也在由 block 组成的 grid 上启动。每个 block 执行 tile kernel，并可以查询自身在 grid 中的位置，以确定自己负责处理哪部分数据。程序员只需指定 grid 维度；每个 block 的 thread 数量由编译器根据 kernel 中的 tile 操作确定，见[图 8](#figure-08)。

<a id="figure-08"></a>

![图 8：SIMT 与 tile programming 的程序员视角](images/chapter-01/figure-08.png)

*图 8：SIMT 和 tile programming 模型中的程序员视角。在 SIMT 中，程序员编写以 thread 为单位的代码，并控制各 thread 如何访问数据。在 tile programming 中，程序员编写以 block 为单位的代码，对 tile 进行操作；编译器负责将这些操作映射到 block 中的 thread。*

在 tile kernel 内部，整个 block 遵循单一控制流。程序员指定对 tile 的操作，编译器将工作分配给 block 内 thread。条件语句、循环等标准控制流结构均受支持，但由于 block 遵循单一控制流，因此没有 warp divergence 的概念。计算索引或循环边界等标量操作，由 block 内一个 thread 执行；两个 tile 逐元素相加等 tile 操作，则由 block 内所有 thread 共同并行执行。

需要区分：**block（线程块）是执行单元，tile（数据块）是数据单元**。同一个 block 可以创建和操作多个形状、数据类型各不相同的 tile。

<a id="sec-1-2-2-3-1"></a>

##### 1.2.2.3.1 数组与 tile

tile kernel 处理两类数据：**array（数组）**和**tile（数据块）**。数组，也称全局数组，是存放在 device memory 中的多维元素容器。数组是可变的：kernel 中的 store 操作可以修改其内容。数组具有形状和数据类型。

tile 是一个多维值集合，只存在于 tile 代码中，且局限于单个 block。tile 是不可变的：每次操作都会产生一个新 tile，而不是修改已有 tile。与数组不同，tile 不一定具有对应的内存表示。编译器决定如何存储 tile 中的数据，可能使用 register、shared memory 或 SM 的其他资源。tile 的每个维度都必须是 2 的幂，而且必须在编译时已知，即必须在 kernel 执行之前就能确定，而不是在执行期间计算得到。tile 不能作为 kernel 参数传入；它们完全在 tile 代码内部创建和使用。

<a id="sec-1-2-2-3-2"></a>

##### 1.2.2.3.2 tile space 与数据移动

数据通过 load（加载）和 store（存储）操作，在数组和 tile 之间移动。这些操作使用**tile space（数据块空间）**的概念：在概念上，将数组划分为大小相等、互不重叠的 tile 后，便得到 tile space。

例如，对于形状为 `(M, N)` 的二维数组，如果 load 操作指定 tile 形状为 `(t_m, t_n)`，那么该数组在概念上就被划分为 `⌈M / t_m⌉` 行、`⌈N / t_n⌉` 列的 tile。tile space 中的索引，例如 `(i, j)`，用来指定要 load 哪个 tile。load 操作会返回形状为 `(t_m, t_n)` 的 tile，其中包含数组中对应的元素。

当 tile 超出数组边界时，例如数组维度不是 tile 维度的整数倍而导致边缘部分越界，load 操作会指定越界元素的处理方式，例如填零，见[图 9](#figure-09)。

<a id="figure-09"></a>

![图 9：tile space 以及数组与 tile 之间的数据移动](images/chapter-01/figure-09.png)

*图 9：tile space 与数据移动。形状为 `(M, N)` 的二维数组，在概念上被划分为形状为 `(t_m, t_n)` 的 tile 组成的网格。在 tile space 索引 `(i, j)` 处执行 load，会返回对应的 tile。在数组边界处，超出数组范围的元素可以用零填充。store 操作则将一个 tile 写回数组中指定的 tile space 索引位置。*

store 操作执行相反的过程：给定一个 tile 及其在 tile space 中的索引，将 tile 的元素写入数组对应区域。任何超出数组边界的写入都会被直接丢弃。tile 程序还支持 gather（聚集）和 scatter（散布）操作，可从数组中的任意位置 load 数据，或向任意位置 store 数据。

<a id="sec-1-2-2-3-3"></a>

##### 1.2.2.3.3 tile 操作

tile 程序提供了一组作用于 tile 的内置操作，包括逐元素算术、矩阵乘法、沿一个或多个轴的归约（例如求和、求最大值）、形状操作（例如 reshape 和 transpose）以及类型转换。当两个不同形状的 tile 参与同一个操作时，较小的 tile 会在操作执行之前自动扩展，以匹配较大的 tile。

<a id="sec-1-2-2-3-4"></a>

##### 1.2.2.3.4 与 SIMT 编程的关系

tile programming 与 SIMT 编程在 CUDA 中共存。一个应用程序可以同时包含 SIMT kernel 和 tile kernel，两类 kernel 都能操作 device memory 中的同一份数据。采用哪种编程模型，可以针对每个 kernel 分别决定。tile programming 并不会取代 SIMT 编程。

SIMT 提供对单个 thread 的细粒度控制，对某些算法和优化技术而言，这种控制仍然必不可少。tile programming 提供了更高层次的抽象，可以简化 kernel 开发。由于 thread 级决策交给编译器处理，同一个 tile kernel 无需修改源代码，就能运行在不同的 GPU 架构上。两种模型都建立在前面介绍的相同底层硬件之上，即 SM、block 和 grid；也都使用下一节介绍的相同 device memory 空间。

<a id="sec-1-2-3"></a>

### 1.2.3 GPU memory

在现代计算系统中，高效利用内存与充分利用执行计算的功能单元同样重要。异构系统具有多个 memory space；除了 cache，GPU 还包含多种可编程 on-chip memory（片上内存）。下面各小节将更详细地介绍这些 memory space。

<a id="sec-1-2-3-1"></a>

#### 1.2.3.1 异构系统中的 DRAM memory

GPU 和 CPU 都有直接连接的 DRAM 芯片。在包含多个 GPU 的系统中，每个 GPU 都有自己的内存。从 device code 的角度看，直接连接 GPU 的 DRAM 称为**global memory（全局内存）**，因为 GPU 内的所有 SM 都可以访问它。这里的“全局”并不意味着它必然能从系统中任何位置访问。直接连接 CPU 的 DRAM 称为**system memory（系统内存）**或**host memory（主机内存）**。

与 CPU 一样，GPU 也使用 virtual memory addressing。在本版指南所支持的所有系统上，CPU 与 GPU 使用同一个 unified virtual address space。这意味着，系统中每个 GPU 的 virtual address 范围都是唯一的，与 CPU 以及其他 GPU 的地址范围互不重叠。对于给定的 virtual memory address，可以确定该地址位于 GPU memory 还是 system memory；在多 GPU 系统中，还可以确定它属于哪个 GPU 的内存。

CUDA 提供了分配 GPU memory、CPU memory，以及在不同分配区域之间复制数据的 API。复制可以发生在 CPU 与 GPU 之间、同一 GPU 内部，或多 GPU 系统的不同 GPU 之间。需要时，可以显式控制数据的存放位置。下面介绍的 Unified Memory，则允许 CUDA runtime 或系统硬件自动处理内存的放置。

<a id="sec-1-2-3-2"></a>

#### 1.2.3.2 locality domain

**locality domain（局部性域）**是 GPU memory 和 SM 的一个子集。所有 GPU 都至少有一个 locality domain，较大的 GPU 可能有多个。在拥有多个 locality domain 的 GPU 上，使用局部化的内存与计算，可以提升某些 kernel 的性能。

**localized memory（局部化内存）**是完全位于单个 locality domain 内的 memory allocation。使用这部分内存的 kernel 可以被调度到同一个 locality domain 内的 SM 上。当应用程序将内存与计算划分到不同 locality domain，并且能够向各个域并行提交工作时，就能从局部化中受益。

分配 GPU memory 时，如果没有指定 locality domain，则称其为**unlocalized（未局部化）**。除了非常小的分配之外，unlocalized memory allocation 会跨越多个 locality domain。

对于只有一个 locality domain 的 GPU，所有 SM 和内存都属于这个域。[第 4.7 节 Locality Domains](CUDA%E7%BC%96%E7%A8%8B%E6%8C%87%E5%8D%97-%E7%AC%AC4%E7%AB%A0-CUDA%E5%8A%9F%E8%83%BD.md#sec-4-7)介绍了将 GPU memory 与计算局部化到特定域的方法。locality domain 是一项性能特性：即使 CUDA 代码不对内存或计算进行局部化，仍然可以正确执行。

<a id="sec-1-2-3-3"></a>

#### 1.2.3.3 GPU 中的 on-chip memory

除了 global memory，每个 GPU 还有一些 on-chip memory。每个 SM 都拥有自己的 register file 和 shared memory。这些内存是 SM 的组成部分，在该 SM 上执行的 thread 可以非常快速地访问它们。

register file 存放 thread-local 变量，通常由编译器分配。shared memory 可由 block 或 cluster 中的所有 thread 访问，可用于在 block 或 cluster 内交换数据。

SM 内 register file 和 unified data cache 的大小都是有限的。各 SM 的 register file 大小、unified data cache 大小，以及如何配置 unified data cache 在 L1 与 shared memory 之间的容量平衡，参见[第 5.1.3 节 Features and Technical Specifications 中的表 31：Memory Information per Compute Capability](CUDA%E7%BC%96%E7%A8%8B%E6%8C%87%E5%8D%97-%E7%AC%AC5%E7%AB%A0-%E6%8A%80%E6%9C%AF%E9%99%84%E5%BD%95.md#sec-5-1-3)。register file、shared memory 空间以及 L1 cache 由 block 中的所有 thread 共同使用。

要将一个 block 调度到某个 SM 上，每个 thread 所需的 register 数量乘以 block 中的 thread 数量，必须小于或等于该 SM 的可用 register 数量。如果一个 block 需要的 register 数量超过 register file 的容量，kernel 就无法启动；此时必须减少 block 中的 thread 数量，才能使其可启动。

shared memory 按 block 分配。也就是说，与按 thread 分配 register 不同，shared memory 分配属于整个 block。

<a id="sec-1-2-3-3-1"></a>

##### 1.2.3.3.1 cache

除了可编程内存，GPU 还包含 L1 和 L2 cache。每个 SM 都有自己的 L1 cache，它是 unified data cache 的一部分。GPU 内所有 SM 则共享一个更大的 L2 cache。[图 2](#figure-02) 的 GPU 框图展示了这种组织方式。

每个 SM 还有单独的 constant cache（常量缓存），用于缓存 global memory 中被声明为在 kernel 整个生命周期内保持不变的值。编译器也可能将 kernel 参数放入 constant memory（常量内存）。这样，kernel 参数就能缓存在 SM 中与 L1 data cache 分离的 cache 内，从而可能提升 kernel 性能。

<a id="sec-1-2-3-4"></a>

#### 1.2.3.4 Unified Memory

当应用程序显式地在 GPU 或 CPU 上分配内存时，这部分内存只供运行在相应设备上的代码访问。也就是说，CPU memory 只能由 CPU 代码访问，而 GPU memory 只能由在 GPU 上运行的 kernel 访问。[^2] 应用程序使用 CPU 与 GPU 之间的 memory copy API，在适当时机显式地将数据复制到正确的内存中。

CUDA 的**Unified Memory（统一内存）**功能，允许应用程序分配 CPU 和 GPU 均可访问的内存。CUDA runtime 或底层硬件会在需要时提供访问能力，或者将数据迁移到正确位置。即使使用 Unified Memory，要获得最佳性能，也应尽量减少 memory migration，并尽可能让数据由直接连接其所在内存的处理器访问。

不同 memory space 之间如何访问和交换数据，取决于系统的硬件特性。[第 2.6.2 节 Unified Memory](CUDA%E7%BC%96%E7%A8%8B%E6%8C%87%E5%8D%97-%E7%AC%AC2%E7%AB%A0-%E4%BD%BF%E7%94%A8CUDA%E7%BC%96%E7%A8%8BGPU.md#sec-2-6-2)介绍了不同类型的 Unified Memory 系统；[第 4.1 节 Unified Memory](CUDA%E7%BC%96%E7%A8%8B%E6%8C%87%E5%8D%97-%E7%AC%AC4%E7%AB%A0-CUDA%E5%8A%9F%E8%83%BD.md#sec-4-1)则更详细地介绍了各种场景下 Unified Memory 的用法与行为。

<a id="sec-1-3"></a>

## 1.3 CUDA 平台

NVIDIA CUDA 平台由许多软硬件组件构成，并包含为异构系统计算而开发的多项重要技术。本节介绍其中一些应用程序开发者需要理解的基本概念和组件。与[第 1.2 节编程模型](#sec-1-2)一样，本节不针对特定编程语言，而适用于所有使用 CUDA 平台的程序。

<a id="sec-1-3-1"></a>

### 1.3.1 Compute Capability 与 SM 版本

每个 NVIDIA GPU 都有一个**Compute Capability（计算能力，CC）**编号，表明该 GPU 支持哪些功能，并规定其部分硬件参数。这些规格记录在[第 5.1 节 Compute Capabilities](CUDA%E7%BC%96%E7%A8%8B%E6%8C%87%E5%8D%97-%E7%AC%AC5%E7%AB%A0-%E6%8A%80%E6%9C%AF%E9%99%84%E5%BD%95.md#sec-5-1)中。NVIDIA 的 [CUDA GPU Compute Capability](https://developer.nvidia.com/cuda-gpus) 页面维护了所有 NVIDIA GPU 及其 Compute Capability 的列表。

Compute Capability 采用 `X.Y` 的格式，由主版本号和次版本号组成，其中 `X` 是主版本号，`Y` 是次版本号。例如，CC 12.0 的主版本号是 12，次版本号是 0。Compute Capability 与 SM 的版本号直接对应。例如，CC 12.0 GPU 内的 SM 版本为 `sm_120`。这个版本用于标记二进制文件。

[第 5.1.1 节 Obtain the GPU Compute Capability](CUDA%E7%BC%96%E7%A8%8B%E6%8C%87%E5%8D%97-%E7%AC%AC5%E7%AB%A0-%E6%8A%80%E6%9C%AF%E9%99%84%E5%BD%95.md#sec-5-1-1)介绍了如何查询并确定系统中 GPU 的 Compute Capability。

<a id="sec-1-3-2"></a>

### 1.3.2 CUDA Toolkit 与 NVIDIA 驱动程序

可以将 NVIDIA 驱动程序理解为 GPU 的操作系统。它是必须安装在 host 操作系统上的软件组件，任何 GPU 用途都需要它，包括显示和图形功能。NVIDIA 驱动程序是 CUDA 平台的基础。除了 CUDA，它还提供 Vulkan、Direct3D 等其他使用 GPU 的方式。NVIDIA 驱动程序使用诸如 `r580` 这样的版本号。

**CUDA Toolkit** 是用于编写、构建和分析 GPU 计算软件的一组库、头文件及工具。CUDA Toolkit 与 NVIDIA 驱动程序是两个独立的软件产品。

**CUDA runtime（CUDA 运行时）**是 CUDA Toolkit 所提供库中的一个特殊组件。它提供 API 和一些语言扩展，用于处理常见任务，例如分配内存、在 GPU 与其他 GPU 或 CPU 之间复制数据，以及启动 kernel。CUDA runtime 的 API 部分称为 **CUDA Runtime API**。

[CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html) 文档详细说明了不同 GPU、NVIDIA 驱动程序与 CUDA Toolkit 版本之间的兼容性。

<a id="sec-1-3-2-1"></a>

#### 1.3.2.1 CUDA Runtime API 与 CUDA Driver API

CUDA Runtime API 建立在更底层的 **CUDA Driver API** 之上，后者是由 NVIDIA 驱动程序公开的 API。本指南主要介绍 CUDA Runtime API 提供的接口。如果需要，只使用 Driver API 也能实现相同的功能；有些功能仅通过 Driver API 提供。应用程序可以使用任意一种 API，也可以将两者互操作地结合使用。[第 3.3 节 The CUDA Driver API](CUDA%E7%BC%96%E7%A8%8B%E6%8C%87%E5%8D%97-%E7%AC%AC3%E7%AB%A0-CUDA%E8%BF%9B%E9%98%B6.md#sec-3-3)介绍了两者之间的互操作。

CUDA Runtime API 各函数的完整参考，参见 [CUDA Runtime API Documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html)。

CUDA Driver API 各函数的完整参考，参见 [CUDA Driver API Documentation](https://docs.nvidia.com/cuda/cuda-driver-api/index.html)。

<a id="sec-1-3-3"></a>

### 1.3.3 PTX

CUDA 平台中一个基础但有时对开发者不可见的层，是**PTX（Parallel Thread Execution，并行线程执行）虚拟指令集架构（ISA）**。PTX 是面向 NVIDIA GPU 的高级汇编语言，在真实 GPU 硬件的物理指令集之上提供抽象层。与其他平台类似，也可以直接使用这种汇编语言编写应用程序，但这可能给软件开发增加不必要的复杂度和难度。

领域专用语言及高级语言编译器可以生成 PTX 代码，将其作为中间表示（Intermediate Representation，IR），然后使用 NVIDIA 的离线或即时（Just-in-Time，JIT）编译工具，生成可执行的 GPU 二进制代码。这样，除了 NVIDIA 工具（例如[第 2.7 节 NVCC: The NVIDIA CUDA Compiler](CUDA%E7%BC%96%E7%A8%8B%E6%8C%87%E5%8D%97-%E7%AC%AC2%E7%AB%A0-%E4%BD%BF%E7%94%A8CUDA%E7%BC%96%E7%A8%8BGPU.md#sec-2-7)所介绍的 NVCC）直接支持的语言外，其他语言也能用于 CUDA 平台编程。

由于 GPU 能力会随着时间变化和增强，PTX 虚拟 ISA 规范也具有版本。PTX 版本与 SM 版本类似，对应于某个 Compute Capability。例如，支持 Compute Capability 8.0 全部功能的 PTX 被称为 `compute_80`。

PTX 的完整文档参见 [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)。

<a id="sec-1-3-4"></a>

### 1.3.4 Cubin 与 Fatbin

CUDA 应用程序和库通常使用 C++ 等较高级语言编写。这些语言先编译为 PTX，然后 PTX 再被编译为针对物理 GPU 的实际二进制代码，称为 **CUDA binary**，简称 **cubin**。Cubin 使用特定的二进制格式，面向某个具体的 SM 版本，例如 `sm_120`。

使用 GPU 计算的可执行文件和库二进制文件，同时包含 CPU 与 GPU 代码。GPU 代码存放在一个称为 **fatbin** 的容器中。Fatbin 可以包含面向多个不同目标的 cubin 和 PTX。例如，一个应用程序可以同时构建面向多种 GPU 架构，即不同 SM 版本的二进制代码。应用程序运行时，其 GPU 代码会被加载到某个具体的 GPU 上，并使用 fatbin 中最适合该 GPU 的二进制版本。

<a id="figure-10"></a>

![图 10：可执行文件或库中的 CPU 代码与 GPU fatbin](images/chapter-01/figure-10.png)

*图 10：可执行文件或库的二进制文件，同时包含 CPU 二进制代码，以及用于 GPU 代码的 fatbin 容器。Fatbin 可以同时包含 cubin 格式的 GPU 二进制代码和 PTX 虚拟 ISA 代码。PTX 代码可通过 JIT 编译，支持未来的目标架构。*

Fatbin 还可以包含一个或多个 PTX 版本的 GPU 代码，其用途见[第 1.3.4.2 节 PTX 兼容性](#sec-1-3-4-2)。[图 10](#figure-10) 展示了一个应用程序或库二进制文件，其中包含多个 cubin 版本的 GPU 代码，以及一个 PTX 版本的代码。

<a id="sec-1-3-4-1"></a>

#### 1.3.4.1 二进制兼容性

NVIDIA GPU 在某些条件下保证二进制兼容。具体来说，在 Compute Capability 主版本号相同的情况下，如果 GPU 的 Compute Capability 次版本号大于或等于 cubin 的目标次版本号，那么该 GPU 就能加载并执行该 cubin。例如，如果一个应用程序包含针对 Compute Capability 8.6 编译的 cubin，那么它可以在 Compute Capability 为 8.6 或 8.9 的 GPU 上加载和执行。但它不能在 Compute Capability 为 8.0 的 GPU 上加载，因为 GPU 的 CC 次版本号 0 低于代码的次版本号 6。

不同 Compute Capability 主版本之间不具备二进制兼容性。也就是说，针对 Compute Capability 8.6 编译的 cubin，不能在 Compute Capability 为 9.0 的 GPU 上加载。

讨论二进制代码时，通常会使用类似上述示例中的 `sm_86` 来表示代码版本。这相当于说，该二进制代码是面向 Compute Capability 8.6 构建的。开发者向 NVIDIA CUDA 编译器 `nvcc` 指定二进制构建目标时，就使用这种写法，因此经常采用这一简称。

> **注意**：二进制兼容性保证仅适用于由 `nvcc` 等 NVIDIA 工具生成的二进制文件。不支持手工修改或自行生成 NVIDIA GPU 二进制代码。如果以任何方式修改二进制文件，兼容性保证即失效。

<a id="sec-1-3-4-2"></a>

#### 1.3.4.2 PTX 兼容性

GPU 代码可以以二进制或 PTX 形式存储在可执行文件中，参见[第 1.3.4 节 Cubin 与 Fatbin](#sec-1-3-4)。如果应用程序保存了 GPU 代码的 PTX 版本，那么运行时就可以针对 Compute Capability 等于或高于该 PTX 目标 Compute Capability 的 GPU，对其进行 JIT 编译。例如，如果应用程序包含面向 `compute_80` 的 PTX，那么运行时就可以将其 JIT 编译为面向 `sm_120` 等后续 SM 版本的代码。这使应用程序和库无需重新构建，也能向前兼容未来的 GPU。

<a id="sec-1-3-4-3"></a>

#### 1.3.4.3 JIT 编译

应用程序在运行时加载的 PTX 代码，由设备驱动程序编译为二进制代码。这称为**JIT 编译（即时编译）**。JIT 编译会增加应用程序加载时间，但能使应用程序受益于新设备驱动程序所包含的编译器改进。它还允许应用程序运行在其编译时尚不存在的设备上。

设备驱动程序为应用程序 JIT 编译 PTX 时，会自动缓存一份生成的二进制代码，避免后续启动应用程序时重复编译。该 cache 称为**compute cache（计算缓存）**。设备驱动程序升级时，compute cache 会自动失效，使应用程序能够受益于新驱动程序内置 JIT 编译器的改进。

与 CUDA 的早期版本相比，运行时何时以及如何对 PTX 进行 JIT 编译的限制已经放宽，开发者可以更加灵活地决定是否对部分或全部 kernel 执行 JIT，以及何时执行。[第 4.8 节 Lazy Loading（延迟加载）](CUDA%E7%BC%96%E7%A8%8B%E6%8C%87%E5%8D%97-%E7%AC%AC4%E7%AB%A0-CUDA%E5%8A%9F%E8%83%BD.md#sec-4-8)介绍了可用选项，以及如何控制 JIT 行为。此外，还有一些环境变量可以控制 JIT 编译行为，详见[第 5.2 节 CUDA Environment Variables](CUDA%E7%BC%96%E7%A8%8B%E6%8C%87%E5%8D%97-%E7%AC%AC5%E7%AB%A0-%E6%8A%80%E6%9C%AF%E9%99%84%E5%BD%95.md#sec-5-2)。

除了使用 `nvcc` 编译 CUDA C++ device code，还可以使用 **NVRTC** 在运行时将 CUDA C++ device code 编译为 PTX。NVRTC 是 CUDA C++ 的运行时编译库，更多信息参见 [NVRTC User Guide](https://docs.nvidia.com/cuda/nvrtc/index.html)。

<a id="sec-1-3-4-4"></a>

#### 1.3.4.4 binary finalization

从 Compute Capability 10.0 开始，同一 GPU 家族内部的二进制兼容性由 NVIDIA 驱动程序通过软件管理。面向 CC 10.0 及后续架构的 cubin 采用扩展二进制表示的格式，以提供家族内部的二进制兼容性。

针对特定目标架构生成，即完成**finalization（二进制定型）**的二进制代码，可以直接在该目标架构上执行，也可以在运行时重新 finalization，以适配任何具有二进制兼容性保证的 GPU。Compute Capability 为 9.x 及更早版本的 cubin，不支持运行时重新 finalization。

如果一个面向 CC 10.0 或更高版本的 cubin，是为某个目标架构生成的，但被加载到同一 GPU 家族中另一个兼容架构上，那么驱动程序会进行即时 finalization，将 cubin 绑定到目标 GPU 的架构。该过程称为**off-target finalization（非原目标定型）**。

off-target finalization 支持**off-target execution（非原目标执行）**：为一个架构完成 finalization 的 cubin，可以在同一 GPU 家族内另一个兼容架构上运行。如果 cubin 加载到它最初生成时针对的架构上，则属于**on-target execution（原目标执行）**，无需进一步 finalization。

面向 off-target execution 的即时 finalization，与 PTX JIT 编译是不同的过程。PTX JIT 编译将 PTX 代码编译为 GPU 二进制代码，并且已经针对目标架构完成 finalization；cubin 的 off-target finalization，则是将现有 cubin 重新 finalization 到另一个兼容架构。off-target finalization 的开销显著低于 PTX JIT 编译，但与 on-target execution 相比，仍可能增加模块加载时间。

off-target finalization 的开销取决于需要进行 finalization 的代码量、模块加载期间可用的 host CPU 资源，以及源架构和目标架构之间的关系。如果应用程序加载大型模块或大量模块，在 cubin 需要进行 off-target finalization 时，可能会观察到启动时间增加。

与 PTX JIT 编译一样，NVIDIA 驱动程序也可能缓存 cubin off-target finalization 的结果，以避免在应用程序后续运行时重复执行 finalization 工作。该 cache 由[第 5.2.2 节 JIT Compilation](CUDA%E7%BC%96%E7%A8%8B%E6%8C%87%E5%8D%97-%E7%AC%AC5%E7%AB%A0-%E6%8A%80%E6%9C%AF%E9%99%84%E5%BD%95.md#sec-5-2-2)所述的同一组 CUDA cache 环境变量控制。

## 脚注

[^1]: 在使用 CUDA 动态并行等功能的某些情况下，block 可能被挂起并保存到内存中。这意味着 SM 的状态会存入由系统管理的 GPU memory 区域，SM 随后可以执行其他 block。这与 CPU 的上下文切换类似，但并不常见。

[^2]: 一个例外是 mapped memory（映射内存），它是使用特定属性分配的 CPU memory，允许 GPU 直接访问。不过，映射访问需要经过 PCIe 或 NVLINK 连接。GPU 无法通过并行执行隐藏这种访问较高的延迟和较低的带宽，因此 mapped memory 不能作为 Unified Memory 或将数据放在合适 memory space 中的高性能替代方案。
