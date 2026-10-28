# 大模型训练算子开发与优化学习路径

面向具有 Megatron 背景的开发者    16 周实践计划    2026 年 10 月

目标是在熟悉训练框架的基础上，掌握单 GPU 内的工作划分、数据搬运、数值计算与性能分析，最终完成一个可接入训练链路的算子。路线以 CUDA 建立硬件直觉，以 Triton 完成常见训练算子，再进入 GEMM、Attention 和 Megatron 集成。

### 学习范围与投入

按每周 8–12 小时规划，总投入约 128–192 小时。周数是建议节奏，进入下一阶段以验收结果为准。默认使用 Linux 与 NVIDIA GPU；本地 Mac 可用于阅读和编辑，CUDA 实验在远程 GPU 上执行。先固定一张卡和一套环境，避免同时处理多代架构差异。

已有的模型结构、自动微分、TP 和 PP 知识可以直接复用。新增重点是线程与内存映射、归约、Tensor Core、反向传播的并行实现，以及可靠的性能证据。前半程以 FP32 与 BF16 为主，FP16 按需要补充；FP8、MoE 和通信计算融合放在基础路线完成后。

| 阶段 | 周次 | 主要产出 |
| --- | --- | --- |
| GPU 基础与测量 | 1–4 | CUDA 基础算子、benchmark、首次 profile |
| 常见训练算子 | 5–8 | RMSNorm 与 SwiGLU 的前向和反向 |
| GEMM 与训练形状 | 9–11 | 分块 GEMM、Tensor Core、三类 GEMM 对比 |
| Attention | 12–14 | 分块前向、反向、重计算实验 |
| 训练链路集成 | 15–16 | 真实热点替换、数值与吞吐报告 |

### 完成后的能力标准

- 能从张量公式推导线程或 tile 的划分，并估算计算量、访存量和临时显存。
- 能实现并验证 forward、输入梯度和参数梯度，说明精度与确定性的边界。
- 能解释优化在不同 shape 下的收益和退化，并区分 kernel 加速与训练 step 加速。

推荐学习节奏：每周约 2 小时阅读，5–7 小时编码与实验，1–3 小时记录和复盘。每个阶段保留一个正确基线、至少一个优化版本和一份实验记录。

## 第一阶段  第 1 至 4 周

### 第 1 周  CUDA 执行模型与可靠计时

学习 grid、block、warp、SM、SIMT、kernel launch 和异步执行。写 vector add，处理尾部越界，画出连续线程访问的地址。只补当前需要的 C++：指针、数组、模板基础、编译与错误检查。[1]

任务：建立 PyTorch reference、固定随机种子、warmup 和 CUDA events 计时；把编译、内存分配与数据传输排除在纯 kernel 测量之外，同时单独记录完整调用成本。

验收：覆盖空输入策略、非整齐长度和大输入；能解释 CPU 计时为什么可能只测到提交开销；交付 vector add 源码和可重复运行的 benchmark。

### 第 2 周  访存与矩阵转置

学习 global memory、L2、shared memory、寄存器、合并访存和 bank conflict。实现直接转置，再实现 shared-memory tiled transpose；比较 padding 等布局调整前后的结果。[1][2]

任务：用相同 shape、dtype 和计时方法比较各版本；估算一次读加一次写的必要流量。数据规模要覆盖缓存可容纳与明显大于缓存的情况，避免把缓存带宽误认成显存带宽。

验收：能画出一个 warp 的读写地址；能解释为什么转置的读和写可能有不同的访存效率；至少保留一个优化无效或退化的例子。

### 第 3 周  Reduction 与同步

实现行归约 sum 和 max，从 shared-memory 树形归约推进到 warp shuffle，再处理跨 warp 的结果汇总。学习 block barrier、warp 同步边界，以及普通 kernel 内不能随意实现全 grid 屏障的原因。

验收：正确处理 mask、非 2 的幂长度、全负数 max 和较大归约维度；理解浮点加法顺序改变会带来误差。用 Compute Sanitizer 的 memcheck、racecheck、synccheck 检查适用问题。[5]

### 第 4 周  性能分析与 Triton 入门

学习 Triton program、tl.arange、mask、stride 和 num_warps；重写 vector add 与行归约。Triton 的 program 可作为一块数据的工作单元来理解，具体线程布局仍需结合生成代码和 profile 判断。[3]

用 Nsight Systems 看调用时间线，用 Nsight Compute 深入一个 kernel。先看内存与计算吞吐、寄存器、shared memory、occupancy 和 spill；依据证据再看 stall 原因。[4]

验收：提交一页 profile 报告，包含瓶颈假设、证据、一次修改、多个 shape 的结果。能区分带宽受限、算力受限、启动开销和并行度不足；不以 occupancy 越高作为唯一目标。

## 第二阶段  第 5 至 8 周

### 第 5 周  Softmax 与 RMSNorm 前向

按 Triton 官方教程实现稳定 Softmax，使用减去行最大值的计算形式。随后实现 RMSNorm，学习逐行归约、参数广播、epsilon 和 FP32 中间累加。先完成连续输入版本，再明确哪些 stride 被支持。[3]

任务：比较一行一个 program 与适合小行宽的多行处理；测试不同 token 数和 hidden size。解释减少中间张量如何降低显存流量，以及大行宽如何增加寄存器压力。

验收：对照高精度 reference，覆盖极小值、较大幅值、非整齐维度；记录输出误差和带宽。不能因为一个 shape 快就宣布整个算子更快。

### 第 6 周  RMSNorm 反向与参数梯度

独立推导输入梯度 dX 和缩放参数梯度 dGamma。dX 主要涉及行内归约，dGamma 需要跨 token 归约，两者的并行划分不同。先做可解释的两阶段归约，再评估其他实现。

任务：比较保存归一化统计量与反向重计算的成本；记录 forward、backward 和合计时间。理解原子累加对冲突、精度和确定性的影响，不要把所有参数梯度直接交给原子操作。

验收：用随机上游梯度比较 dX 与 dGamma；数学推导用小规模高精度有限差分校验。若 kernel 不支持 FP64，不把高精度 gradcheck 直接套在低精度 kernel 上作为唯一依据。

### 第 7 周  Residual 与 RMSNorm 融合

实现 residual add + RMSNorm 的训练版本。先写清接口：是否返回 residual 相加后的张量、输出精度、哪些值需要保存供 backward 使用。若其他分支需要 residual 输出，就不能把这次写回当作可消除流量。

任务：画出融合前后的张量读写；验证来自所有输出分支的梯度都被累加。分别测保存激活和重计算的显存、延迟，确认融合没有改变训练语义。

验收：前向与全部梯度通过测试；解释减少的 launch 和读写量，并记录融合是否因寄存器压力而在部分 shape 退化。

### 第 8 周  SwiGLU 与自动微分接入

实现 SwiGLU 的逐元素门控部分及其 backward，先不融合前后 GEMM。把拆分维度、门控分支顺序、stride 和输出布局作为接口契约。学习 PyTorch 自定义算子与 autograd 接入；选择 torch.compile 场景时，再验证其注册要求。[6]

阶段交付：RMSNorm 和 SwiGLU 各有 reference、kernel、正确性测试、benchmark 与性能报告。覆盖 FP32、目标 GPU 支持的 BF16，以及实际训练需要的布局；支持范围外显式报错或走 fallback。

## 第三阶段  第 9 至 11 周

### 第 9 周  GEMM 分块与数据复用

从朴素 CUDA GEMM 出发，依次加入 shared-memory tiling 和 register tiling。说明每个 block、warp、线程负责的输出区域，以及 A、B 的每个 tile 如何被重复使用。初期目标是理解性能差距，不要求超过 cuBLAS。

任务：估算 2MNK 的计算量和必要数据流量；比较多组 tile。解释 tile 变大为什么可能同时提高复用、减少可驻留 block，并加重寄存器压力。

验收：覆盖边界 tile 和不同 M、N、K；保留朴素、shared-memory、register 三个版本，给出延迟与 TFLOPS 曲线或表格。

### 第 10 周  Tensor Core 与流水线

先用 Triton matmul 教程学习 Tensor Core 路径、tile 参数和调优，再选择 CUTLASS C++ 或 CuTe DSL 学习布局及数据搬运。暂时只选一条深入，避免同时承担两套抽象的学习成本。[3][9]

任务：比较不同 num_warps、tile 和流水线配置。区分输入 dtype、乘法精度、累加精度与输出 dtype；与库实现对比时对齐 TF32 等精度设置。

验收：能解释计算和搬运重叠的目的，以及 pipeline stage 增加的资源成本；明确记录比较双方的数值契约。架构专属指令在目标卡支持时再学习。

### 第 11 周  训练中的三类 GEMM

对线性层 Y = XW，令 X 为 M×K、W 为 K×N。前向计算 XW；输入梯度计算 dY·Wᵀ；权重梯度计算 Xᵀ·dY。三者的形状、访问布局和归约维度不同，必须分别测试。

| 方向 | 主要关注点 | 实验要求 |
| --- | --- | --- |
| Forward | 局部 M、N、K 与输出布局 | 从实际训练调用提取 shape |
| 输入梯度 dgrad | 转置访问与上游梯度布局 | 与同精度 reference 对比 |
| 权重梯度 wgrad | 长归约、累加与参数梯度缓冲 | 验证多个 micro-batch 的累加 |

结合 Megatron 检查 TP、sequence parallel、context parallel 等配置下的本地张量。不要直接把全局 batch 或完整 hidden size 代入单卡算子；collective 前后形状也可能不同。梯度累加融合与异步通信相关接口可参考 Megatron Core 文档。[10]

阶段交付：一份来自真实配置的 shape 清单，以及 forward、dgrad、wgrad 的对比报告。每项都写出优化适用范围、精度设置与没有继续优化的原因。

## 第四阶段  第 12 至 14 周

### 第 12 周  Online Softmax 与 Attention 前向

阅读 FlashAttention 的 IO 分析，独立推导分块时如何更新行最大值、归一化因子和输出累加器。先用简单 reference 验证数学，再实现固定 head dimension 的非 causal Triton Attention。[7]

任务：把 QKᵀ、softmax 和 PV 的分块数据流画出来；说明哪些值保留在片上，哪些需要写回。避免在显存中物化完整注意力分数矩阵，同时记录其他中间状态的空间。

验收：与显式 Attention reference 和可用的 PyTorch SDPA 路径比较，记录实际后端；检查缩放因子、累加精度和输出误差。仅凭速度不能认定实现了相同语义。

### 第 13 周  Causal 与工作划分

加入 causal mask、尾部 tile 和多个序列长度。先支持标准 MHA；GQA、变长序列与 packed sequence 在基础版本稳定后扩展。对全 masked 行等边界情况，明确接口约定并专门测试。

阅读 FlashAttention-2，关注并行划分、片上通信与非矩阵乘计算的成本。比较按 batch、head 和 query tile 分配工作的方式，观察小 batch 或少 head 时的并行度问题。[8]

验收：提交 shape 矩阵下的延迟和峰值显存结果；能解释不同 head dimension 与序列长度下性能发生变化的原因。

### 第 14 周  Attention 反向与重计算

推导并实现 dQ、dK、dV。先做不含 dropout 的正确版本，再研究训练配置确实需要的 dropout。理解保存 log-sum-exp 等统计量和重计算概率矩阵之间的关系，以及反向结果归约的并行方式。

若加入 dropout，固定 seed 还不够：必须设计可在 backward 和 activation checkpoint 重计算时恢复的随机数映射。改变 tile 或执行顺序可能改变随机数分配，测试时要区分随机样本差异与公式错误。

验收：使用随机上游梯度验证 dQ、dK、dV；记录 forward + backward 总时间与峰值显存，并检查 checkpoint 开关前后的数值和训练行为。

### 进度控制

Attention backward 是本路线中最容易超时的部分。若超过预算，先用正确但性能普通的分阶段实现完成训练闭环，再继续优化。把生产级的 dropout、变长序列和多架构覆盖列为后续工作，不影响先完成基础毕业项目。

## 第五阶段  第 15 至 16 周

### 第 15 周  从真实训练中选择热点

选择一个已有训练配置，使用 profiler 确认热点、调用次数、局部 shape、dtype、stride 和当前后端。优先选择 residual + RMSNorm 或 SwiGLU 作为毕业项目；已有高度优化的 Attention 后端可作为后续挑战。

先在独立 Transformer block 中接入，再接入小规模 Megatron 训练。确认实际执行路径使用了新 kernel，避免被其他融合实现、编译器或 Transformer Engine 路径替换。保留开关与 fallback，便于对照。

检查 TP 分片、sequence parallel、梯度累加、activation checkpoint、参数梯度缓冲和 stream 语义。修改只围绕选定热点展开，避免同时改变模型配置和优化器行为。

### 第 16 周  数值验证与收益报告

固定初始权重、输入、随机数策略和训练配置，比较 reference 与新实现的输出、梯度、参数更新，以及若干步的 loss 轨迹。检查误差是否异常扩大；短程 smoke test 只能证明基本可运行，不能替代长程收敛验证。

分别测单 kernel、forward + backward、Transformer block 和完整 step。记录 tokens/s、step 时间、峰值显存及测量波动；多卡训练以整体完成时间为准，并关注最慢 rank。

如果一个热点占原 step 时间的 5%，其耗时减半，在其他条件不变且不考虑重叠变化时，整体加速约为 1 ÷ (0.95 + 0.05 ÷ 2) = 1.026，即约 2.6%。实际通信重叠与调度变化还可能改变收益。

### 毕业项目交付清单

- 接口说明：输入输出、支持的 dtype 与布局、精度策略、不支持场景与 fallback。
- 实现：reference、优化 kernel、autograd 或算子注册、训练接入开关。
- 正确性：forward、所有梯度、micro-batch 累加、checkpoint，以及相关边界用例。
- 性能：固定环境下的 shape 扫描、真实调用权重、训练 step 对照与 profile 证据。
- 结论：哪些配置有收益、哪些退化；数值和确定性边界；后续最值得做的一项改进。

### 完成判定

合格结果可以是“在目标形状下获得稳定收益”，也可以是“验证没有端到端收益并解释原因”。必须有可复现证据。学习目标是具备独立开发与判断能力，不预设超过成熟库实现的速度目标。

## 实验规范与每周执行方式

### 正确性先于调优

建立 FP32 或更高精度 reference，分别比较输出、输入梯度和参数梯度。误差阈值按 dtype、归约长度和业务需求确定，先观察 reference 的数值行为，再锁定容差；不能为了让优化版通过而反复放宽阈值。

至少覆盖小尺寸、生产常见尺寸、非整齐尺寸、不同幅值、零输入和支持的非连续布局。空输入、NaN、Inf、全 masked 行等按接口契约处理；未经实现和测试的布局明确拒绝。随机测试与手工构造的边界用例同时保留。

### 性能测量与瓶颈判断

编译和 autotune 先完成，再 warmup；多次重复并报告中位数及波动。统一同步边界、缓存条件、精度设置、是否使用 CUDA Graph，以及是否包含分配。运行 profile 与纯计时分开，Nsight Compute 的 replay 等机制会改变执行条件。[4]

用必要字节数除以延迟估算有效带宽，用计算量除以延迟估算 FLOPS；这不等于硬件计数器测到的真实流量。Roofline 可帮助判断方向，但启动、同步、依赖和并行度不足也会限制性能。

每次只验证一个主要假设，例如减少一次写回、改变 tile、降低寄存器压力。调参使用一组 shape，最终评估增加未参与调参的 shape。报告退化案例，并比较实际使用的融合基线，避免仅对比低效的 eager 算子链。

### 实验记录模板

| 字段 | 需要记录的内容 |
| --- | --- |
| 环境 | GPU、驱动、CUDA、PyTorch、Triton、代码版本 |
| 输入与语义 | shape、stride、dtype、mask、累加精度、是否训练 |
| 假设与改动 | 预计瓶颈、profile 证据、只改变了什么 |
| 结果 | 误差、延迟中位数与波动、显存、带宽或 TFLOPS |
| 解释与下一步 | 哪些变快或变慢、适用边界、下一次实验 |

### 第一周的具体安排

第 1 次：确认 GPU 环境并读执行模型，写 vector add。第 2 次：加入越界检查、reference 与 CUDA events 计时。第 3 次：扫描数组长度与 block size，记录结果。第 4 次：写一页解释，回答线程如何访问数据、计时是否包含异步工作、当前瓶颈是什么。

每周结束用同一标准复盘：我完成了什么，证据是什么，仍不理解什么，下周只优先解决哪一个问题。所有性能结论都保留对应的代码版本和运行配置。

## 阅读顺序与进阶方向

以下优先使用官方文档与原始论文。按当前阶段阅读相关章节，不需要通读全部资料。工具安装与支持范围以目标 GPU 和所选软件版本的文档为准。链接核对日期为 2026 年 10 月 15 日。

[1] [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)。执行模型、SIMT kernel、同步与存储层级。第 1–3 周。

[2] [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)。访存、带宽、资源占用和优化方法。贯穿全程。

[3] [Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/)。Vector Addition、Softmax、LayerNorm、Matmul、Attention。第 4 周起。

[4] [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)。硬件指标、replay、缓存与测量条件。第 4 周起。

[5] [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)。内存越界、shared-memory 竞争与同步错误检查。第 3 周起。

[6] [PyTorch 自定义 C++ 与 CUDA 算子](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)。算子注册、autograd 与框架集成。第 8 周起。

[7] [FlashAttention 原始论文](https://arxiv.org/abs/2205.14135)。IO-aware 分块算法与重计算。第 12 周。

[8] [FlashAttention 2 原始论文](https://arxiv.org/abs/2307.08691)。并行划分与工作分配优化。第 13 周。

[9] [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass)。Tensor Core、CuTe 布局与流水线实现。第 10 周起选读。

[10] [Megatron Core 张量并行线性层](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.tensor_parallel.layers.html)。梯度累加融合和异步通信接口。第 11、15 周。

### 完成基础路线后选择一个方向

- Dense 训练：fused cross entropy、优化器更新与梯度处理；先确认真实占比。
- MoE 训练：token dispatch 与 combine、permutation、grouped GEMM、负载不均。
- 低精度训练：FP8 缩放、amax、转换融合及数值验证，围绕选定硬件展开。
- 架构优化：按目标卡学习异步搬运、TMA、warp specialization 等机制，再评估通信计算融合。

建议首选毕业项目：residual + RMSNorm 的训练版本。它足够接近实际模型，又能完整覆盖归约、融合、参数梯度、混合精度和框架集成。
