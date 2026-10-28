# CUDA 向量加法

计算一维 FP32 数组的逐元素加法：`c[i] = a[i] + b[i]`。一个线程处理一个元素，尾部通过 `i < n` 检查；CPU 端在 `n = 0` 时直接返回。

输入约定：`n >= 0`，非空时输入输出指向足够大的连续 GPU 存储区，三个数组互不重叠。示例使用 CPU reference 检查结果。

配套文章：[第一周：硬件介绍与入门代码](../../../notebook/第一周-硬件介绍与入门代码.md)。

## 文件与用途

| 文件 | 内容 |
| --- | --- |
| [vector_add.cu](vector_add.cu) | 文章中的完整程序，默认验证 `n = 1025` |
| [test_boundaries.cu](test_boundaries.cu) | 验证空输入、warp/block 边界与大输入，共 10 个长度 |
| [test_random.cu](test_random.cu) | 固定种子 2026，使用随机输入验证同一组长度 |
| [benchmark_events.cu](benchmark_events.cu) | 50 次预热，5 组各 1000 次，使用 CUDA events 计时 |
| [benchmark_roundtrip.cu](benchmark_roundtrip.cu) | CPU 时钟测量输入传输、提交、等待和结果回传 |

每个文件都有完整的 `main()`，可以单独复制到在线编辑器。学习阶段保留独立程序，便于逐步对照文章，不依赖公共头文件或框架。

## 编译与运行

从仓库根目录执行：

```sh
make cuda
make test-cuda
make benchmark-cuda
```

单独编译最小示例：

```sh
nvcc -std=c++17 -O3 -arch=native operators/cuda/vector_add/vector_add.cu -o /tmp/vector_add
/tmp/vector_add
```

正确性通过时输出 `n=1025: PASS`，进程退出状态为 0。边界与随机测试覆盖 `0, 1, 31, 32, 33, 255, 256, 257, 1025, 1000003`。

## 验证记录

五个版本已于 2026-10-28 在 LeetGPU FUNCTIONAL 模式编译并运行成功，边界和随机测试各 10 个用例全部通过。原始控制台记录见 [validation/leetgpu-functional.json](validation/leetgpu-functional.json)。

该模式报告的是 Intel CPU 设备。这些结果用于验证示例在该执行模式下的可运行性，不作为真实 NVIDIA GPU 的性能证据；当前没有真实 GPU 的验证报告。

CUDA events 示例报告的是批次平均时间的中位数，区间可能包含 CPU 提交空档，且反复使用同一组缓冲区。完整调用计时包含 H2D、kernel 提交和等待、D2H，显存分配与正确性校验在区间外。环境与记录规范见 [docs/experiments.md](../../../docs/experiments.md)。
