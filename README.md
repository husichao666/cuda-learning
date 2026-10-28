# 算子学习与实践

从 CUDA 执行模型和基础算子出发，逐步学习访存、归约、Triton、GEMM、Attention，以及训练框架中的算子接入。每个阶段同时保留学习笔记、可运行代码、正确性检查和实验记录。

## 学习入口

- [学习路线](docs/learning-path.md)：16 周的建议安排，按实际验收进度推进。
- [第一周笔记](notebook/第一周-硬件介绍与入门代码.md)：GPU 硬件、线程组织、向量加法、边界处理与计时。
- [算子目录](operators/README.md)：已有实现与后续方向。
- [向量加法示例](operators/cuda/vector_add/README.md)：五份可独立编译或粘贴到在线编辑器运行的 CUDA 程序。
- [CUDA 中文参考文档站](cuda-docs/README.md)：第一、二章已翻译，第三至六部分目前为目录框架。

## 当前进度

| 内容 | 状态 |
| --- | --- |
| 第一周笔记 | 已整理 |
| CUDA 向量加法、边界与随机测试 | 已在 LeetGPU FUNCTIONAL 模式运行通过 |
| CUDA events 与完整调用计时示例 | 已验证可运行，尚无真实 NVIDIA GPU 性能结论 |
| 访存、归约与后续训练算子 | 按学习路线逐步补充 |

## 快速开始

阅读笔记和运行仓库检查不需要 GPU。仓库检查使用 Python 3.9 或更新版本，且不依赖额外 Python 包：

```sh
python3 scripts/check_repository.py
```

在有 NVIDIA GPU、驱动和 CUDA Toolkit 的环境中，从仓库根目录编译、验证第一周示例：

```sh
make cuda
make test-cuda
```

计时示例单独运行：

```sh
make benchmark-cuda
```

`make cuda` 默认使用 `nvcc -std=c++17 -O3 -arch=native`。目标架构可通过 `NVCCFLAGS` 调整，详见[环境说明](docs/environment.md)。没有本地 GPU 时，可以将单个 `.cu` 文件粘贴到 [LeetGPU Playground](https://leetgpu.com/playground) 学习和验证功能；记录执行模式，再解释计时结果。

文档站使用 Node.js 24 与 pnpm 11.25.0。在仓库根目录执行：

```sh
pnpm --dir cuda-docs install --frozen-lockfile
pnpm --dir cuda-docs dev
```

## 目录结构

```text
.
├── docs/                  # 学习路线、环境、实验与目录说明
├── notebook/              # 按学习阶段整理的笔记和博客稿
├── operators/             # 可运行算子、测试、计时与验证记录
│   └── cuda/vector_add/   # 第一周 CUDA 向量加法
├── cuda-docs/             # CUDA 中文参考文档站，保留现有页面路由
├── references/            # 参考来源、历史译稿与本地资料说明
├── scripts/               # 不依赖 GPU 的仓库检查
├── .github/workflows/     # 仓库检查与文档站构建
├── Makefile               # 检查、编译、测试与计时入口
├── CONTRIBUTING.md        # 内容与代码贡献约定
├── THIRD_PARTY_NOTICES.md # 第三方来源与许可边界
└── LICENSE                # 原创内容的 MIT 许可
```

旧文件的位置和内容维护入口见[目录说明](docs/repository-layout.md)。原 PDF、Word 和网站预览图保存在 `references/local/`，默认不进入 Git。

## 实验记录与贡献

新增算子时，一起提交公式与接口说明、参考实现、边界测试、运行命令和验证记录。性能报告明确写出设备、输入、精度、计时范围和波动；功能模拟环境的时间不作为真实 GPU 性能证据。

具体流程见[贡献指南](CONTRIBUTING.md)与[实验记录约定](docs/experiments.md)。当前 GitHub Actions 只执行仓库检查和文档站构建，不声称已进行 GPU 测试。

## 许可与来源

本仓库原创代码和原创文字采用 [MIT License](LICENSE)。NVIDIA 文档原文、译文中的第三方内容、原始配图及其他依赖不在本仓库 MIT 授权范围内，保留各自权利与许可，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

本项目是独立学习记录，与 NVIDIA 无隶属或官方背书关系。
