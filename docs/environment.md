# 运行环境

## 阅读与仓库检查

阅读 Markdown、整理代码和运行仓库检查不需要 GPU。检查脚本使用 Python 3.9 或更新版本的标准库：

```sh
python3 scripts/check_repository.py
```

## CUDA 示例

真实 CUDA 实验需要 NVIDIA GPU、兼容的驱动和 CUDA Toolkit。先记录环境：

```sh
nvidia-smi
nvcc --version
```

从仓库根目录执行：

```sh
make cuda
make test-cuda
make benchmark-cuda
```

默认构建产物位于 `build/cuda/vector_add/`，不会进入 Git。默认编译选项 `-arch=native` 由支持它的 `nvcc` 根据当前可见 GPU 生成代码；若交叉编译或需要显式选择架构，可覆盖选项，例如针对 T4 的编译目标：

```sh
make cuda NVCCFLAGS='-std=c++17 -O3 -arch=sm_75'
```

`sm_75` 只是一个具体目标示例，不适用于所有设备。请结合目标 GPU 与 Toolkit 支持范围选择参数。没有 NVIDIA GPU 的 Mac 可以阅读、编辑和执行仓库检查，CUDA 编译与运行放到远程环境。

## 在线练习

可以把 `operators/cuda/vector_add/` 中任意一个 `.cu` 文件完整粘贴到 [LeetGPU Playground](https://leetgpu.com/playground) 运行，每个文件都有独立的 `main()`。

本仓库第一周示例已在 LeetGPU 的 FUNCTIONAL 模式验证，记录见 [验证日志](../operators/cuda/vector_add/validation/leetgpu-functional.json)。日志中报告的是 Intel CPU 设备，不能将这些耗时解释为真实 NVIDIA GPU 性能。

## 参考文档站

文档站固定使用 VitePress 1.6.4 和已有锁文件。推荐环境为 Node.js 24、pnpm 11.25.0：

```sh
pnpm --dir cuda-docs install --frozen-lockfile
pnpm --dir cuda-docs dev
```

构建与检查：

```sh
make docs-check
```

打开浏览器访问启动命令打印的地址。macOS 也可以使用 `cuda-docs/start.command`，它要求 `node` 和依赖已经安装，不再依赖某台机器上的私有运行时路径。
