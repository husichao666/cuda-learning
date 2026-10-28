# 算子目录

## 已有实现

| 后端 | 算子 | 学习内容 | 验证情况 |
| --- | --- | --- | --- |
| CUDA C++ | [Vector Add](cuda/vector_add/README.md) | 执行模型、索引、边界、数据搬运与计时 | LeetGPU FUNCTIONAL 模式通过；真实 GPU 性能待测 |

## 后续方向

按[学习路线](../docs/learning-path.md)逐步增加转置、行归约、Softmax、RMSNorm、SwiGLU、GEMM 和 Attention。已经完成的算子再加入上表，并链接到独立 README 和验证记录。

新增目录采用 `operators/<backend>/<operator>/`。每个目录给出接口、依赖、运行命令、reference、正确性与性能记录，具体约定见[贡献指南](../CONTRIBUTING.md)。
