# 贡献指南

欢迎提交笔记勘误、可运行示例和带证据的算子实验。修改围绕一个清楚的主题展开，并在说明中写出改动、验证方式与尚未验证的环境。

## 内容位置

- 学习路线和运行规范放在 `docs/`。
- 面向读者的学习笔记放在 `notebook/`，标题使用陈述式短语，代码出现前交代目的和上下文。
- 可运行算子放在 `operators/<backend>/<operator>/`，每个算子目录包含自己的 README。
- CUDA 参考译文直接维护 `cuda-docs/docs/`，保留已有 `sec-*` 锚点和页面路由。
- `references/cuda-programming-guide/legacy/` 保留迁移前译稿，用于追溯；新增正文不再写入该归档。

## 算子提交内容

每个算子说明公式、输入输出、shape、dtype、布局、空输入策略，以及不支持的情况。先提供可解释的基线和参考结果，再增加优化版本。

正确性同时覆盖常见形状与边界形状，固定随机种子。说明容差依据；内存安全检查和数值检查分别记录。性能实验写出环境、完整命令、计时边界、重复次数和波动，报告变慢或没有收益的情况。记录格式见 [docs/experiments.md](docs/experiments.md)。

## 提交前验证

从仓库根目录执行不依赖 GPU 的检查：

```sh
make check
```

修改参考文档站时，额外执行：

```sh
pnpm --dir cuda-docs install --frozen-lockfile
make docs-check
```

有 CUDA 环境时，修改第一周示例后执行：

```sh
make test-cuda
```

性能相关修改再运行 `make benchmark-cuda`，记录真实环境与结果。如果没有 NVIDIA GPU，请明确写出只做了哪些检查，不将未运行的 GPU 测试标为通过。

第一周博客与 `vector_add.cu` 的完整示例保持一致，`make check` 会检查这一点。其他独立示例也应与文章中的修改步骤同步维护。

修改笔记、文档或网站时，运行 `pnpm --dir cuda-docs pages:build` 验证两套在线页面。文章与发布流程见[网站维护说明](docs/website.md)；推送到 `main` 后自动更新 GitHub Pages。

## 资料与许可

原创贡献按本仓库 MIT 许可提供。引用第三方内容时保留来源和许可，不将 NVIDIA 原图、译文涉及的原始内容或依赖改标为本仓库 MIT。新增来源同步写入 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

PDF 下载、个人导出、构建结果、设备日志和环境配置放在被忽略的位置；可复现实验所需的精简、无敏感信息的结果可以提交到对应算子的 `validation/` 目录。
