# 仓库目录与维护入口

## 内容职责

`notebook/` 保存学习文章，`operators/` 保存可独立运行的代码，`docs/` 保存学习路线和实验规范。参考译文的网站继续放在 `cuda-docs/`，保持现有页面路由与 `sec-*` 锚点。

文档站后续翻译直接修改 `cuda-docs/docs/chapter-*/section-*.md`。历史长章译稿保存在 `references/cuda-programming-guide/legacy/`，用于追溯，不与网站正文双向同步。

## 本次整理的路径变更

| 原位置 | 新位置 |
| --- | --- |
| 根目录的 `大模型训练算子开发与优化学习路径.md` | [docs/learning-path.md](learning-path.md) |
| 根目录的 `CUDA编程指南-*.md` | [references/cuda-programming-guide/legacy](../references/cuda-programming-guide/legacy/README.md) |
| 根目录的 `images/` | `references/cuda-programming-guide/legacy/images/`，与旧译稿一起移动 |
| 根目录的 `cuda-programming-guide.pdf` | `references/local/cuda-programming-guide.pdf`，本地保留、Git 忽略 |
| 根目录的学习路线 Word 文件 | `references/local/`，本地保留、Git 忽略 |
| `cuda-docs/preview*.png` | `references/local/site-previews/`，本地保留、Git 忽略 |

第一周笔记路径保持为 `notebook/第一周-硬件介绍与入门代码.md`。网站源码、主题、图片和页面路径保持原有组织；迁移与检查脚本已使用新的历史资料位置。

## 新增算子的组织方式

后续按后端和算子组织，例如 `operators/cuda/transpose/`、`operators/triton/rmsnorm/`。只为实际开始实现的内容建立目录，计划中的算子先登记在学习路线和算子目录中。

构建产物统一写入根目录 `build/` 或对应工具的忽略目录。个人参考资料放在 `references/local/`，避免与可维护源码混在一起。
