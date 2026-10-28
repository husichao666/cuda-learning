# 第三方资料与许可说明

根目录 [LICENSE](LICENSE) 适用于本仓库拥有授权权利的原创代码与原创文字，不改变第三方原文、配图、商标、示例和依赖本身的许可。

## NVIDIA CUDA Programming Guide

- 原始来源：[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)。
- 当前整理版本：Release 13.4.2；具体内容以原文版本和各页面标注为准。
- 涉及目录：`cuda-docs/docs/chapter-*/`、`cuda-docs/docs/public/images/`、`references/cuda-programming-guide/legacy/`。
- 笔记中引用的 NVIDIA 官方图片也属于第三方内容，即使图片通过外部链接展示。

上述内容包括翻译、目录整理和原图引用。原始内容的著作权及相关权利归 NVIDIA 或相应权利人所有，第三方内容不因收入本仓库而改为 MIT 许可。使用、复制和再分发时，应遵守 NVIDIA 原始文档的声明与适用条款；本仓库不额外授予这些第三方内容的权利。

文档站保留了[原文声明部分的目录入口](cuda-docs/docs/chapter-6/index.md)，这些页面尚未全部翻译。实际条款请阅读 NVIDIA [Notices](https://docs.nvidia.com/cuda/cuda-programming-guide/notices.html)，不要将中文整理稿当作条款原文。

## 文档站依赖

文档站使用 [VitePress](https://github.com/vuejs/vitepress) 和 [markdown-it-footnote](https://github.com/markdown-it/markdown-it-footnote) 等第三方包，各自按其发行包中的许可使用。确切依赖版本由 `cuda-docs/pnpm-lock.yaml` 记录；依赖内容不会由本仓库的 MIT 许可重新授权。

## 本地参考文件

`references/local/` 用于保存个人下载的 PDF、Word 导出和预览截图，默认被 Git 忽略。这些本地文件不会随普通 Git 提交自动发布。参考文档的公开入口和版本说明见 [references/README.md](references/README.md)。

新增第三方资料时，请在对应页面注明来源、作者或权利人及适用许可，并同步更新本文件。
