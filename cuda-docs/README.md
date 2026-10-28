# CUDA 学习文档站

[在线阅读 CUDA Programming Guide 中文版](https://husichao666.github.io/cuda-learning/guide/) · [学习笔记](https://husichao666.github.io/cuda-learning/)

使用开源 [VitePress](https://github.com/vuejs/vitepress) 1.6.4（MIT）构建。网站支持章节侧栏、页内目录、全文搜索、深浅色切换及上一节/下一节导航。

## 打开网站

macOS 可双击 `start.command`。启动后访问终端显示的本地地址，通常为 <http://127.0.0.1:5173>。按 Control+C 停止服务。

使用 Node.js 24 和 pnpm 11.25.0 运行；依赖版本由锁文件固定：

```sh
pnpm install --frozen-lockfile
pnpm dev
```

## 维护内容

- `docs/chapter-1/` 至 `docs/chapter-6/`：每章一个目录。
- 每章 `index.md`：本章概览与目录。
- `section-2-1.md` 等文件：2.1、2.2 等小节各占一个页面，更深的标题保留在页内目录。
- `docs/.vitepress/sidebar.json`：左侧章节导航。
- `docs/public/images/chapter-01/`：PDF 中提取的原始配图。
- 标题上方的 `sec-*` 锚点用于跨页面引用，编辑标题时应保留。

后续翻译直接编辑对应的小节 Markdown，并删除页面中的“待翻译”提示。六份长章 Markdown 已归档到 [references/cuda-programming-guide/legacy](../references/cuda-programming-guide/legacy/README.md)，网站不会自动用旧译稿覆盖正文。`scripts/migrate.py` 是一次性迁移工具，目标页面存在时会拒绝覆盖。

第二章已翻译至 2.7，位于 `docs/chapter-2/section-2-1.md` 至 `section-2-7.md`，包括代码示例、PDF 原图和跨节链接。第二章原图位于 `docs/public/images/chapter-02/`。

第一章的两个脚注位于 `docs/chapter-1/section-1-2.md` 底部；第一章图 1–10 与全部章节引用已迁移为站内链接。

## 构建和检查

```sh
pnpm build
pnpm check
pnpm preview
```

构建产物在 `docs/.vitepress/dist/`，为纯静态 HTML、CSS 与 JavaScript。`scripts/check_links.py` 检查生成页面中的本地链接和锚点、947 个原书编号标题、48 个小节页面及原图一致性。`pnpm pages:build` 会同时构建学习笔记与中文版指南，为两个站点设置各自的发布路径，详见[网站维护说明](../docs/website.md)。

原文来源：NVIDIA《CUDA Programming Guide》Release 13.4.2。现有第一、二章正文已翻译，第 3–6 部分仍为目录框架。原文及配图权利归各自权利人所有。

本目录是[算子学习仓库](../README.md)中的参考文档站。原创网站代码采用仓库 MIT 许可，NVIDIA 原文、译文涉及的原始内容和原图的来源与许可范围见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。
