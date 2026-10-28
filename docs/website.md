# 网站发布与维护

## 在线入口

- [CUDA 学习笔记](https://husichao666.github.io/cuda-learning/)：学习主页、文章目录与笔记正文。
- [CUDA Programming Guide 中文版](https://husichao666.github.io/cuda-learning/guide/)：独立的参考文档站，保留章节导航与全文搜索。

两个站点共用一个 GitHub Pages 发布流程。学习笔记使用蓝色主题，中文指南保留绿色主题，导航栏可以互相跳转。中文指南是独立学习项目整理的非官方译文，翻译进度和来源在首页说明。

## 新增与修改文章

直接编辑 `notebook/` 中的 Markdown 文章。新增文章使用独立文件，第一行写一级标题；构建时自动生成文章目录与侧栏，按文件路径排序。`README.md` 作为仓库索引，不作为独立文章发布。

笔记正文中的代码保持原样。指向其他笔记的相对链接转换为站内链接，指向算子源码等仓库文件的链接转换为 GitHub 链接；相对路径引用的图片会复制到静态资源目录。不要直接编辑 `cuda-docs/notes/articles/`，这是自动生成并被 Git 忽略的目录。

中文指南继续直接编辑 `cuda-docs/docs/chapter-*/`。网站首页、导航和样式分别位于 `cuda-docs/notes/` 与 `cuda-docs/docs/` 中，原有指南小节的 `sec-*` 锚点保持不变。

## 构建与发布

使用 Node.js 24、pnpm 11.25.0 和 Python 3.9 或更新版本，在仓库根目录执行：

```sh
pnpm --dir cuda-docs install --frozen-lockfile
pnpm --dir cuda-docs pages:build
```

两套页面生成到 `build/pages/`，并检查子路径、文章目录、站内链接、锚点、脚本、样式与图片。需要本地预览时，从仓库根目录执行：

```sh
pnpm --dir cuda-docs pages:preview
```

然后打开 <http://localhost:4173/cuda-learning/>，预览两套站点与文章。两套站点之间的顶部导航链接指向线上地址；预览指南首页可直接访问 <http://localhost:4173/cuda-learning/guide/>。

推送到 `main` 会触发 [Pages 工作流](../.github/workflows/pages.yml)，先构建并检查两套站点，再发布到 GitHub Pages。Pull request 只构建和检查，不发布。可在仓库 Actions 中查看结果，也可以手动运行该工作流。
