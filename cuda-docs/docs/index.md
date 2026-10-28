---
title: CUDA Programming Guide 中文版
outline: false
---

<script setup>
import { withBase } from 'vitepress'
</script>

# CUDA Programming Guide 中文版

按章节学习 CUDA。左侧选择章节，右侧查看当前页面的小节目录；使用顶部搜索可以查找术语与小节标题。

基于 NVIDIA《CUDA Programming Guide》13.4.2 整理。正文使用中文，block、grid、warp、kernel 及内存术语保留英文，配图来自原 PDF。

::: info 翻译进度与来源
这是独立学习项目整理的非官方中文译文。第一、二章正文已翻译，第三至六部分目前为目录框架。原始文档和配图权利归 NVIDIA 等相应权利人所有，技术细节请以 [NVIDIA 官方原文](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) 为准。
:::

[开始阅读：1.1 引言 →](/chapter-1/section-1-1)

<div class="chapter-list">
  <div class="chapter-card">
    <span class="eyebrow">CHAPTER 01 · 已翻译</span>
    <a :href="withBase('/chapter-1/')">CUDA 简介</a>
    <p>GPU、编程模型、memory 与 CUDA 平台。</p>
  </div>
  <div class="chapter-card">
    <span class="eyebrow">CHAPTER 02 · 已翻译</span>
    <a :href="withBase('/chapter-2/')">使用 CUDA 编程 GPU</a>
    <p>CUDA C++、Python、SIMT、tile 与异步执行。</p>
  </div>
  <div class="chapter-card">
    <span class="eyebrow">CHAPTER 03 · 目录框架</span>
    <a :href="withBase('/chapter-3/')">CUDA 进阶</a>
    <p>硬件执行、同步、Driver API 与多 GPU。</p>
  </div>
  <div class="chapter-card">
    <span class="eyebrow">CHAPTER 04 · 目录框架</span>
    <a :href="withBase('/chapter-4/')">CUDA 功能</a>
    <p>Unified Memory、Graphs、Cooperative Groups 等专题。</p>
  </div>
  <div class="chapter-card">
    <span class="eyebrow">CHAPTER 05 · 目录框架</span>
    <a :href="withBase('/chapter-5/')">技术附录</a>
    <p>Compute Capability、语言扩展、API 与执行模型。</p>
  </div>
  <div class="chapter-card">
    <span class="eyebrow">PART 06 · 目录框架</span>
    <a :href="withBase('/chapter-6/')">声明</a>
    <p>Notice、OpenCL 与商标声明。</p>
  </div>
</div>
