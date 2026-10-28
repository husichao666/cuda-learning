---
layout: home

hero:
  name: CUDA 学习笔记
  text: 从第一个程序开始，理解 GPU 上的计算。
  tagline: 硬件认识 · 线程分工 · 算子实现 · 实验记录
  actions:
    - theme: brand
      text: 阅读学习笔记
      link: /articles/
    - theme: alt
      text: CUDA Programming Guide 中文版
      link: https://husichao666.github.io/cuda-learning/guide/

features:
  - title: 01 · 硬件与入门代码
    details: 从 SM、block、warp 与 thread 的关系出发，完成向量加法、结果验证与计时。
    link: /articles/第一周-硬件介绍与入门代码
    linkText: 阅读第一周
  - title: 可运行的算子示例
    details: 每个阶段保留源码、边界与随机测试、运行命令和验证记录，边读边做。
    link: https://github.com/husichao666/cuda-learning/tree/main/operators
    linkText: 查看代码
  - title: 持续更新的学习记录
    details: 后续按学习进度补充访存、归约、GEMM 等主题，文章会汇总到笔记目录。
    link: /articles/
    linkText: 查看全部文章
---
