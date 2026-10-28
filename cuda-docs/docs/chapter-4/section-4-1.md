---
title: "4.1 Unified Memory"
description: "CUDA 编程指南 · 第 4 章 CUDA 功能"
---

<a id="sec-4-1"></a>

# 4.1 Unified Memory

::: info 待翻译
本节已建立目录框架，正文尚未翻译。
:::

<a id="sec-4-1-1"></a>

## 4.1.1 Unified Memory on Devices with Full CUDA Unified Memory Support

<a id="sec-4-1-1-1"></a>

### 4.1.1.1 Unified Memory: In-Depth Examples

<a id="sec-4-1-1-1-1"></a>

#### 4.1.1.1.1 File-backed Unified Memory

<a id="sec-4-1-1-1-2"></a>

#### 4.1.1.1.2 Inter-Process Communication (IPC) with Unified Memory

<a id="sec-4-1-1-2"></a>

### 4.1.1.2 Performance Tuning

<a id="sec-4-1-1-2-1"></a>

#### 4.1.1.2.1 Memory Paging and Page Sizes

<a id="sec-4-1-1-2-1-1"></a>

##### 4.1.1.2.1.1 Choosing the Right Page Size

<a id="sec-4-1-1-2-1-2"></a>

##### 4.1.1.2.1.2 CPU and GPU Page Tables: Hardware Coherency vs. Software Coherency

<a id="sec-4-1-1-2-1-3"></a>

##### 4.1.1.2.1.3 Mixing Hardware and Software Coherency

<a id="sec-4-1-1-2-2"></a>

#### 4.1.1.2.2 Direct Unified Memory Access from the Host

<a id="sec-4-1-1-2-3"></a>

#### 4.1.1.2.3 Host Native Atomics

<a id="sec-4-1-1-2-4"></a>

#### 4.1.1.2.4 Atomic Accesses and Synchronization Primitives

<a id="sec-4-1-1-2-5"></a>

#### 4.1.1.2.5 Memcpy()/Memset() Behavior With Unified Memory

<a id="sec-4-1-1-2-6"></a>

#### 4.1.1.2.6 Overview of Memory Allocators for Unified Memory

<a id="sec-4-1-1-2-7"></a>

#### 4.1.1.2.7 Locality Domains and Unified Memory

<a id="sec-4-1-1-2-8"></a>

#### 4.1.1.2.8 Access Counter Migration

<a id="sec-4-1-1-2-9"></a>

#### 4.1.1.2.9 Avoid Frequent Writes to GPU-Resident Memory from the CPU

<a id="sec-4-1-1-2-10"></a>

#### 4.1.1.2.10 Exploiting Asynchronous Access to System Memory

<a id="sec-4-1-2"></a>

## 4.1.2 Unified Memory on Devices with only CUDA Managed Memory Support

<a id="sec-4-1-3"></a>

## 4.1.3 Unified Memory on Windows, WSL, and Tegra

<a id="sec-4-1-3-1"></a>

### 4.1.3.1 Multi-GPU

<a id="sec-4-1-3-2"></a>

### 4.1.3.2 Coherency and Concurrency

<a id="sec-4-1-3-3"></a>

### 4.1.3.3 Stream Associated Unified Memory

<a id="sec-4-1-3-3-1"></a>

#### 4.1.3.3.1 Stream Callbacks

<a id="sec-4-1-3-3-2"></a>

#### 4.1.3.3.2 Managed memory associated to streams allows for finer-grained control

<a id="sec-4-1-3-3-3"></a>

#### 4.1.3.3.3 A more elaborate example on multithreaded host programs

<a id="sec-4-1-3-3-4"></a>

#### 4.1.3.3.4 Data Movement of Stream Associated Unified Memory

<a id="sec-4-1-4"></a>

## 4.1.4 Performance Hints

<a id="sec-4-1-4-1"></a>

### 4.1.4.1 Data Prefetching

<a id="sec-4-1-4-2"></a>

### 4.1.4.2 Data Usage Hints

<a id="sec-4-1-4-3"></a>

### 4.1.4.3 Memory Discarding

<a id="sec-4-1-4-4"></a>

### 4.1.4.4 Querying Data Usage Attributes on Managed Memory

<a id="sec-4-1-4-5"></a>

### 4.1.4.5 GPU Memory Oversubscription
