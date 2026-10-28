---
title: "5.6 Device-Callable APIs and Intrinsics"
description: "CUDA 编程指南 · 第 5 章 技术附录"
---

<a id="sec-5-6"></a>

# 5.6 Device-Callable APIs and Intrinsics

::: info 待翻译
本节已建立目录框架，正文尚未翻译。
:::

<a id="sec-5-6-1"></a>

## 5.6.1 Memory Barrier Primitives Interface

<a id="sec-5-6-1-1"></a>

### 5.6.1.1 Data Types

<a id="sec-5-6-1-2"></a>

### 5.6.1.2 Memory Barrier Primitives API

<a id="sec-5-6-2"></a>

## 5.6.2 Pipeline Primitives Interface

<a id="sec-5-6-2-1"></a>

### 5.6.2.1 memcpy_async Primitive

<a id="sec-5-6-2-2"></a>

### 5.6.2.2 Commit Primitive

<a id="sec-5-6-2-3"></a>

### 5.6.2.3 Wait Primitive

<a id="sec-5-6-2-4"></a>

### 5.6.2.4 Arrive On Barrier Primitive

<a id="sec-5-6-3"></a>

## 5.6.3 Cooperative Groups API

<a id="sec-5-6-3-1"></a>

### 5.6.3.1 cooperative_groups.h

<a id="sec-5-6-3-1-1"></a>

#### 5.6.3.1.1 class thread_block

<a id="sec-5-6-3-1-2"></a>

#### 5.6.3.1.2 class cluster_group

<a id="sec-5-6-3-1-3"></a>

#### 5.6.3.1.3 class grid_group

<a id="sec-5-6-3-1-4"></a>

#### 5.6.3.1.4 class thread_block_tile

<a id="sec-5-6-3-1-5"></a>

#### 5.6.3.1.5 class coalesced_group

<a id="sec-5-6-3-2"></a>

### 5.6.3.2 cooperative_groups/async.h

<a id="sec-5-6-3-2-1"></a>

#### 5.6.3.2.1 memcpy_async

<a id="sec-5-6-3-2-2"></a>

#### 5.6.3.2.2 wait and wait_prior

<a id="sec-5-6-3-3"></a>

### 5.6.3.3 cooperative_groups/partition.h

<a id="sec-5-6-3-3-1"></a>

#### 5.6.3.3.1 tiled_partition

<a id="sec-5-6-3-3-2"></a>

#### 5.6.3.3.2 labeled_partition

<a id="sec-5-6-3-3-3"></a>

#### 5.6.3.3.3 binary_partition

<a id="sec-5-6-3-4"></a>

### 5.6.3.4 cooperative_groups/reduce.h

<a id="sec-5-6-3-4-1"></a>

#### 5.6.3.4.1 Reduce Operators

<a id="sec-5-6-3-4-2"></a>

#### 5.6.3.4.2 reduce

<a id="sec-5-6-3-5"></a>

### 5.6.3.5 cooperative_groups/scan.h

<a id="sec-5-6-3-5-1"></a>

#### 5.6.3.5.1 inclusive_scan and exclusive_scan

<a id="sec-5-6-3-6"></a>

### 5.6.3.6 cooperative_groups/sync.h

<a id="sec-5-6-3-6-1"></a>

#### 5.6.3.6.1 barrier_arrive and barrier_wait

<a id="sec-5-6-3-6-2"></a>

#### 5.6.3.6.2 sync

<a id="sec-5-6-3-6-2-1"></a>

##### 5.6.3.6.2.1 Grid Synchronization

<a id="sec-5-6-4"></a>

## 5.6.4 CUDA Device Runtime

<a id="sec-5-6-4-1"></a>

### 5.6.4.1 Including Device Runtime API in CUDA Code

<a id="sec-5-6-4-2"></a>

### 5.6.4.2 Memory in the CUDA Device Runtime

<a id="sec-5-6-4-2-1"></a>

#### 5.6.4.2.1 Configuration Options

<a id="sec-5-6-4-2-2"></a>

#### 5.6.4.2.2 Allocation and Lifetime

<a id="sec-5-6-4-2-2-1"></a>

##### 5.6.4.2.2.1 Memory Declarations

<a id="sec-5-6-4-2-2-2"></a>

##### 5.6.4.2.2.2 Device and Constant Memory

<a id="sec-5-6-4-2-2-3"></a>

##### 5.6.4.2.2.3 Textures and Surfaces

<a id="sec-5-6-4-2-2-4"></a>

##### 5.6.4.2.2.4 Shared Memory Variable Declarations

<a id="sec-5-6-4-2-2-5"></a>

##### 5.6.4.2.2.5 Constant Memory

<a id="sec-5-6-4-2-2-6"></a>

##### 5.6.4.2.2.6 Symbol Addresses

<a id="sec-5-6-4-3"></a>

### 5.6.4.3 SM Id and Warp Id

<a id="sec-5-6-4-4"></a>

### 5.6.4.4 Launch Setup APIs

<a id="sec-5-6-4-5"></a>

### 5.6.4.5 Device Management

<a id="sec-5-6-4-6"></a>

### 5.6.4.6 API Reference

<a id="sec-5-6-4-7"></a>

### 5.6.4.7 API Errors and Launch Failures

<a id="sec-5-6-4-8"></a>

### 5.6.4.8 Device Runtime Streams

<a id="sec-5-6-4-8-1"></a>

#### 5.6.4.8.1 The Implicit (NULL) Stream

<a id="sec-5-6-4-8-2"></a>

#### 5.6.4.8.2 The Fire-and-Forget Stream

<a id="sec-5-6-4-8-3"></a>

#### 5.6.4.8.3 The Tail Launch Stream

<a id="sec-5-6-4-9"></a>

### 5.6.4.9 ECC Errors
