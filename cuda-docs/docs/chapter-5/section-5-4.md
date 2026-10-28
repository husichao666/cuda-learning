---
title: "5.4 C/C++ Language Extensions"
description: "CUDA 编程指南 · 第 5 章 技术附录"
---

<a id="sec-5-4"></a>

# 5.4 C/C++ Language Extensions

::: info 待翻译
本节已建立目录框架，正文尚未翻译。
:::

<a id="sec-5-4-1"></a>

## 5.4.1 Function and Variable Annotations

<a id="sec-5-4-1-1"></a>

### 5.4.1.1 Execution Space Specifiers

<a id="sec-5-4-1-2"></a>

### 5.4.1.2 Memory Space Specifiers

<a id="sec-5-4-1-2-1"></a>

#### 5.4.1.2.1 __shared__ Memory

<a id="sec-5-4-1-2-2"></a>

#### 5.4.1.2.2 __managed__ Memory

<a id="sec-5-4-1-2-3"></a>

#### 5.4.1.2.3 __tile__ Variables

<a id="sec-5-4-1-3"></a>

### 5.4.1.3 Inlining Specifiers

<a id="sec-5-4-1-4"></a>

### 5.4.1.4 __restrict__ Pointers

<a id="sec-5-4-1-5"></a>

### 5.4.1.5 __grid_constant__ Parameters

<a id="sec-5-4-1-6"></a>

### 5.4.1.6 Annotation Summary

<a id="sec-5-4-2"></a>

## 5.4.2 Built-in Types and Variables

<a id="sec-5-4-2-1"></a>

### 5.4.2.1 Host Compiler Type Extensions

<a id="sec-5-4-2-2"></a>

### 5.4.2.2 Built-in Variables

<a id="sec-5-4-2-3"></a>

### 5.4.2.3 Built-in Types

<a id="sec-5-4-3"></a>

## 5.4.3 Kernel Configuration

<a id="sec-5-4-3-1"></a>

### 5.4.3.1 Thread Block Cluster

<a id="sec-5-4-3-2"></a>

### 5.4.3.2 Launch Bounds

<a id="sec-5-4-3-3"></a>

### 5.4.3.3 Maximum Number of Registers per Thread

<a id="sec-5-4-4"></a>

## 5.4.4 Synchronization Primitives

<a id="sec-5-4-4-1"></a>

### 5.4.4.1 Thread Block Synchronization Functions

<a id="sec-5-4-4-2"></a>

### 5.4.4.2 Warp Synchronization Function

<a id="sec-5-4-4-3"></a>

### 5.4.4.3 Memory Fence Functions

<a id="sec-5-4-5"></a>

## 5.4.5 Atomic Functions

<a id="sec-5-4-5-1"></a>

### 5.4.5.1 Legacy Atomic Functions

<a id="sec-5-4-5-1-1"></a>

#### 5.4.5.1.1 atomicAdd()

<a id="sec-5-4-5-1-2"></a>

#### 5.4.5.1.2 atomicSub()

<a id="sec-5-4-5-1-3"></a>

#### 5.4.5.1.3 atomicInc()

<a id="sec-5-4-5-1-4"></a>

#### 5.4.5.1.4 atomicDec()

<a id="sec-5-4-5-1-5"></a>

#### 5.4.5.1.5 atomicAnd()

<a id="sec-5-4-5-1-6"></a>

#### 5.4.5.1.6 atomicOr()

<a id="sec-5-4-5-1-7"></a>

#### 5.4.5.1.7 atomicXor()

<a id="sec-5-4-5-1-8"></a>

#### 5.4.5.1.8 atomicMin()

<a id="sec-5-4-5-1-9"></a>

#### 5.4.5.1.9 atomicMax()

<a id="sec-5-4-5-1-10"></a>

#### 5.4.5.1.10 atomicExch()

<a id="sec-5-4-5-1-11"></a>

#### 5.4.5.1.11 atomicCAS()

<a id="sec-5-4-5-2"></a>

### 5.4.5.2 Built-in Atomic Functions

<a id="sec-5-4-5-2-1"></a>

#### 5.4.5.2.1 __nv_atomic_fetch_add(), __nv_atomic_add()

<a id="sec-5-4-5-2-2"></a>

#### 5.4.5.2.2 __nv_atomic_fetch_sub(), __nv_atomic_sub()

<a id="sec-5-4-5-2-3"></a>

#### 5.4.5.2.3 __nv_atomic_fetch_and(), __nv_atomic_and()

<a id="sec-5-4-5-2-4"></a>

#### 5.4.5.2.4 __nv_atomic_fetch_or(), __nv_atomic_or()

<a id="sec-5-4-5-2-5"></a>

#### 5.4.5.2.5 __nv_atomic_fetch_xor(), __nv_atomic_xor()

<a id="sec-5-4-5-2-6"></a>

#### 5.4.5.2.6 __nv_atomic_fetch_min(), __nv_atomic_min()

<a id="sec-5-4-5-2-7"></a>

#### 5.4.5.2.7 __nv_atomic_fetch_max(), __nv_atomic_max()

<a id="sec-5-4-5-2-8"></a>

#### 5.4.5.2.8 __nv_atomic_exchange(), __nv_atomic_exchange_n()

<a id="sec-5-4-5-2-9"></a>

#### 5.4.5.2.9 __nv_atomic_compare_exchange(), __nv_atomic_compare_exchange_n()

<a id="sec-5-4-5-2-10"></a>

#### 5.4.5.2.10 __nv_atomic_load(), __nv_atomic_load_n()

<a id="sec-5-4-5-2-11"></a>

#### 5.4.5.2.11 __nv_atomic_store(), __nv_atomic_store_n()

<a id="sec-5-4-5-2-12"></a>

#### 5.4.5.2.12 __nv_atomic_thread_fence()

<a id="sec-5-4-6"></a>

## 5.4.6 Warp Functions

<a id="sec-5-4-6-1"></a>

### 5.4.6.1 Warp Active Mask

<a id="sec-5-4-6-2"></a>

### 5.4.6.2 Warp Vote Functions

<a id="sec-5-4-6-3"></a>

### 5.4.6.3 Warp Match Functions

<a id="sec-5-4-6-4"></a>

### 5.4.6.4 Warp Reduce Functions

<a id="sec-5-4-6-5"></a>

### 5.4.6.5 Warp Shuffle Functions

<a id="sec-5-4-6-6"></a>

### 5.4.6.6 Warp __sync Intrinsic Constraints

<a id="sec-5-4-7"></a>

## 5.4.7 CUDA-Specific Macros

<a id="sec-5-4-7-1"></a>

### 5.4.7.1 __CUDA_ARCH__

<a id="sec-5-4-7-2"></a>

### 5.4.7.2 __CUDA_ARCH_SPECIFIC__ and __CUDA_ARCH_FAMILY_SPECIFIC__

<a id="sec-5-4-7-3"></a>

### 5.4.7.3 CUDA Feature Testing Macros

<a id="sec-5-4-7-4"></a>

### 5.4.7.4 __nv_pure__ Attribute

<a id="sec-5-4-8"></a>

## 5.4.8 CUDA-Specific Functions

<a id="sec-5-4-8-1"></a>

### 5.4.8.1 Address Space Predicate Functions

<a id="sec-5-4-8-2"></a>

### 5.4.8.2 Address Space Conversion Functions

<a id="sec-5-4-8-3"></a>

### 5.4.8.3 Low-Level Load and Store Functions

<a id="sec-5-4-8-4"></a>

### 5.4.8.4 __trap()

<a id="sec-5-4-8-5"></a>

### 5.4.8.5 __nanosleep()

<a id="sec-5-4-8-6"></a>

### 5.4.8.6 Dynamic Programming eXtension (DPX) Instructions

<a id="sec-5-4-9"></a>

## 5.4.9 Compiler Optimization Hints

<a id="sec-5-4-9-1"></a>

### 5.4.9.1 #pragma unroll

<a id="sec-5-4-9-2"></a>

### 5.4.9.2 __builtin_assume_aligned()

<a id="sec-5-4-9-3"></a>

### 5.4.9.3 __builtin_assume() and __assume()

<a id="sec-5-4-9-4"></a>

### 5.4.9.4 __builtin_constant_p()

<a id="sec-5-4-9-5"></a>

### 5.4.9.5 __builtin_expect()

<a id="sec-5-4-9-6"></a>

### 5.4.9.6 __builtin_unreachable()

<a id="sec-5-4-9-7"></a>

### 5.4.9.7 Custom ABI Pragmas

<a id="sec-5-4-9-8"></a>

### 5.4.9.8 MMA Throughput Pragma

<a id="sec-5-4-10"></a>

## 5.4.10 Debugging and Diagnostics

<a id="sec-5-4-10-1"></a>

### 5.4.10.1 Assertion

<a id="sec-5-4-10-2"></a>

### 5.4.10.2 Breakpoint Function

<a id="sec-5-4-10-3"></a>

### 5.4.10.3 Diagnostic Pragmas

<a id="sec-5-4-11"></a>

## 5.4.11 Warp Matrix Functions

<a id="sec-5-4-11-1"></a>

### 5.4.11.1 Description

<a id="sec-5-4-11-2"></a>

### 5.4.11.2 Alternate Floating Point

<a id="sec-5-4-11-3"></a>

### 5.4.11.3 Double Precision

<a id="sec-5-4-11-4"></a>

### 5.4.11.4 Sub-byte Operations

<a id="sec-5-4-11-5"></a>

### 5.4.11.5 Restrictions

<a id="sec-5-4-11-6"></a>

### 5.4.11.6 Element Types and Matrix Sizes

<a id="sec-5-4-11-7"></a>

### 5.4.11.7 Example
