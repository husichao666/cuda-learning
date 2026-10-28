---
title: "4.3 Stream-Ordered Memory Allocator"
description: "CUDA 编程指南 · 第 4 章 CUDA 功能"
---

<a id="sec-4-3"></a>

# 4.3 Stream-Ordered Memory Allocator

::: info 待翻译
本节已建立目录框架，正文尚未翻译。
:::

<a id="sec-4-3-1"></a>

## 4.3.1 Introduction

<a id="sec-4-3-2"></a>

## 4.3.2 Memory Management

<a id="sec-4-3-2-1"></a>

### 4.3.2.1 Allocating Memory

<a id="sec-4-3-2-2"></a>

### 4.3.2.2 Freeing Memory

<a id="sec-4-3-3"></a>

## 4.3.3 Memory Pools

<a id="sec-4-3-3-1"></a>

### 4.3.3.1 Default/Implicit Pools

<a id="sec-4-3-3-2"></a>

### 4.3.3.2 Explicit Pools

<a id="sec-4-3-3-3"></a>

### 4.3.3.3 Device Accessibility for Multi-GPU Support

<a id="sec-4-3-3-4"></a>

### 4.3.3.4 Enabling Memory Pools for IPC

<a id="sec-4-3-3-4-1"></a>

#### 4.3.3.4.1 Creating and Sharing IPC Memory Pools

<a id="sec-4-3-3-4-2"></a>

#### 4.3.3.4.2 Set Access in the Importing Process

<a id="sec-4-3-3-4-3"></a>

#### 4.3.3.4.3 Creating and Sharing Allocations from an Exported Pool

<a id="sec-4-3-3-4-4"></a>

#### 4.3.3.4.4 IPC Export Pool Limitations

<a id="sec-4-3-3-4-5"></a>

#### 4.3.3.4.5 IPC Import Pool Limitations

<a id="sec-4-3-4"></a>

## 4.3.4 Best Practices and Tuning

<a id="sec-4-3-4-1"></a>

### 4.3.4.1 Query for Support

<a id="sec-4-3-4-2"></a>

### 4.3.4.2 Physical Page Caching Behavior

<a id="sec-4-3-4-3"></a>

### 4.3.4.3 Resource Usage Statistics

<a id="sec-4-3-4-4"></a>

### 4.3.4.4 Memory Reuse Policies

<a id="sec-4-3-4-4-1"></a>

#### 4.3.4.4.1 cudaMemPoolReuseFollowEventDependencies

<a id="sec-4-3-4-4-2"></a>

#### 4.3.4.4.2 cudaMemPoolReuseAllowOpportunistic

<a id="sec-4-3-4-4-3"></a>

#### 4.3.4.4.3 cudaMemPoolReuseAllowInternalDependencies

<a id="sec-4-3-4-4-4"></a>

#### 4.3.4.4.4 Disabling Reuse Policies

<a id="sec-4-3-4-5"></a>

### 4.3.4.5 Synchronization API Actions

<a id="sec-4-3-5"></a>

## 4.3.5 Addendums

<a id="sec-4-3-5-1"></a>

### 4.3.5.1 cudaMemcpyAsync Current Context/Device Sensitivity

<a id="sec-4-3-5-2"></a>

### 4.3.5.2 cudaPointerGetAttributes Query

<a id="sec-4-3-5-3"></a>

### 4.3.5.3 cudaGraphAddMemsetNode

<a id="sec-4-3-5-4"></a>

### 4.3.5.4 Pointer Attributes

<a id="sec-4-3-5-5"></a>

### 4.3.5.5 CPU Virtual Memory
