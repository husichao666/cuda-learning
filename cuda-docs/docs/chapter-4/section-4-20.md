---
title: "4.20 CUDA Dynamic Parallelism"
description: "CUDA 编程指南 · 第 4 章 CUDA 功能"
---

<a id="sec-4-20"></a>

# 4.20 CUDA Dynamic Parallelism

::: info 待翻译
本节已建立目录框架，正文尚未翻译。
:::

<a id="sec-4-20-1"></a>

## 4.20.1 Introduction

<a id="sec-4-20-1-1"></a>

### 4.20.1.1 Overview

<a id="sec-4-20-2"></a>

## 4.20.2 Execution Environment

<a id="sec-4-20-2-1"></a>

### 4.20.2.1 Parent and Child Grids

<a id="sec-4-20-2-2"></a>

### 4.20.2.2 Scope of CUDA Primitives

<a id="sec-4-20-2-3"></a>

### 4.20.2.3 Streams and Events

<a id="sec-4-20-2-4"></a>

### 4.20.2.4 Ordering and Concurrency

<a id="sec-4-20-3"></a>

## 4.20.3 Memory Coherence and Consistency

<a id="sec-4-20-3-1"></a>

### 4.20.3.1 Global Memory

<a id="sec-4-20-3-2"></a>

### 4.20.3.2 Mapped Memory

<a id="sec-4-20-3-3"></a>

### 4.20.3.3 Shared and Local Memory

<a id="sec-4-20-3-4"></a>

### 4.20.3.4 Local Memory

<a id="sec-4-20-3-4-1"></a>

#### 4.20.3.4.1 Texture Memory

<a id="sec-4-20-4"></a>

## 4.20.4 Programming Interface

<a id="sec-4-20-4-1"></a>

### 4.20.4.1 Basics

<a id="sec-4-20-4-2"></a>

### 4.20.4.2 C++ Language Interface for CDP

<a id="sec-4-20-4-2-1"></a>

#### 4.20.4.2.1 Device-Side Kernel Launch

<a id="sec-4-20-4-2-1-1"></a>

##### 4.20.4.2.1.1 Launches are Asynchronous

<a id="sec-4-20-4-2-1-2"></a>

##### 4.20.4.2.1.2 Launch Environment Configuration

<a id="sec-4-20-4-2-2"></a>

#### 4.20.4.2.2 Events

<a id="sec-4-20-4-2-3"></a>

#### 4.20.4.2.3 Synchronization

<a id="sec-4-20-4-2-4"></a>

#### 4.20.4.2.4 Device Management

<a id="sec-4-20-5"></a>

## 4.20.5 Programming Guidelines

<a id="sec-4-20-5-1"></a>

### 4.20.5.1 Performance

<a id="sec-4-20-5-1-1"></a>

#### 4.20.5.1.1 Dynamic-parallelism-enabled Kernel Overhead

<a id="sec-4-20-5-2"></a>

### 4.20.5.2 Implementation Restrictions and Limitations

<a id="sec-4-20-5-2-1"></a>

#### 4.20.5.2.1 Runtime

<a id="sec-4-20-5-2-1-1"></a>

##### 4.20.5.2.1.1 Memory Footprint

<a id="sec-4-20-5-2-1-2"></a>

##### 4.20.5.2.1.2 Pending Kernel Launches

<a id="sec-4-20-5-3"></a>

### 4.20.5.3 Compatibility and Interoperability

<a id="sec-4-20-6"></a>

## 4.20.6 Device-side Launch from PTX

<a id="sec-4-20-6-1"></a>

### 4.20.6.1 Kernel Launch APIs

<a id="sec-4-20-6-1-1"></a>

#### 4.20.6.1.1 cudaLaunchDevice

<a id="sec-4-20-6-1-2"></a>

#### 4.20.6.1.2 cudaGetParameterBuffer

<a id="sec-4-20-6-2"></a>

### 4.20.6.2 Parameter Buffer Layout
