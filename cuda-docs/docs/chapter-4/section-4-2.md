---
title: "4.2 CUDA Graphs"
description: "CUDA 编程指南 · 第 4 章 CUDA 功能"
---

<a id="sec-4-2"></a>

# 4.2 CUDA Graphs

::: info 待翻译
本节已建立目录框架，正文尚未翻译。
:::

<a id="sec-4-2-1"></a>

## 4.2.1 Graph Structure

<a id="sec-4-2-1-1"></a>

### 4.2.1.1 Node Types

<a id="sec-4-2-1-2"></a>

### 4.2.1.2 Edge Data

<a id="sec-4-2-2"></a>

## 4.2.2 Building and Running Graphs

<a id="sec-4-2-2-1"></a>

### 4.2.2.1 Graph Creation

<a id="sec-4-2-2-1-1"></a>

#### 4.2.2.1.1 Graph APIs

<a id="sec-4-2-2-1-2"></a>

#### 4.2.2.1.2 Stream Capture

<a id="sec-4-2-2-1-2-1"></a>

##### 4.2.2.1.2.1 Cross-stream Dependencies and Events

<a id="sec-4-2-2-1-2-2"></a>

##### 4.2.2.1.2.2 Prohibited and Unhandled Operations

<a id="sec-4-2-2-1-2-3"></a>

##### 4.2.2.1.2.3 Invalidation

<a id="sec-4-2-2-1-2-4"></a>

##### 4.2.2.1.2.4 Capture Introspection

<a id="sec-4-2-2-1-3"></a>

#### 4.2.2.1.3 Putting It All Together

<a id="sec-4-2-2-2"></a>

### 4.2.2.2 Graph Instantiation

<a id="sec-4-2-2-3"></a>

### 4.2.2.3 Graph Execution

<a id="sec-4-2-3"></a>

## 4.2.3 Updating Instantiated Graphs

<a id="sec-4-2-3-1"></a>

### 4.2.3.1 Whole Graph Update

<a id="sec-4-2-3-2"></a>

### 4.2.3.2 Individual Node Update

<a id="sec-4-2-3-3"></a>

### 4.2.3.3 Individual Node Enable

<a id="sec-4-2-3-4"></a>

### 4.2.3.4 Graph Update Limitations

<a id="sec-4-2-4"></a>

## 4.2.4 Conditional Graph Nodes

<a id="sec-4-2-4-1"></a>

### 4.2.4.1 Conditional Handles

<a id="sec-4-2-4-2"></a>

### 4.2.4.2 Conditional Node Body Graph Requirements

<a id="sec-4-2-4-3"></a>

### 4.2.4.3 Conditional IF Nodes

<a id="sec-4-2-4-4"></a>

### 4.2.4.4 Conditional WHILE Nodes

<a id="sec-4-2-4-5"></a>

### 4.2.4.5 Conditional SWITCH Nodes

<a id="sec-4-2-5"></a>

## 4.2.5 Graph Memory Nodes

<a id="sec-4-2-5-1"></a>

### 4.2.5.1 Introduction

<a id="sec-4-2-5-2"></a>

### 4.2.5.2 API Fundamentals

<a id="sec-4-2-5-2-1"></a>

#### 4.2.5.2.1 Graph Node APIs

<a id="sec-4-2-5-2-2"></a>

#### 4.2.5.2.2 Stream Capture

<a id="sec-4-2-5-2-3"></a>

#### 4.2.5.2.3 Accessing and Freeing Graph Memory Outside of the Allocating Graph

<a id="sec-4-2-5-2-4"></a>

#### 4.2.5.2.4 cudaGraphInstantiateFlagAutoFreeOnLaunch

<a id="sec-4-2-5-2-5"></a>

#### 4.2.5.2.5 Memory Nodes in Child Graphs

<a id="sec-4-2-5-3"></a>

### 4.2.5.3 Optimized Memory Reuse

<a id="sec-4-2-5-3-1"></a>

#### 4.2.5.3.1 Address Reuse within a Graph

<a id="sec-4-2-5-3-2"></a>

#### 4.2.5.3.2 Physical Memory Management and Sharing

<a id="sec-4-2-5-4"></a>

### 4.2.5.4 Performance Considerations

<a id="sec-4-2-5-4-1"></a>

#### 4.2.5.4.1 First Launch / cudaGraphUpload

<a id="sec-4-2-5-5"></a>

### 4.2.5.5 Physical Memory Footprint

<a id="sec-4-2-5-6"></a>

### 4.2.5.6 Peer Access

<a id="sec-4-2-5-6-1"></a>

#### 4.2.5.6.1 Peer Access with Graph Node APIs

<a id="sec-4-2-5-6-2"></a>

#### 4.2.5.6.2 Peer Access with Stream Capture

<a id="sec-4-2-6"></a>

## 4.2.6 Device Graph Launch

<a id="sec-4-2-6-1"></a>

### 4.2.6.1 Device Graph Creation

<a id="sec-4-2-6-1-1"></a>

#### 4.2.6.1.1 Device Graph Requirements

<a id="sec-4-2-6-1-2"></a>

#### 4.2.6.1.2 Device Graph Upload

<a id="sec-4-2-6-1-3"></a>

#### 4.2.6.1.3 Device Graph Update

<a id="sec-4-2-6-2"></a>

### 4.2.6.2 Device Launch

<a id="sec-4-2-6-2-1"></a>

#### 4.2.6.2.1 Fire and Forget Launch

<a id="sec-4-2-6-2-1-1"></a>

##### 4.2.6.2.1.1 Graph Execution Environments

<a id="sec-4-2-6-2-2"></a>

#### 4.2.6.2.2 Tail Launch

<a id="sec-4-2-6-2-2-1"></a>

##### 4.2.6.2.2.1 Tail Self-launch

<a id="sec-4-2-6-2-3"></a>

#### 4.2.6.2.3 Sibling Launch

<a id="sec-4-2-7"></a>

## 4.2.7 Using Graph APIs

<a id="sec-4-2-8"></a>

## 4.2.8 CUDA User Objects
