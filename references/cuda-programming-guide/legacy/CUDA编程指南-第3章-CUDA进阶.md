# 第 3 章 CUDA 进阶

<a id="sec-3-1"></a>

## 3.1 Advanced CUDA APIs and Features

<a id="sec-3-1-1"></a>

### 3.1.1 cudaLaunchKernelEx

<a id="sec-3-1-2"></a>

### 3.1.2 Launching Clusters:

<a id="sec-3-1-2-1"></a>

#### 3.1.2.1 Launching with Clusters using cudaLaunchKernelEx

<a id="sec-3-1-2-2"></a>

#### 3.1.2.2 Blocks as Clusters

<a id="sec-3-1-3"></a>

### 3.1.3 More on Streams and Events

<a id="sec-3-1-3-1"></a>

#### 3.1.3.1 Stream Priorities

<a id="sec-3-1-3-2"></a>

#### 3.1.3.2 Explicit Synchronization

<a id="sec-3-1-3-3"></a>

#### 3.1.3.3 Implicit Synchronization

<a id="sec-3-1-4"></a>

### 3.1.4 Programmatic Dependent Kernel Launch

<a id="sec-3-1-5"></a>

### 3.1.5 Batched Memory Transfers

<a id="sec-3-1-6"></a>

### 3.1.6 Environment Variables

<a id="sec-3-2"></a>

## 3.2 Advanced Kernel Programming

<a id="sec-3-2-1"></a>

### 3.2.1 Using PTX

<a id="sec-3-2-2"></a>

### 3.2.2 Hardware Implementation

<a id="sec-3-2-2-1"></a>

#### 3.2.2.1 SIMT Execution Model

<a id="sec-3-2-2-1-1"></a>

##### 3.2.2.1.1 Independent Thread Scheduling

<a id="sec-3-2-2-2"></a>

#### 3.2.2.2 Hardware Multithreading

<a id="sec-3-2-2-3"></a>

#### 3.2.2.3 Asynchronous Execution Features

<a id="sec-3-2-2-3-1"></a>

##### 3.2.2.3.1 Async Thread and Async Proxy

<a id="sec-3-2-3"></a>

### 3.2.3 Thread Scopes

<a id="sec-3-2-4"></a>

### 3.2.4 Advanced Synchronization Primitives

<a id="sec-3-2-4-1"></a>

#### 3.2.4.1 Scoped Atomics

<a id="sec-3-2-4-1-1"></a>

##### 3.2.4.1.1 Thread Scope and Memory Ordering

<a id="sec-3-2-4-1-2"></a>

##### 3.2.4.1.2 Performance Considerations

<a id="sec-3-2-4-2"></a>

#### 3.2.4.2 Asynchronous Barriers

<a id="sec-3-2-4-3"></a>

#### 3.2.4.3 Pipelines

<a id="sec-3-2-5"></a>

### 3.2.5 Asynchronous Data Copies

<a id="sec-3-2-6"></a>

### 3.2.6 Configuring L1/Shared Memory Balance

<a id="sec-3-3"></a>

## 3.3 The CUDA Driver API

<a id="sec-3-3-1"></a>

### 3.3.1 Context

<a id="sec-3-3-2"></a>

### 3.3.2 Module

<a id="sec-3-3-3"></a>

### 3.3.3 Kernel Execution

<a id="sec-3-3-4"></a>

### 3.3.4 Interoperability between Runtime and Driver APIs

<a id="sec-3-4"></a>

## 3.4 Programming Systems with Multiple GPUs

<a id="sec-3-4-1"></a>

### 3.4.1 Multi-Device Context and Execution Management

<a id="sec-3-4-1-1"></a>

#### 3.4.1.1 Device Enumeration

<a id="sec-3-4-1-2"></a>

#### 3.4.1.2 Device Selection

<a id="sec-3-4-1-3"></a>

#### 3.4.1.3 Multi-Device Stream, Event, and Memory Copy Behavior

<a id="sec-3-4-2"></a>

### 3.4.2 Multi-Device Peer-to-Peer Transfers and Memory Access

<a id="sec-3-4-2-1"></a>

#### 3.4.2.1 Peer-to-Peer Memory Transfers

<a id="sec-3-4-2-2"></a>

#### 3.4.2.2 Peer-to-Peer Memory Access

<a id="sec-3-4-2-3"></a>

#### 3.4.2.3 Peer-to-Peer Memory Consistency

<a id="sec-3-4-2-4"></a>

#### 3.4.2.4 Multi-Device Managed Memory

<a id="sec-3-4-2-5"></a>

#### 3.4.2.5 Host IOMMU Hardware, PCI Access Control Services, and VMs

<a id="sec-3-4-2-6"></a>

#### 3.4.2.6 Compute Fabric Transport

<a id="sec-3-5"></a>

## 3.5 A Tour of CUDA Features

<a id="sec-3-5-1"></a>

### 3.5.1 Improving Kernel Performance

<a id="sec-3-5-1-1"></a>

#### 3.5.1.1 Asynchronous Barriers

<a id="sec-3-5-1-2"></a>

#### 3.5.1.2 Asynchronous Data Copies and the Tensor Memory Accelerator (TMA)

<a id="sec-3-5-1-3"></a>

#### 3.5.1.3 Pipelines

<a id="sec-3-5-1-4"></a>

#### 3.5.1.4 Work Stealing with Cluster Launch Control

<a id="sec-3-5-2"></a>

### 3.5.2 Improving Latencies

<a id="sec-3-5-2-1"></a>

#### 3.5.2.1 Green Contexts

<a id="sec-3-5-2-2"></a>

#### 3.5.2.2 Locality Domains

<a id="sec-3-5-2-3"></a>

#### 3.5.2.3 Stream-Ordered Memory Allocation

<a id="sec-3-5-2-4"></a>

#### 3.5.2.4 CUDA Graphs

<a id="sec-3-5-2-5"></a>

#### 3.5.2.5 Programmatic Dependent Launch

<a id="sec-3-5-2-6"></a>

#### 3.5.2.6 Lazy Loading

<a id="sec-3-5-3"></a>

### 3.5.3 Functionality Features

<a id="sec-3-5-3-1"></a>

#### 3.5.3.1 Extended GPU Memory

<a id="sec-3-5-3-2"></a>

#### 3.5.3.2 Dynamic Parallelism

<a id="sec-3-5-4"></a>

### 3.5.4 CUDA Interoperability

<a id="sec-3-5-4-1"></a>

#### 3.5.4.1 CUDA Interoperability with other APIs

<a id="sec-3-5-4-2"></a>

#### 3.5.4.2 Interprocess Communication

<a id="sec-3-5-5"></a>

### 3.5.5 Fine-Grained Control

<a id="sec-3-5-5-1"></a>

#### 3.5.5.1 Virtual Memory Management

<a id="sec-3-5-5-2"></a>

#### 3.5.5.2 Compute Fabric Transport

<a id="sec-3-5-5-3"></a>

#### 3.5.5.3 Driver Entry Point Access

<a id="sec-3-5-5-4"></a>

#### 3.5.5.4 Error Log Management
