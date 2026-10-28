# 第 2 章 使用 CUDA 编程 GPU

<a id="sec-2-1"></a>

## 2.1 Intro to CUDA C++

<a id="sec-2-1-1"></a>

### 2.1.1 Compilation with NVCC

<a id="sec-2-1-2"></a>

### 2.1.2 Kernels

<a id="sec-2-1-2-1"></a>

#### 2.1.2.1 Specifying Kernels

<a id="sec-2-1-2-2"></a>

#### 2.1.2.2 Launching Kernels

<a id="sec-2-1-2-2-1"></a>

##### 2.1.2.2.1 Triple Chevron Notation

<a id="sec-2-1-2-3"></a>

#### 2.1.2.3 Thread and Grid Index Intrinsics

<a id="sec-2-1-2-3-1"></a>

##### 2.1.2.3.1 Bounds Checking

<a id="sec-2-1-3"></a>

### 2.1.3 Memory in GPU Computing

<a id="sec-2-1-3-1"></a>

#### 2.1.3.1 Unified Memory

<a id="sec-2-1-3-2"></a>

#### 2.1.3.2 Explicit Memory Management

<a id="sec-2-1-3-3"></a>

#### 2.1.3.3 Memory Management and Application Performance

<a id="sec-2-1-4"></a>

### 2.1.4 Synchronizing CPU and GPU

<a id="sec-2-1-5"></a>

### 2.1.5 Putting it All Together

<a id="sec-2-1-6"></a>

### 2.1.6 Runtime Initialization

<a id="sec-2-1-7"></a>

### 2.1.7 Error Checking in CUDA

<a id="sec-2-1-7-1"></a>

#### 2.1.7.1 Error State

<a id="sec-2-1-7-2"></a>

#### 2.1.7.2 Asynchronous Errors

<a id="sec-2-1-7-3"></a>

#### 2.1.7.3 CUDA_LOG_FILE

<a id="sec-2-1-8"></a>

### 2.1.8 Device and Host Functions

<a id="sec-2-1-9"></a>

### 2.1.9 Variable Specifiers

<a id="sec-2-1-9-1"></a>

#### 2.1.9.1 Detecting Device Compilation

<a id="sec-2-1-10"></a>

### 2.1.10 Thread Block Clusters

<a id="sec-2-1-10-1"></a>

#### 2.1.10.1 Launching with Clusters in Triple Chevron Notation

<a id="sec-2-2"></a>

## 2.2 Intro to CUDA Python

<a id="sec-2-2-1"></a>

### 2.2.1 CUDA Python Ecosystem

<a id="sec-2-2-1-1"></a>

#### 2.2.1.1 Using CUDA Libraries in Python

<a id="sec-2-2-1-2"></a>

#### 2.2.1.2 Scope of This Chapter

<a id="sec-2-2-1-3"></a>

#### 2.2.1.3 Getting Setup

<a id="sec-2-2-1-4"></a>

#### 2.2.1.4 Running CUDA Python Applications

<a id="sec-2-2-2"></a>

### 2.2.2 SIMT Kernels in Python

<a id="sec-2-2-2-1"></a>

#### 2.2.2.1 Specifying Kernels

<a id="sec-2-2-2-2"></a>

#### 2.2.2.2 Launching Kernels

<a id="sec-2-2-2-2-1"></a>

##### 2.2.2.2.1 Kernel Launch

<a id="sec-2-2-2-2-2"></a>

##### 2.2.2.2.2 Multi-Dimension Grids and Thread Blocks

<a id="sec-2-2-2-3"></a>

#### 2.2.2.3 Thread and Grid Index Intrinsics

<a id="sec-2-2-3"></a>

### 2.2.3 Memory in GPU computing

<a id="sec-2-2-3-1"></a>

#### 2.2.3.1 Instantiating arrays on the GPU

<a id="sec-2-2-3-2"></a>

#### 2.2.3.2 Copying Arrays between the Host and GPU memory

<a id="sec-2-2-3-3"></a>

#### 2.2.3.3 The ndarray object type

<a id="sec-2-2-4"></a>

### 2.2.4 Synchronizing the CPU and the GPU

<a id="sec-2-2-5"></a>

### 2.2.5 Putting it All Together

<a id="sec-2-2-6"></a>

### 2.2.6 Error Checking in CUDA Python

<a id="sec-2-3"></a>

## 2.3 Writing SIMT Kernels

<a id="sec-2-3-1"></a>

### 2.3.1 Basics of SIMT

<a id="sec-2-3-2"></a>

### 2.3.2 Thread Hierarchy

<a id="sec-2-3-2-1"></a>

#### 2.3.2.1 Thread Block Synchronization

<a id="sec-2-3-3"></a>

### 2.3.3 GPU Device Memory Spaces

<a id="sec-2-3-3-1"></a>

#### 2.3.3.1 Global Memory

<a id="sec-2-3-3-2"></a>

#### 2.3.3.2 Shared Memory

<a id="sec-2-3-3-2-1"></a>

##### 2.3.3.2.1 Static Allocation of Shared Memory

<a id="sec-2-3-3-2-2"></a>

##### 2.3.3.2.2 Dynamic Allocation of Shared Memory

<a id="sec-2-3-3-3"></a>

#### 2.3.3.3 Registers

<a id="sec-2-3-3-4"></a>

#### 2.3.3.4 Local Memory

<a id="sec-2-3-3-5"></a>

#### 2.3.3.5 Constant Memory

<a id="sec-2-3-3-6"></a>

#### 2.3.3.6 Caches

<a id="sec-2-3-3-7"></a>

#### 2.3.3.7 Texture and Surface Memory

<a id="sec-2-3-3-8"></a>

#### 2.3.3.8 Distributed Shared Memory

<a id="sec-2-3-4"></a>

### 2.3.4 Memory Performance

<a id="sec-2-3-4-1"></a>

#### 2.3.4.1 Coalesced Global Memory Access

<a id="sec-2-3-4-1-1"></a>

##### 2.3.4.1.1 Matrix Transpose Example Using Global Memory

<a id="sec-2-3-4-2"></a>

#### 2.3.4.2 Shared Memory Access Patterns

<a id="sec-2-3-4-2-1"></a>

##### 2.3.4.2.1 Matrix Transpose Example Using Shared Memory

<a id="sec-2-3-4-2-2"></a>

##### 2.3.4.2.2 Shared Memory Bank Conflicts

<a id="sec-2-3-5"></a>

### 2.3.5 Atomics

<a id="sec-2-3-5-1"></a>

#### 2.3.5.1 C++ std::atomic-like Atomics

<a id="sec-2-3-5-2"></a>

#### 2.3.5.2 Memory Atomics in Python

<a id="sec-2-3-6"></a>

### 2.3.6 Cooperative Groups

<a id="sec-2-3-7"></a>

### 2.3.7 Kernel Launch and Occupancy

<a id="sec-2-4"></a>

## 2.4 Writing Tile Kernels

<a id="sec-2-4-1"></a>

### 2.4.1 Kernel and Function Declarations

<a id="sec-2-4-2"></a>

### 2.4.2 Launching Kernels

<a id="sec-2-4-2-1"></a>

#### 2.4.2.1 Grid-Sizing Pattern

<a id="sec-2-4-3"></a>

### 2.4.3 Querying Block Position

<a id="sec-2-4-4"></a>

### 2.4.4 Creating Tiles

<a id="sec-2-4-5"></a>

### 2.4.5 Compile-Time Constants

<a id="sec-2-4-5-1"></a>

#### 2.4.5.1 Python Constant[T]

<a id="sec-2-4-5-2"></a>

#### 2.4.5.2 C++ integral_constant and _ic Literals

<a id="sec-2-4-6"></a>

### 2.4.6 Loading and Storing Tiles

<a id="sec-2-4-6-1"></a>

#### 2.4.6.1 Tile-Space Loads and Stores

<a id="sec-2-4-6-1-1"></a>

##### 2.4.6.1.1 Partition View Loads and Stores

<a id="sec-2-4-6-1-2"></a>

##### 2.4.6.1.2 Python One-Call Load and Store

<a id="sec-2-4-6-1-3"></a>

##### 2.4.6.1.3 Tile-Space Boundary Handling

<a id="sec-2-4-6-2"></a>

#### 2.4.6.2 Gather and Scatter

<a id="sec-2-4-6-2-1"></a>

##### 2.4.6.2.1 Gather and Scatter Boundary Handling

<a id="sec-2-4-7"></a>

### 2.4.7 Control Flow

<a id="sec-2-4-7-1"></a>

#### 2.4.7.1 Loops

<a id="sec-2-4-7-2"></a>

#### 2.4.7.2 Conditionals

<a id="sec-2-4-8"></a>

### 2.4.8 Element-wise Arithmetic and Broadcasting

<a id="sec-2-4-8-1"></a>

#### 2.4.8.1 Broadcasting

<a id="sec-2-4-8-2"></a>

#### 2.4.8.2 Arithmetic Operators

<a id="sec-2-4-9"></a>

### 2.4.9 Tile Primitives

<a id="sec-2-4-9-1"></a>

#### 2.4.9.1 Matrix Multiply

<a id="sec-2-4-9-2"></a>

#### 2.4.9.2 Reductions and Scans

<a id="sec-2-4-9-3"></a>

#### 2.4.9.3 Transpose and Permutation

<a id="sec-2-4-9-4"></a>

#### 2.4.9.4 Element-wise Selection

<a id="sec-2-4-9-5"></a>

#### 2.4.9.5 Mathematical Functions

<a id="sec-2-4-10"></a>

### 2.4.10 Atomic Memory Operations

<a id="sec-2-4-10-1"></a>

#### 2.4.10.1 Cross-block Contention

<a id="sec-2-4-10-2"></a>

#### 2.4.10.2 Intra-block Contention

<a id="sec-2-4-10-3"></a>

#### 2.4.10.3 Supported Atomic Operations

<a id="sec-2-4-11"></a>

### 2.4.11 Optimization Hints

<a id="sec-2-4-11-1"></a>

#### 2.4.11.1 C++ – the cutile::hint Attribute

<a id="sec-2-4-11-2"></a>

#### 2.4.11.2 Python – Decorator Arguments and Call-Site Keywords

<a id="sec-2-4-11-3"></a>

#### 2.4.11.3 Hint Kinds

<a id="sec-2-4-11-3-1"></a>

##### 2.4.11.3.1 CTAs per cluster

<a id="sec-2-4-11-3-2"></a>

##### 2.4.11.3.2 Occupancy

<a id="sec-2-4-11-3-3"></a>

##### 2.4.11.3.3 Memory access latency

<a id="sec-2-4-11-3-4"></a>

##### 2.4.11.3.4 Allow TMA

<a id="sec-2-4-12"></a>

### 2.4.12 C++ Performance Tips

<a id="sec-2-4-12-1"></a>

#### 2.4.12.1 Use __restrict__ Pointers for Arrays in Memory

<a id="sec-2-4-12-2"></a>

#### 2.4.12.2 Mark Array Pointers as 16-Byte Aligned

<a id="sec-2-4-12-3"></a>

#### 2.4.12.3 Prefer ct::partition_view for Memory Access

<a id="sec-2-4-12-4"></a>

#### 2.4.12.4 Use ct::irange for Bounded Loops

<a id="sec-2-5"></a>

## 2.5 Asynchronous Execution

<a id="sec-2-5-1"></a>

### 2.5.1 What is Asynchronous Concurrent Execution?

<a id="sec-2-5-2"></a>

### 2.5.2 CUDA Streams

<a id="sec-2-5-2-1"></a>

#### 2.5.2.1 Creating and Destroying CUDA Streams

<a id="sec-2-5-2-2"></a>

#### 2.5.2.2 Launching Kernels in CUDA Streams

<a id="sec-2-5-2-3"></a>

#### 2.5.2.3 Launching Memory Transfers in CUDA Streams

<a id="sec-2-5-2-4"></a>

#### 2.5.2.4 Stream Synchronization

<a id="sec-2-5-3"></a>

### 2.5.3 CUDA Events

<a id="sec-2-5-3-1"></a>

#### 2.5.3.1 Creating and Destroying CUDA Events

<a id="sec-2-5-3-2"></a>

#### 2.5.3.2 Inserting Events into CUDA Streams

<a id="sec-2-5-3-3"></a>

#### 2.5.3.3 Timing Operations in CUDA Streams

<a id="sec-2-5-3-4"></a>

#### 2.5.3.4 Checking the Status of CUDA Events

<a id="sec-2-5-4"></a>

### 2.5.4 Callback Functions from Streams

<a id="sec-2-5-4-1"></a>

#### 2.5.4.1 Using cudaStreamAddCallback()

<a id="sec-2-5-4-2"></a>

#### 2.5.4.2 Asynchronous Error Handling

<a id="sec-2-5-5"></a>

### 2.5.5 CUDA Stream Ordering

<a id="sec-2-5-6"></a>

### 2.5.6 Blocking and non-blocking streams and the default stream

<a id="sec-2-5-6-1"></a>

#### 2.5.6.1 Legacy Default Stream

<a id="sec-2-5-6-2"></a>

#### 2.5.6.2 Per-thread Default Stream

<a id="sec-2-5-7"></a>

### 2.5.7 Explicit Synchronization

<a id="sec-2-5-8"></a>

### 2.5.8 Implicit Synchronization

<a id="sec-2-5-9"></a>

### 2.5.9 Miscellaneous and Advanced topics

<a id="sec-2-5-9-1"></a>

#### 2.5.9.1 Stream Prioritization

<a id="sec-2-5-9-2"></a>

#### 2.5.9.2 Introduction to CUDA Graphs with Stream Capture

<a id="sec-2-5-10"></a>

### 2.5.10 Summary of Asynchronous Execution

<a id="sec-2-6"></a>

## 2.6 Unified and System Memory

<a id="sec-2-6-1"></a>

### 2.6.1 Unified Virtual Address Space

<a id="sec-2-6-2"></a>

### 2.6.2 Unified Memory

<a id="sec-2-6-2-1"></a>

#### 2.6.2.1 Unified Memory Paradigms

<a id="sec-2-6-2-1-1"></a>

##### 2.6.2.1.1 Unified Memory Paradigm: Code Example

<a id="sec-2-6-2-2"></a>

#### 2.6.2.2 Full Unified Memory Feature Support

<a id="sec-2-6-2-2-1"></a>

##### 2.6.2.2.1 Full Unified Memory with Hardware Coherency

<a id="sec-2-6-2-2-2"></a>

##### 2.6.2.2.2 HMM - Full Unified Memory with Software Coherency

<a id="sec-2-6-2-3"></a>

#### 2.6.2.3 Limited Unified Memory Support

<a id="sec-2-6-2-4"></a>

#### 2.6.2.4 Memory Advise and Prefetch

<a id="sec-2-6-3"></a>

### 2.6.3 Page-Locked Host Memory

<a id="sec-2-6-3-1"></a>

#### 2.6.3.1 Mapped Memory

<a id="sec-2-6-3-1-1"></a>

##### 2.6.3.1.1 cudaMallocHost and cudaHostAlloc

<a id="sec-2-6-3-1-2"></a>

##### 2.6.3.1.2 cudaHostRegister

<a id="sec-2-6-3-1-3"></a>

##### 2.6.3.1.3 Comparing Unified Memory and Mapped Memory

<a id="sec-2-6-4"></a>

### 2.6.4 Summary

<a id="sec-2-7"></a>

## 2.7 NVCC: The NVIDIA CUDA Compiler

<a id="sec-2-7-1"></a>

### 2.7.1 CUDA Source Files and Headers

<a id="sec-2-7-2"></a>

### 2.7.2 NVCC Compilation Workflow

<a id="sec-2-7-3"></a>

### 2.7.3 NVCC Basic Usage

<a id="sec-2-7-3-1"></a>

#### 2.7.3.1 NVCC PTX and Cubin Generation

<a id="sec-2-7-3-2"></a>

#### 2.7.3.2 Host Code Compilation Notes

<a id="sec-2-7-3-3"></a>

#### 2.7.3.3 Separate Compilation of GPU Code

<a id="sec-2-7-4"></a>

### 2.7.4 Common Compiler Options

<a id="sec-2-7-4-1"></a>

#### 2.7.4.1 Language Features

<a id="sec-2-7-4-2"></a>

#### 2.7.4.2 Debugging Options

<a id="sec-2-7-4-3"></a>

#### 2.7.4.3 Optimization Options

<a id="sec-2-7-4-4"></a>

#### 2.7.4.4 Link-Time Optimization (LTO)

<a id="sec-2-7-4-5"></a>

#### 2.7.4.5 Profiling Options

<a id="sec-2-7-4-6"></a>

#### 2.7.4.6 Fatbin Compression

<a id="sec-2-7-4-7"></a>

#### 2.7.4.7 Compiler Performance Controls
