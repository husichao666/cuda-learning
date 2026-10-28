# 第 4 章 CUDA 功能

<a id="sec-4-1"></a>

## 4.1 Unified Memory

<a id="sec-4-1-1"></a>

### 4.1.1 Unified Memory on Devices with Full CUDA Unified Memory Support

<a id="sec-4-1-1-1"></a>

#### 4.1.1.1 Unified Memory: In-Depth Examples

<a id="sec-4-1-1-1-1"></a>

##### 4.1.1.1.1 File-backed Unified Memory

<a id="sec-4-1-1-1-2"></a>

##### 4.1.1.1.2 Inter-Process Communication (IPC) with Unified Memory

<a id="sec-4-1-1-2"></a>

#### 4.1.1.2 Performance Tuning

<a id="sec-4-1-1-2-1"></a>

##### 4.1.1.2.1 Memory Paging and Page Sizes

<a id="sec-4-1-1-2-1-1"></a>

###### 4.1.1.2.1.1 Choosing the Right Page Size

<a id="sec-4-1-1-2-1-2"></a>

###### 4.1.1.2.1.2 CPU and GPU Page Tables: Hardware Coherency vs. Software Coherency

<a id="sec-4-1-1-2-1-3"></a>

###### 4.1.1.2.1.3 Mixing Hardware and Software Coherency

<a id="sec-4-1-1-2-2"></a>

##### 4.1.1.2.2 Direct Unified Memory Access from the Host

<a id="sec-4-1-1-2-3"></a>

##### 4.1.1.2.3 Host Native Atomics

<a id="sec-4-1-1-2-4"></a>

##### 4.1.1.2.4 Atomic Accesses and Synchronization Primitives

<a id="sec-4-1-1-2-5"></a>

##### 4.1.1.2.5 Memcpy()/Memset() Behavior With Unified Memory

<a id="sec-4-1-1-2-6"></a>

##### 4.1.1.2.6 Overview of Memory Allocators for Unified Memory

<a id="sec-4-1-1-2-7"></a>

##### 4.1.1.2.7 Locality Domains and Unified Memory

<a id="sec-4-1-1-2-8"></a>

##### 4.1.1.2.8 Access Counter Migration

<a id="sec-4-1-1-2-9"></a>

##### 4.1.1.2.9 Avoid Frequent Writes to GPU-Resident Memory from the CPU

<a id="sec-4-1-1-2-10"></a>

##### 4.1.1.2.10 Exploiting Asynchronous Access to System Memory

<a id="sec-4-1-2"></a>

### 4.1.2 Unified Memory on Devices with only CUDA Managed Memory Support

<a id="sec-4-1-3"></a>

### 4.1.3 Unified Memory on Windows, WSL, and Tegra

<a id="sec-4-1-3-1"></a>

#### 4.1.3.1 Multi-GPU

<a id="sec-4-1-3-2"></a>

#### 4.1.3.2 Coherency and Concurrency

<a id="sec-4-1-3-3"></a>

#### 4.1.3.3 Stream Associated Unified Memory

<a id="sec-4-1-3-3-1"></a>

##### 4.1.3.3.1 Stream Callbacks

<a id="sec-4-1-3-3-2"></a>

##### 4.1.3.3.2 Managed memory associated to streams allows for finer-grained control

<a id="sec-4-1-3-3-3"></a>

##### 4.1.3.3.3 A more elaborate example on multithreaded host programs

<a id="sec-4-1-3-3-4"></a>

##### 4.1.3.3.4 Data Movement of Stream Associated Unified Memory

<a id="sec-4-1-4"></a>

### 4.1.4 Performance Hints

<a id="sec-4-1-4-1"></a>

#### 4.1.4.1 Data Prefetching

<a id="sec-4-1-4-2"></a>

#### 4.1.4.2 Data Usage Hints

<a id="sec-4-1-4-3"></a>

#### 4.1.4.3 Memory Discarding

<a id="sec-4-1-4-4"></a>

#### 4.1.4.4 Querying Data Usage Attributes on Managed Memory

<a id="sec-4-1-4-5"></a>

#### 4.1.4.5 GPU Memory Oversubscription

<a id="sec-4-2"></a>

## 4.2 CUDA Graphs

<a id="sec-4-2-1"></a>

### 4.2.1 Graph Structure

<a id="sec-4-2-1-1"></a>

#### 4.2.1.1 Node Types

<a id="sec-4-2-1-2"></a>

#### 4.2.1.2 Edge Data

<a id="sec-4-2-2"></a>

### 4.2.2 Building and Running Graphs

<a id="sec-4-2-2-1"></a>

#### 4.2.2.1 Graph Creation

<a id="sec-4-2-2-1-1"></a>

##### 4.2.2.1.1 Graph APIs

<a id="sec-4-2-2-1-2"></a>

##### 4.2.2.1.2 Stream Capture

<a id="sec-4-2-2-1-2-1"></a>

###### 4.2.2.1.2.1 Cross-stream Dependencies and Events

<a id="sec-4-2-2-1-2-2"></a>

###### 4.2.2.1.2.2 Prohibited and Unhandled Operations

<a id="sec-4-2-2-1-2-3"></a>

###### 4.2.2.1.2.3 Invalidation

<a id="sec-4-2-2-1-2-4"></a>

###### 4.2.2.1.2.4 Capture Introspection

<a id="sec-4-2-2-1-3"></a>

##### 4.2.2.1.3 Putting It All Together

<a id="sec-4-2-2-2"></a>

#### 4.2.2.2 Graph Instantiation

<a id="sec-4-2-2-3"></a>

#### 4.2.2.3 Graph Execution

<a id="sec-4-2-3"></a>

### 4.2.3 Updating Instantiated Graphs

<a id="sec-4-2-3-1"></a>

#### 4.2.3.1 Whole Graph Update

<a id="sec-4-2-3-2"></a>

#### 4.2.3.2 Individual Node Update

<a id="sec-4-2-3-3"></a>

#### 4.2.3.3 Individual Node Enable

<a id="sec-4-2-3-4"></a>

#### 4.2.3.4 Graph Update Limitations

<a id="sec-4-2-4"></a>

### 4.2.4 Conditional Graph Nodes

<a id="sec-4-2-4-1"></a>

#### 4.2.4.1 Conditional Handles

<a id="sec-4-2-4-2"></a>

#### 4.2.4.2 Conditional Node Body Graph Requirements

<a id="sec-4-2-4-3"></a>

#### 4.2.4.3 Conditional IF Nodes

<a id="sec-4-2-4-4"></a>

#### 4.2.4.4 Conditional WHILE Nodes

<a id="sec-4-2-4-5"></a>

#### 4.2.4.5 Conditional SWITCH Nodes

<a id="sec-4-2-5"></a>

### 4.2.5 Graph Memory Nodes

<a id="sec-4-2-5-1"></a>

#### 4.2.5.1 Introduction

<a id="sec-4-2-5-2"></a>

#### 4.2.5.2 API Fundamentals

<a id="sec-4-2-5-2-1"></a>

##### 4.2.5.2.1 Graph Node APIs

<a id="sec-4-2-5-2-2"></a>

##### 4.2.5.2.2 Stream Capture

<a id="sec-4-2-5-2-3"></a>

##### 4.2.5.2.3 Accessing and Freeing Graph Memory Outside of the Allocating Graph

<a id="sec-4-2-5-2-4"></a>

##### 4.2.5.2.4 cudaGraphInstantiateFlagAutoFreeOnLaunch

<a id="sec-4-2-5-2-5"></a>

##### 4.2.5.2.5 Memory Nodes in Child Graphs

<a id="sec-4-2-5-3"></a>

#### 4.2.5.3 Optimized Memory Reuse

<a id="sec-4-2-5-3-1"></a>

##### 4.2.5.3.1 Address Reuse within a Graph

<a id="sec-4-2-5-3-2"></a>

##### 4.2.5.3.2 Physical Memory Management and Sharing

<a id="sec-4-2-5-4"></a>

#### 4.2.5.4 Performance Considerations

<a id="sec-4-2-5-4-1"></a>

##### 4.2.5.4.1 First Launch / cudaGraphUpload

<a id="sec-4-2-5-5"></a>

#### 4.2.5.5 Physical Memory Footprint

<a id="sec-4-2-5-6"></a>

#### 4.2.5.6 Peer Access

<a id="sec-4-2-5-6-1"></a>

##### 4.2.5.6.1 Peer Access with Graph Node APIs

<a id="sec-4-2-5-6-2"></a>

##### 4.2.5.6.2 Peer Access with Stream Capture

<a id="sec-4-2-6"></a>

### 4.2.6 Device Graph Launch

<a id="sec-4-2-6-1"></a>

#### 4.2.6.1 Device Graph Creation

<a id="sec-4-2-6-1-1"></a>

##### 4.2.6.1.1 Device Graph Requirements

<a id="sec-4-2-6-1-2"></a>

##### 4.2.6.1.2 Device Graph Upload

<a id="sec-4-2-6-1-3"></a>

##### 4.2.6.1.3 Device Graph Update

<a id="sec-4-2-6-2"></a>

#### 4.2.6.2 Device Launch

<a id="sec-4-2-6-2-1"></a>

##### 4.2.6.2.1 Fire and Forget Launch

<a id="sec-4-2-6-2-1-1"></a>

###### 4.2.6.2.1.1 Graph Execution Environments

<a id="sec-4-2-6-2-2"></a>

##### 4.2.6.2.2 Tail Launch

<a id="sec-4-2-6-2-2-1"></a>

###### 4.2.6.2.2.1 Tail Self-launch

<a id="sec-4-2-6-2-3"></a>

##### 4.2.6.2.3 Sibling Launch

<a id="sec-4-2-7"></a>

### 4.2.7 Using Graph APIs

<a id="sec-4-2-8"></a>

### 4.2.8 CUDA User Objects

<a id="sec-4-3"></a>

## 4.3 Stream-Ordered Memory Allocator

<a id="sec-4-3-1"></a>

### 4.3.1 Introduction

<a id="sec-4-3-2"></a>

### 4.3.2 Memory Management

<a id="sec-4-3-2-1"></a>

#### 4.3.2.1 Allocating Memory

<a id="sec-4-3-2-2"></a>

#### 4.3.2.2 Freeing Memory

<a id="sec-4-3-3"></a>

### 4.3.3 Memory Pools

<a id="sec-4-3-3-1"></a>

#### 4.3.3.1 Default/Implicit Pools

<a id="sec-4-3-3-2"></a>

#### 4.3.3.2 Explicit Pools

<a id="sec-4-3-3-3"></a>

#### 4.3.3.3 Device Accessibility for Multi-GPU Support

<a id="sec-4-3-3-4"></a>

#### 4.3.3.4 Enabling Memory Pools for IPC

<a id="sec-4-3-3-4-1"></a>

##### 4.3.3.4.1 Creating and Sharing IPC Memory Pools

<a id="sec-4-3-3-4-2"></a>

##### 4.3.3.4.2 Set Access in the Importing Process

<a id="sec-4-3-3-4-3"></a>

##### 4.3.3.4.3 Creating and Sharing Allocations from an Exported Pool

<a id="sec-4-3-3-4-4"></a>

##### 4.3.3.4.4 IPC Export Pool Limitations

<a id="sec-4-3-3-4-5"></a>

##### 4.3.3.4.5 IPC Import Pool Limitations

<a id="sec-4-3-4"></a>

### 4.3.4 Best Practices and Tuning

<a id="sec-4-3-4-1"></a>

#### 4.3.4.1 Query for Support

<a id="sec-4-3-4-2"></a>

#### 4.3.4.2 Physical Page Caching Behavior

<a id="sec-4-3-4-3"></a>

#### 4.3.4.3 Resource Usage Statistics

<a id="sec-4-3-4-4"></a>

#### 4.3.4.4 Memory Reuse Policies

<a id="sec-4-3-4-4-1"></a>

##### 4.3.4.4.1 cudaMemPoolReuseFollowEventDependencies

<a id="sec-4-3-4-4-2"></a>

##### 4.3.4.4.2 cudaMemPoolReuseAllowOpportunistic

<a id="sec-4-3-4-4-3"></a>

##### 4.3.4.4.3 cudaMemPoolReuseAllowInternalDependencies

<a id="sec-4-3-4-4-4"></a>

##### 4.3.4.4.4 Disabling Reuse Policies

<a id="sec-4-3-4-5"></a>

#### 4.3.4.5 Synchronization API Actions

<a id="sec-4-3-5"></a>

### 4.3.5 Addendums

<a id="sec-4-3-5-1"></a>

#### 4.3.5.1 cudaMemcpyAsync Current Context/Device Sensitivity

<a id="sec-4-3-5-2"></a>

#### 4.3.5.2 cudaPointerGetAttributes Query

<a id="sec-4-3-5-3"></a>

#### 4.3.5.3 cudaGraphAddMemsetNode

<a id="sec-4-3-5-4"></a>

#### 4.3.5.4 Pointer Attributes

<a id="sec-4-3-5-5"></a>

#### 4.3.5.5 CPU Virtual Memory

<a id="sec-4-4"></a>

## 4.4 Cooperative Groups

<a id="sec-4-4-1"></a>

### 4.4.1 Introduction

<a id="sec-4-4-2"></a>

### 4.4.2 Cooperative Group Handle & Member Functions

<a id="sec-4-4-3"></a>

### 4.4.3 Default Behavior / Groupless Execution

<a id="sec-4-4-3-1"></a>

#### 4.4.3.1 Create Implicit Group Handles As Early As Possible

<a id="sec-4-4-3-2"></a>

#### 4.4.3.2 Only Pass Group Handles by Reference

<a id="sec-4-4-4"></a>

### 4.4.4 Creating Cooperative Groups

<a id="sec-4-4-4-1"></a>

#### 4.4.4.1 Avoiding Group Creation Hazards

<a id="sec-4-4-5"></a>

### 4.4.5 Synchronization

<a id="sec-4-4-5-1"></a>

#### 4.4.5.1 Sync

<a id="sec-4-4-5-2"></a>

#### 4.4.5.2 Barriers

<a id="sec-4-4-6"></a>

### 4.4.6 Collective Operations

<a id="sec-4-4-6-1"></a>

#### 4.4.6.1 Reduce

<a id="sec-4-4-6-2"></a>

#### 4.4.6.2 Scans

<a id="sec-4-4-6-3"></a>

#### 4.4.6.3 Invoke One

<a id="sec-4-4-7"></a>

### 4.4.7 Asynchronous Data Movement

<a id="sec-4-4-7-1"></a>

#### 4.4.7.1 Memcpy Async Alignment Requirements

<a id="sec-4-4-8"></a>

### 4.4.8 Large Scale Groups

<a id="sec-4-4-8-1"></a>

#### 4.4.8.1 When to use cudaLaunchCooperativeKernel

<a id="sec-4-5"></a>

## 4.5 Programmatic Dependent Launch and Synchronization

<a id="sec-4-5-1"></a>

### 4.5.1 Background

<a id="sec-4-5-2"></a>

### 4.5.2 API Description

<a id="sec-4-5-3"></a>

### 4.5.3 Use in CUDA Graphs

<a id="sec-4-6"></a>

## 4.6 Green Contexts

<a id="sec-4-6-1"></a>

### 4.6.1 Motivation / When to Use

<a id="sec-4-6-2"></a>

### 4.6.2 Green Contexts: Ease of use

<a id="sec-4-6-3"></a>

### 4.6.3 Green Contexts: Device Resource and Resource Descriptor

<a id="sec-4-6-4"></a>

### 4.6.4 Green Context Creation Example

<a id="sec-4-6-4-1"></a>

#### 4.6.4.1 Step 1: Get available GPU resources

<a id="sec-4-6-4-2"></a>

#### 4.6.4.2 Step 2: Partition SM resources

<a id="sec-4-6-4-3"></a>

#### 4.6.4.3 Step 2 (continued): Add workqueue resources

<a id="sec-4-6-4-4"></a>

#### 4.6.4.4 Step 3: Create a Resource Descriptor

<a id="sec-4-6-4-5"></a>

#### 4.6.4.5 Step 4: Create a Green Context

<a id="sec-4-6-5"></a>

### 4.6.5 Green Contexts - Launching work

<a id="sec-4-6-6"></a>

### 4.6.6 Additional Execution Contexts APIs

<a id="sec-4-6-7"></a>

### 4.6.7 Green Contexts Example

<a id="sec-4-7"></a>

## 4.7 Locality Domains

<a id="sec-4-7-1"></a>

### 4.7.1 Discovering Locality Domains

<a id="sec-4-7-2"></a>

### 4.7.2 Localizing Compute Resources

<a id="sec-4-7-3"></a>

### 4.7.3 Allocating Localized Device Memory

<a id="sec-4-7-3-1"></a>

#### 4.7.3.1 Creating a Localized Memory Pool

<a id="sec-4-7-3-2"></a>

#### 4.7.3.2 Creating a Localized VMM Allocation

<a id="sec-4-7-4"></a>

### 4.7.4 Querying Memory Placement

<a id="sec-4-7-5"></a>

### 4.7.5 Programming Guidance

<a id="sec-4-7-5-1"></a>

#### 4.7.5.1 Localized Kernels With Fork/Join Pattern

<a id="sec-4-7-5-2"></a>

#### 4.7.5.2 CUDA Graphs

<a id="sec-4-7-5-2-1"></a>

##### 4.7.5.2.1 Stream Capture

<a id="sec-4-7-5-2-2"></a>

##### 4.7.5.2.2 Graph Construction

<a id="sec-4-7-5-3"></a>

#### 4.7.5.3 Compatibility Notes

<a id="sec-4-8"></a>

## 4.8 Lazy Loading

<a id="sec-4-8-1"></a>

### 4.8.1 Introduction

<a id="sec-4-8-2"></a>

### 4.8.2 Change History

<a id="sec-4-8-3"></a>

### 4.8.3 Requirements for Lazy Loading

<a id="sec-4-8-3-1"></a>

#### 4.8.3.1 CUDA Runtime Version Requirement

<a id="sec-4-8-3-2"></a>

#### 4.8.3.2 CUDA Driver Version Requirement

<a id="sec-4-8-3-3"></a>

#### 4.8.3.3 Compiler Requirements

<a id="sec-4-8-3-4"></a>

#### 4.8.3.4 Kernel Requirements

<a id="sec-4-8-4"></a>

### 4.8.4 Usage

<a id="sec-4-8-4-1"></a>

#### 4.8.4.1 Enabling & Disabling

<a id="sec-4-8-4-2"></a>

#### 4.8.4.2 Checking if Lazy Loading is Enabled at Runtime

<a id="sec-4-8-4-3"></a>

#### 4.8.4.3 Forcing a Module to Load Eagerly at Runtime

<a id="sec-4-8-5"></a>

### 4.8.5 Potential Hazards

<a id="sec-4-8-5-1"></a>

#### 4.8.5.1 Impact on Concurrent Kernel Execution

<a id="sec-4-8-5-2"></a>

#### 4.8.5.2 Large Memory Allocations

<a id="sec-4-8-5-3"></a>

#### 4.8.5.3 Impact on Performance Measurements

<a id="sec-4-9"></a>

## 4.9 Error Log Management

<a id="sec-4-9-1"></a>

### 4.9.1 Background

<a id="sec-4-9-2"></a>

### 4.9.2 Activation

<a id="sec-4-9-3"></a>

### 4.9.3 Output

<a id="sec-4-9-4"></a>

### 4.9.4 API description

<a id="sec-4-9-4-1"></a>

#### 4.9.4.1 Callback registration

<a id="sec-4-9-4-2"></a>

#### 4.9.4.2 Iterators and log dumps

<a id="sec-4-9-4-3"></a>

#### 4.9.4.3 Dump behavior

<a id="sec-4-9-5"></a>

### 4.9.5 Limitations and known issues

<a id="sec-4-10"></a>

## 4.10 Asynchronous Barriers

<a id="sec-4-10-1"></a>

### 4.10.1 Initialization

<a id="sec-4-10-2"></a>

### 4.10.2 A Barrier’s Phase: Arrival, Countdown, Completion, and Reset

<a id="sec-4-10-2-1"></a>

#### 4.10.2.1 Warp Entanglement

<a id="sec-4-10-3"></a>

### 4.10.3 Explicit Phase Tracking

<a id="sec-4-10-4"></a>

### 4.10.4 Early Exit

<a id="sec-4-10-5"></a>

### 4.10.5 Completion Function

<a id="sec-4-10-6"></a>

### 4.10.6 Tracking Asynchronous Memory Operations

<a id="sec-4-10-7"></a>

### 4.10.7 Producer-Consumer Pattern Using Barriers

<a id="sec-4-11"></a>

## 4.11 Pipelines

<a id="sec-4-11-1"></a>

### 4.11.1 Initialization

<a id="sec-4-11-2"></a>

### 4.11.2 Submitting Work

<a id="sec-4-11-3"></a>

### 4.11.3 Consuming Work

<a id="sec-4-11-4"></a>

### 4.11.4 Warp Entanglement

<a id="sec-4-11-5"></a>

### 4.11.5 Early Exit

<a id="sec-4-11-6"></a>

### 4.11.6 Tracking Asynchronous Memory Operations

<a id="sec-4-11-7"></a>

### 4.11.7 Producer-Consumer Pattern using Pipelines

<a id="sec-4-12"></a>

## 4.12 Asynchronous Data Copies

<a id="sec-4-12-1"></a>

### 4.12.1 Using LDGSTS

<a id="sec-4-12-1-1"></a>

#### 4.12.1.1 Batching Loads in Conditional Code

<a id="sec-4-12-1-2"></a>

#### 4.12.1.2 Prefetching Data

<a id="sec-4-12-1-3"></a>

#### 4.12.1.3 Producer-Consumer Pattern Through Warp Specialization

<a id="sec-4-12-2"></a>

### 4.12.2 Using the Tensor Memory Accelerator (TMA)

<a id="sec-4-12-2-1"></a>

#### 4.12.2.1 Using TMA to transfer one-dimensional arrays

<a id="sec-4-12-2-1-1"></a>

##### 4.12.2.1.1 Prefetching Data

<a id="sec-4-12-2-2"></a>

#### 4.12.2.2 Using TMA to transfer multi-dimensional arrays

<a id="sec-4-12-2-2-1"></a>

##### 4.12.2.2.1 Encoding a Tensor Map on Device

<a id="sec-4-12-2-2-2"></a>

##### 4.12.2.2.2 Device-side Encoding and Modification of a Tensor Map

<a id="sec-4-12-2-2-3"></a>

##### 4.12.2.2.3 Usage of a Modified Tensor Map

<a id="sec-4-12-2-2-4"></a>

##### 4.12.2.2.4 Creating a Template Tensor Map Value Using the Driver API

<a id="sec-4-12-2-2-5"></a>

##### 4.12.2.2.5 Shared-Memory Bank Swizzling

<a id="sec-4-12-3"></a>

### 4.12.3 Using STAS

<a id="sec-4-13"></a>

## 4.13 Work Stealing with Cluster Launch Control

<a id="sec-4-13-1"></a>

### 4.13.1 API Details

<a id="sec-4-13-1-1"></a>

#### 4.13.1.1 Thread Block Cancellation

<a id="sec-4-13-1-2"></a>

#### 4.13.1.2 Constraints on Thread Block Cancellation

<a id="sec-4-13-2"></a>

### 4.13.2 Example: Vector-Scalar Multiplication

<a id="sec-4-13-2-1"></a>

#### 4.13.2.1 Use-case: Thread Blocks

<a id="sec-4-13-2-2"></a>

#### 4.13.2.2 Use-case: Thread Block Clusters

<a id="sec-4-14"></a>

## 4.14 L2 Cache Control

<a id="sec-4-14-1"></a>

### 4.14.1 L2 Cache Set-Aside for Persisting Accesses

<a id="sec-4-14-2"></a>

### 4.14.2 L2 Policy for Persisting Accesses

<a id="sec-4-14-3"></a>

### 4.14.3 L2 Access Properties

<a id="sec-4-14-4"></a>

### 4.14.4 L2 Persistence Example

<a id="sec-4-14-5"></a>

### 4.14.5 Reset L2 Access to Normal

<a id="sec-4-14-6"></a>

### 4.14.6 Manage Utilization of L2 set-aside cache

<a id="sec-4-14-7"></a>

### 4.14.7 Query L2 cache Properties

<a id="sec-4-14-8"></a>

### 4.14.8 Control L2 Cache Set-Aside Size for Persisting Memory Access

<a id="sec-4-15"></a>

## 4.15 Memory Synchronization Domains

<a id="sec-4-15-1"></a>

### 4.15.1 Memory Fence Interference

<a id="sec-4-15-2"></a>

### 4.15.2 Isolating Traffic with Domains

<a id="sec-4-15-3"></a>

### 4.15.3 Using Domains in CUDA

<a id="sec-4-16"></a>

## 4.16 Interprocess Communication

<a id="sec-4-16-1"></a>

### 4.16.1 IPC using the Legacy Interprocess Communication API

<a id="sec-4-16-2"></a>

### 4.16.2 IPC using the Virtual Memory Management API

<a id="sec-4-17"></a>

## 4.17 Virtual Memory Management

<a id="sec-4-17-1"></a>

### 4.17.1 Preliminaries

<a id="sec-4-17-1-1"></a>

#### 4.17.1.1 Definitions

<a id="sec-4-17-1-2"></a>

#### 4.17.1.2 Query for Support

<a id="sec-4-17-2"></a>

### 4.17.2 API Overview

<a id="sec-4-17-3"></a>

### 4.17.3 Unicast Memory Sharing

<a id="sec-4-17-3-1"></a>

#### 4.17.3.1 Allocate and Export

<a id="sec-4-17-3-2"></a>

#### 4.17.3.2 Share and Import

<a id="sec-4-17-3-3"></a>

#### 4.17.3.3 Reserve and Map

<a id="sec-4-17-3-4"></a>

#### 4.17.3.4 Access Rights

<a id="sec-4-17-3-5"></a>

#### 4.17.3.5 Releasing the Memory

<a id="sec-4-17-4"></a>

### 4.17.4 Multicast Memory Sharing

<a id="sec-4-17-4-1"></a>

#### 4.17.4.1 Allocating Multicast Objects

<a id="sec-4-17-4-2"></a>

#### 4.17.4.2 Add Devices to Multicast Objects

<a id="sec-4-17-4-3"></a>

#### 4.17.4.3 Bind Memory to Multicast Objects

<a id="sec-4-17-4-4"></a>

#### 4.17.4.4 Use Multicast Mappings

<a id="sec-4-17-5"></a>

### 4.17.5 Advanced Configuration

<a id="sec-4-17-5-1"></a>

#### 4.17.5.1 Memory Type

<a id="sec-4-17-5-2"></a>

#### 4.17.5.2 Compressible Memory

<a id="sec-4-17-5-3"></a>

#### 4.17.5.3 Virtual Aliasing Support

<a id="sec-4-17-5-4"></a>

#### 4.17.5.4 OS-Specific Handle Details for IPC

<a id="sec-4-18"></a>

## 4.18 Compute Fabric Transport

<a id="sec-4-18-1"></a>

### 4.18.1 Prerequisites and Scope

<a id="sec-4-18-2"></a>

### 4.18.2 Preliminaries

<a id="sec-4-18-2-1"></a>

#### 4.18.2.1 Definitions

<a id="sec-4-18-2-2"></a>

#### 4.18.2.2 Query for Support

<a id="sec-4-18-3"></a>

### 4.18.3 API Overview

<a id="sec-4-18-4"></a>

### 4.18.4 Verify clique membership

<a id="sec-4-18-5"></a>

### 4.18.5 Creating an Endpoint

<a id="sec-4-18-6"></a>

### 4.18.6 Logical Endpoint Ids

<a id="sec-4-18-6-1"></a>

#### 4.18.6.1 Limits and Alignment

<a id="sec-4-18-7"></a>

### 4.18.7 Sharing an Endpoint

<a id="sec-4-18-7-1"></a>

#### 4.18.7.1 Confirming Readiness

<a id="sec-4-18-8"></a>

### 4.18.8 Binding Memory

<a id="sec-4-18-9"></a>

### 4.18.9 Fabric Operations

<a id="sec-4-18-9-1"></a>

#### 4.18.9.1 Per-Thread and Warp-Collective Operations

<a id="sec-4-18-10"></a>

### 4.18.10 Completion Status and Error Reporting

<a id="sec-4-18-10-1"></a>

#### 4.18.10.1 Configuring Barrier Arrivals and Transaction Counts

<a id="sec-4-18-11"></a>

### 4.18.11 Cleaning Up

<a id="sec-4-19"></a>

## 4.19 Extended GPU Memory

<a id="sec-4-19-1"></a>

### 4.19.1 Preliminaries

<a id="sec-4-19-1-1"></a>

#### 4.19.1.1 EGM Platforms: System topology

<a id="sec-4-19-1-2"></a>

#### 4.19.1.2 Socket Identifiers: What are they? How to access them?

<a id="sec-4-19-1-3"></a>

#### 4.19.1.3 Allocators and EGM support

<a id="sec-4-19-1-4"></a>

#### 4.19.1.4 Memory management extensions to current APIs

<a id="sec-4-19-2"></a>

### 4.19.2 Using the EGM Interface

<a id="sec-4-19-2-1"></a>

#### 4.19.2.1 Single-Node, Single-GPU

<a id="sec-4-19-2-2"></a>

#### 4.19.2.2 Single-Node, Multi-GPU

<a id="sec-4-19-2-2-1"></a>

##### 4.19.2.2.1 Using VMM APIs

<a id="sec-4-19-2-2-2"></a>

##### 4.19.2.2.2 Using CUDA Memory Pool

<a id="sec-4-19-2-3"></a>

#### 4.19.2.3 Multi-Node, Multi-GPU

<a id="sec-4-20"></a>

## 4.20 CUDA Dynamic Parallelism

<a id="sec-4-20-1"></a>

### 4.20.1 Introduction

<a id="sec-4-20-1-1"></a>

#### 4.20.1.1 Overview

<a id="sec-4-20-2"></a>

### 4.20.2 Execution Environment

<a id="sec-4-20-2-1"></a>

#### 4.20.2.1 Parent and Child Grids

<a id="sec-4-20-2-2"></a>

#### 4.20.2.2 Scope of CUDA Primitives

<a id="sec-4-20-2-3"></a>

#### 4.20.2.3 Streams and Events

<a id="sec-4-20-2-4"></a>

#### 4.20.2.4 Ordering and Concurrency

<a id="sec-4-20-3"></a>

### 4.20.3 Memory Coherence and Consistency

<a id="sec-4-20-3-1"></a>

#### 4.20.3.1 Global Memory

<a id="sec-4-20-3-2"></a>

#### 4.20.3.2 Mapped Memory

<a id="sec-4-20-3-3"></a>

#### 4.20.3.3 Shared and Local Memory

<a id="sec-4-20-3-4"></a>

#### 4.20.3.4 Local Memory

<a id="sec-4-20-3-4-1"></a>

##### 4.20.3.4.1 Texture Memory

<a id="sec-4-20-4"></a>

### 4.20.4 Programming Interface

<a id="sec-4-20-4-1"></a>

#### 4.20.4.1 Basics

<a id="sec-4-20-4-2"></a>

#### 4.20.4.2 C++ Language Interface for CDP

<a id="sec-4-20-4-2-1"></a>

##### 4.20.4.2.1 Device-Side Kernel Launch

<a id="sec-4-20-4-2-1-1"></a>

###### 4.20.4.2.1.1 Launches are Asynchronous

<a id="sec-4-20-4-2-1-2"></a>

###### 4.20.4.2.1.2 Launch Environment Configuration

<a id="sec-4-20-4-2-2"></a>

##### 4.20.4.2.2 Events

<a id="sec-4-20-4-2-3"></a>

##### 4.20.4.2.3 Synchronization

<a id="sec-4-20-4-2-4"></a>

##### 4.20.4.2.4 Device Management

<a id="sec-4-20-5"></a>

### 4.20.5 Programming Guidelines

<a id="sec-4-20-5-1"></a>

#### 4.20.5.1 Performance

<a id="sec-4-20-5-1-1"></a>

##### 4.20.5.1.1 Dynamic-parallelism-enabled Kernel Overhead

<a id="sec-4-20-5-2"></a>

#### 4.20.5.2 Implementation Restrictions and Limitations

<a id="sec-4-20-5-2-1"></a>

##### 4.20.5.2.1 Runtime

<a id="sec-4-20-5-2-1-1"></a>

###### 4.20.5.2.1.1 Memory Footprint

<a id="sec-4-20-5-2-1-2"></a>

###### 4.20.5.2.1.2 Pending Kernel Launches

<a id="sec-4-20-5-3"></a>

#### 4.20.5.3 Compatibility and Interoperability

<a id="sec-4-20-6"></a>

### 4.20.6 Device-side Launch from PTX

<a id="sec-4-20-6-1"></a>

#### 4.20.6.1 Kernel Launch APIs

<a id="sec-4-20-6-1-1"></a>

##### 4.20.6.1.1 cudaLaunchDevice

<a id="sec-4-20-6-1-2"></a>

##### 4.20.6.1.2 cudaGetParameterBuffer

<a id="sec-4-20-6-2"></a>

#### 4.20.6.2 Parameter Buffer Layout

<a id="sec-4-21"></a>

## 4.21 CUDA Interoperability with APIs

<a id="sec-4-21-1"></a>

### 4.21.1 Graphics Interoperability

<a id="sec-4-21-1-1"></a>

#### 4.21.1.1 OpenGL Interoperability

<a id="sec-4-21-1-2"></a>

#### 4.21.1.2 Direct3D Interoperability

<a id="sec-4-21-1-3"></a>

#### 4.21.1.3 Interoperability in a Scalable Link Interface (SLI) configuration

<a id="sec-4-21-2"></a>

### 4.21.2 External resource interoperability

<a id="sec-4-21-2-1"></a>

#### 4.21.2.1 Vulkan interoperability

<a id="sec-4-21-2-1-1"></a>

##### 4.21.2.1.1 Setting up a Vulkan device

<a id="sec-4-21-2-1-2"></a>

##### 4.21.2.1.2 Initializing CUDA with matching device UUIDs

<a id="sec-4-21-2-1-3"></a>

##### 4.21.2.1.3 Exporting Vulkan memory objects

<a id="sec-4-21-2-1-4"></a>

##### 4.21.2.1.4 Exporting Vulkan synchronization objects

<a id="sec-4-21-2-1-5"></a>

##### 4.21.2.1.5 Importing memory objects

<a id="sec-4-21-2-1-6"></a>

##### 4.21.2.1.6 Mapping buffers onto imported memory objects

<a id="sec-4-21-2-1-7"></a>

##### 4.21.2.1.7 Mapping mipmapped arrays onto imported memory objects

<a id="sec-4-21-2-1-8"></a>

##### 4.21.2.1.8 Importing Synchronization Objects

<a id="sec-4-21-2-1-9"></a>

##### 4.21.2.1.9 Signaling/Waiting on Imported Synchronization Objects

<a id="sec-4-21-2-1-10"></a>

##### 4.21.2.1.10 OpenGL Interoperability

<a id="sec-4-21-2-2"></a>

#### 4.21.2.2 Direct3D Interoperability

<a id="sec-4-21-2-2-1"></a>

##### 4.21.2.2.1 Matching Device LUIDs

<a id="sec-4-21-2-2-2"></a>

##### 4.21.2.2.2 Importing Memory Objects

<a id="sec-4-21-2-2-3"></a>

##### 4.21.2.2.3 Mapping Buffers onto Imported Memory Objects

<a id="sec-4-21-2-2-4"></a>

##### 4.21.2.2.4 Mapping Mipmapped Arrays onto Imported Memory Objects

<a id="sec-4-21-2-2-5"></a>

##### 4.21.2.2.5 Importing Synchronization Objects

<a id="sec-4-21-2-2-6"></a>

##### 4.21.2.2.6 Signaling/Waiting on Imported Synchronization Objects

<a id="sec-4-21-2-3"></a>

#### 4.21.2.3 NVIDIA Software Communication Interface Interoperability (NVSCI)

<a id="sec-4-21-2-3-1"></a>

##### 4.21.2.3.1 Importing Memory Objects

<a id="sec-4-21-2-3-2"></a>

##### 4.21.2.3.2 Mapping Buffers onto Imported Memory Objects

<a id="sec-4-21-2-3-3"></a>

##### 4.21.2.3.3 Mapping Mipmapped Arrays onto Imported Memory Objects

<a id="sec-4-21-2-3-4"></a>

##### 4.21.2.3.4 Importing Synchronization Objects

<a id="sec-4-21-2-3-5"></a>

##### 4.21.2.3.5 Signaling/Waiting on Imported Synchronization Objects

<a id="sec-4-22"></a>

## 4.22 Driver Entry Point Access

<a id="sec-4-22-1"></a>

### 4.22.1 Introduction

<a id="sec-4-22-2"></a>

### 4.22.2 Driver Function Typedefs

<a id="sec-4-22-3"></a>

### 4.22.3 Driver Function Retrieval

<a id="sec-4-22-3-1"></a>

#### 4.22.3.1 Using the Driver API

<a id="sec-4-22-3-2"></a>

#### 4.22.3.2 Using the Runtime API

<a id="sec-4-22-3-3"></a>

#### 4.22.3.3 Retrieve Per-thread Default Stream Versions

<a id="sec-4-22-3-4"></a>

#### 4.22.3.4 Access New CUDA features

<a id="sec-4-22-4"></a>

### 4.22.4 Guidelines for cuGetProcAddress

<a id="sec-4-22-4-1"></a>

#### 4.22.4.1 Guidelines for Runtime API Usage

<a id="sec-4-22-5"></a>

### 4.22.5 Determining cuGetProcAddress Failure Reasons
