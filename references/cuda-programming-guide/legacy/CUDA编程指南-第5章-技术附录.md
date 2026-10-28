# 第 5 章 技术附录

<a id="sec-5-1"></a>

## 5.1 Compute Capabilities

<a id="sec-5-1-1"></a>

### 5.1.1 Obtain the GPU Compute Capability

<a id="sec-5-1-2"></a>

### 5.1.2 Feature Availability

<a id="sec-5-1-2-1"></a>

#### 5.1.2.1 Architecture-Specific Features

<a id="sec-5-1-2-2"></a>

#### 5.1.2.2 Family-Specific Features

<a id="sec-5-1-2-3"></a>

#### 5.1.2.3 Feature Set Compiler Targets

<a id="sec-5-1-3"></a>

### 5.1.3 Features and Technical Specifications

<a id="sec-5-2"></a>

## 5.2 CUDA Environment Variables

<a id="sec-5-2-1"></a>

### 5.2.1 Device Enumeration and Properties

<a id="sec-5-2-1-1"></a>

#### 5.2.1.1 CUDA_VISIBLE_DEVICES

<a id="sec-5-2-1-2"></a>

#### 5.2.1.2 CUDA_DEVICE_ORDER

<a id="sec-5-2-1-3"></a>

#### 5.2.1.3 CUDA_MANAGED_FORCE_DEVICE_ALLOC

<a id="sec-5-2-2"></a>

### 5.2.2 JIT Compilation

<a id="sec-5-2-2-1"></a>

#### 5.2.2.1 CUDA_CACHE_DISABLE

<a id="sec-5-2-2-2"></a>

#### 5.2.2.2 CUDA_CACHE_PATH

<a id="sec-5-2-2-3"></a>

#### 5.2.2.3 CUDA_CACHE_MAXSIZE

<a id="sec-5-2-2-4"></a>

#### 5.2.2.4 CUDA_FORCE_PTX_JIT and CUDA_FORCE_JIT

<a id="sec-5-2-2-5"></a>

#### 5.2.2.5 CUDA_DISABLE_PTX_JIT and CUDA_DISABLE_JIT

<a id="sec-5-2-2-6"></a>

#### 5.2.2.6 CUDA_FORCE_PRELOAD_LIBRARIES

<a id="sec-5-2-3"></a>

### 5.2.3 Execution

<a id="sec-5-2-3-1"></a>

#### 5.2.3.1 CUDA_LAUNCH_BLOCKING

<a id="sec-5-2-3-2"></a>

#### 5.2.3.2 CUDA_DEVICE_MAX_CONNECTIONS

<a id="sec-5-2-3-3"></a>

#### 5.2.3.3 CUDA_DEVICE_MAX_COPY_CONNECTIONS

<a id="sec-5-2-3-4"></a>

#### 5.2.3.4 CUDA_SCALE_LAUNCH_QUEUES

<a id="sec-5-2-3-5"></a>

#### 5.2.3.5 CUDA_GRAPHS_USE_NODE_PRIORITY

<a id="sec-5-2-3-6"></a>

#### 5.2.3.6 CUDA_DEVICE_WAITS_ON_EXCEPTION

<a id="sec-5-2-3-7"></a>

#### 5.2.3.7 CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT

<a id="sec-5-2-3-8"></a>

#### 5.2.3.8 CUDA_DISABLE_PERF_BOOST

<a id="sec-5-2-3-9"></a>

#### 5.2.3.9 CUDA_PREFER_SPINLOCKS_ON_HOST

<a id="sec-5-2-3-10"></a>

#### 5.2.3.10 CUDA_AUTO_BOOST [[deprecated]]

<a id="sec-5-2-4"></a>

### 5.2.4 Module Loading

<a id="sec-5-2-4-1"></a>

#### 5.2.4.1 CUDA_MODULE_LOADING

<a id="sec-5-2-4-2"></a>

#### 5.2.4.2 CUDA_MODULE_DATA_LOADING

<a id="sec-5-2-4-3"></a>

#### 5.2.4.3 CUDA_BINARY_LOADER_THREAD_COUNT

<a id="sec-5-2-4-4"></a>

#### 5.2.4.4 CUDA_DEVICE_DEFAULT_STACK_SIZE_MAX

<a id="sec-5-2-5"></a>

### 5.2.5 CUDA Error Log Management

<a id="sec-5-2-5-1"></a>

#### 5.2.5.1 CUDA_LOG_FILE

<a id="sec-5-3"></a>

## 5.3 C++ Language Support

<a id="sec-5-3-1"></a>

### 5.3.1 C++11 Language Features

<a id="sec-5-3-2"></a>

### 5.3.2 C++14 Language Features

<a id="sec-5-3-3"></a>

### 5.3.3 C++17 Language Features

<a id="sec-5-3-4"></a>

### 5.3.4 C++20 Language Features

<a id="sec-5-3-5"></a>

### 5.3.5 C++23 Language Features

<a id="sec-5-3-6"></a>

### 5.3.6 CUDA C++ Standard Library

<a id="sec-5-3-7"></a>

### 5.3.7 C Standard Library Functions

<a id="sec-5-3-7-1"></a>

#### 5.3.7.1 clock() and clock64()

<a id="sec-5-3-7-2"></a>

#### 5.3.7.2 printf()

<a id="sec-5-3-7-3"></a>

#### 5.3.7.3 memcpy() and memset()

<a id="sec-5-3-7-4"></a>

#### 5.3.7.4 malloc() and free()

<a id="sec-5-3-7-5"></a>

#### 5.3.7.5 alloca()

<a id="sec-5-3-8"></a>

### 5.3.8 Lambda Expressions

<a id="sec-5-3-8-1"></a>

#### 5.3.8.1 Lambda Expressions and __global__ Function Parameters

<a id="sec-5-3-8-2"></a>

#### 5.3.8.2 Extended Lambdas

<a id="sec-5-3-8-3"></a>

#### 5.3.8.3 Extended Lambda Type Traits

<a id="sec-5-3-8-4"></a>

#### 5.3.8.4 Extended Lambda Restrictions

<a id="sec-5-3-8-5"></a>

#### 5.3.8.5 Host-Device Lambda Optimization Notes

<a id="sec-5-3-8-6"></a>

#### 5.3.8.6 *this Capture By-Value

<a id="sec-5-3-8-7"></a>

#### 5.3.8.7 Argument Dependent Lookup (ADL)

<a id="sec-5-3-9"></a>

### 5.3.9 Polymorphic Function Wrappers

<a id="sec-5-3-10"></a>

### 5.3.10 C/C++ Language Restrictions

<a id="sec-5-3-10-1"></a>

#### 5.3.10.1 Unsupported Features

<a id="sec-5-3-10-2"></a>

#### 5.3.10.2 Namespace Reservations

<a id="sec-5-3-10-3"></a>

#### 5.3.10.3 Pointers and Memory Addresses

<a id="sec-5-3-10-4"></a>

#### 5.3.10.4 Variables

<a id="sec-5-3-10-4-1"></a>

##### 5.3.10.4.1 Local Variables

<a id="sec-5-3-10-4-2"></a>

##### 5.3.10.4.2 const-qualified Variables

<a id="sec-5-3-10-4-3"></a>

##### 5.3.10.4.3 volatile-qualified Variables

<a id="sec-5-3-10-4-4"></a>

##### 5.3.10.4.4 static Variables

<a id="sec-5-3-10-4-5"></a>

##### 5.3.10.4.5 extern Variables

<a id="sec-5-3-10-5"></a>

#### 5.3.10.5 Functions

<a id="sec-5-3-10-5-1"></a>

##### 5.3.10.5.1 Recursion

<a id="sec-5-3-10-5-2"></a>

##### 5.3.10.5.2 External Linkage

<a id="sec-5-3-10-5-3"></a>

##### 5.3.10.5.3 Formal Parameters

<a id="sec-5-3-10-5-4"></a>

##### 5.3.10.5.4 __global__ Function Parameters

<a id="sec-5-3-10-5-5"></a>

##### 5.3.10.5.5 __global__ Function Arguments Passing

<a id="sec-5-3-10-6"></a>

#### 5.3.10.6 Classes

<a id="sec-5-3-10-6-1"></a>

##### 5.3.10.6.1 Class-type Variables

<a id="sec-5-3-10-6-2"></a>

##### 5.3.10.6.2 Data Members

<a id="sec-5-3-10-6-3"></a>

##### 5.3.10.6.3 Function Members

<a id="sec-5-3-10-6-4"></a>

##### 5.3.10.6.4 Implicitly-Declared and Non-Virtual Explicitly-Defaulted functions

<a id="sec-5-3-10-6-5"></a>

##### 5.3.10.6.5 Polymorphic Classes

<a id="sec-5-3-10-6-6"></a>

##### 5.3.10.6.6 Windows-Specific Class Layout

<a id="sec-5-3-10-7"></a>

#### 5.3.10.7 Templates

<a id="sec-5-3-10-8"></a>

#### 5.3.10.8 Restrictions in Tile Code

<a id="sec-5-3-11"></a>

### 5.3.11 C++11 Restrictions

<a id="sec-5-3-11-1"></a>

#### 5.3.11.1 inline Namespaces

<a id="sec-5-3-11-2"></a>

#### 5.3.11.2 inline Unnamed Namespaces

<a id="sec-5-3-11-3"></a>

#### 5.3.11.3 constexpr Functions

<a id="sec-5-3-11-4"></a>

#### 5.3.11.4 constexpr Variables

<a id="sec-5-3-11-5"></a>

#### 5.3.11.5 __global__ Variadic Template

<a id="sec-5-3-11-6"></a>

#### 5.3.11.6 Defaulted Functions = default

<a id="sec-5-3-11-7"></a>

#### 5.3.11.7 [cuda::]std::initializer_list

<a id="sec-5-3-11-8"></a>

#### 5.3.11.8 [cuda::]std::move, [cuda::]std::forward

<a id="sec-5-3-12"></a>

### 5.3.12 C++14 Restrictions

<a id="sec-5-3-12-1"></a>

#### 5.3.12.1 Functions with Deduced Return Type

<a id="sec-5-3-12-2"></a>

#### 5.3.12.2 Variable Templates

<a id="sec-5-3-13"></a>

### 5.3.13 C++17 Restrictions

<a id="sec-5-3-13-1"></a>

#### 5.3.13.1 inline Variables

<a id="sec-5-3-13-2"></a>

#### 5.3.13.2 Structured Binding

<a id="sec-5-3-14"></a>

### 5.3.14 C++20 Restrictions

<a id="sec-5-3-14-1"></a>

#### 5.3.14.1 Three-way Comparison Operator

<a id="sec-5-3-14-2"></a>

#### 5.3.14.2 consteval Functions

<a id="sec-5-3-15"></a>

### 5.3.15 C++23 Restrictions

<a id="sec-5-3-15-1"></a>

#### 5.3.15.1 Equality Operator (P2468R2)

<a id="sec-5-4"></a>

## 5.4 C/C++ Language Extensions

<a id="sec-5-4-1"></a>

### 5.4.1 Function and Variable Annotations

<a id="sec-5-4-1-1"></a>

#### 5.4.1.1 Execution Space Specifiers

<a id="sec-5-4-1-2"></a>

#### 5.4.1.2 Memory Space Specifiers

<a id="sec-5-4-1-2-1"></a>

##### 5.4.1.2.1 __shared__ Memory

<a id="sec-5-4-1-2-2"></a>

##### 5.4.1.2.2 __managed__ Memory

<a id="sec-5-4-1-2-3"></a>

##### 5.4.1.2.3 __tile__ Variables

<a id="sec-5-4-1-3"></a>

#### 5.4.1.3 Inlining Specifiers

<a id="sec-5-4-1-4"></a>

#### 5.4.1.4 __restrict__ Pointers

<a id="sec-5-4-1-5"></a>

#### 5.4.1.5 __grid_constant__ Parameters

<a id="sec-5-4-1-6"></a>

#### 5.4.1.6 Annotation Summary

<a id="sec-5-4-2"></a>

### 5.4.2 Built-in Types and Variables

<a id="sec-5-4-2-1"></a>

#### 5.4.2.1 Host Compiler Type Extensions

<a id="sec-5-4-2-2"></a>

#### 5.4.2.2 Built-in Variables

<a id="sec-5-4-2-3"></a>

#### 5.4.2.3 Built-in Types

<a id="sec-5-4-3"></a>

### 5.4.3 Kernel Configuration

<a id="sec-5-4-3-1"></a>

#### 5.4.3.1 Thread Block Cluster

<a id="sec-5-4-3-2"></a>

#### 5.4.3.2 Launch Bounds

<a id="sec-5-4-3-3"></a>

#### 5.4.3.3 Maximum Number of Registers per Thread

<a id="sec-5-4-4"></a>

### 5.4.4 Synchronization Primitives

<a id="sec-5-4-4-1"></a>

#### 5.4.4.1 Thread Block Synchronization Functions

<a id="sec-5-4-4-2"></a>

#### 5.4.4.2 Warp Synchronization Function

<a id="sec-5-4-4-3"></a>

#### 5.4.4.3 Memory Fence Functions

<a id="sec-5-4-5"></a>

### 5.4.5 Atomic Functions

<a id="sec-5-4-5-1"></a>

#### 5.4.5.1 Legacy Atomic Functions

<a id="sec-5-4-5-1-1"></a>

##### 5.4.5.1.1 atomicAdd()

<a id="sec-5-4-5-1-2"></a>

##### 5.4.5.1.2 atomicSub()

<a id="sec-5-4-5-1-3"></a>

##### 5.4.5.1.3 atomicInc()

<a id="sec-5-4-5-1-4"></a>

##### 5.4.5.1.4 atomicDec()

<a id="sec-5-4-5-1-5"></a>

##### 5.4.5.1.5 atomicAnd()

<a id="sec-5-4-5-1-6"></a>

##### 5.4.5.1.6 atomicOr()

<a id="sec-5-4-5-1-7"></a>

##### 5.4.5.1.7 atomicXor()

<a id="sec-5-4-5-1-8"></a>

##### 5.4.5.1.8 atomicMin()

<a id="sec-5-4-5-1-9"></a>

##### 5.4.5.1.9 atomicMax()

<a id="sec-5-4-5-1-10"></a>

##### 5.4.5.1.10 atomicExch()

<a id="sec-5-4-5-1-11"></a>

##### 5.4.5.1.11 atomicCAS()

<a id="sec-5-4-5-2"></a>

#### 5.4.5.2 Built-in Atomic Functions

<a id="sec-5-4-5-2-1"></a>

##### 5.4.5.2.1 __nv_atomic_fetch_add(), __nv_atomic_add()

<a id="sec-5-4-5-2-2"></a>

##### 5.4.5.2.2 __nv_atomic_fetch_sub(), __nv_atomic_sub()

<a id="sec-5-4-5-2-3"></a>

##### 5.4.5.2.3 __nv_atomic_fetch_and(), __nv_atomic_and()

<a id="sec-5-4-5-2-4"></a>

##### 5.4.5.2.4 __nv_atomic_fetch_or(), __nv_atomic_or()

<a id="sec-5-4-5-2-5"></a>

##### 5.4.5.2.5 __nv_atomic_fetch_xor(), __nv_atomic_xor()

<a id="sec-5-4-5-2-6"></a>

##### 5.4.5.2.6 __nv_atomic_fetch_min(), __nv_atomic_min()

<a id="sec-5-4-5-2-7"></a>

##### 5.4.5.2.7 __nv_atomic_fetch_max(), __nv_atomic_max()

<a id="sec-5-4-5-2-8"></a>

##### 5.4.5.2.8 __nv_atomic_exchange(), __nv_atomic_exchange_n()

<a id="sec-5-4-5-2-9"></a>

##### 5.4.5.2.9 __nv_atomic_compare_exchange(), __nv_atomic_compare_exchange_n()

<a id="sec-5-4-5-2-10"></a>

##### 5.4.5.2.10 __nv_atomic_load(), __nv_atomic_load_n()

<a id="sec-5-4-5-2-11"></a>

##### 5.4.5.2.11 __nv_atomic_store(), __nv_atomic_store_n()

<a id="sec-5-4-5-2-12"></a>

##### 5.4.5.2.12 __nv_atomic_thread_fence()

<a id="sec-5-4-6"></a>

### 5.4.6 Warp Functions

<a id="sec-5-4-6-1"></a>

#### 5.4.6.1 Warp Active Mask

<a id="sec-5-4-6-2"></a>

#### 5.4.6.2 Warp Vote Functions

<a id="sec-5-4-6-3"></a>

#### 5.4.6.3 Warp Match Functions

<a id="sec-5-4-6-4"></a>

#### 5.4.6.4 Warp Reduce Functions

<a id="sec-5-4-6-5"></a>

#### 5.4.6.5 Warp Shuffle Functions

<a id="sec-5-4-6-6"></a>

#### 5.4.6.6 Warp __sync Intrinsic Constraints

<a id="sec-5-4-7"></a>

### 5.4.7 CUDA-Specific Macros

<a id="sec-5-4-7-1"></a>

#### 5.4.7.1 __CUDA_ARCH__

<a id="sec-5-4-7-2"></a>

#### 5.4.7.2 __CUDA_ARCH_SPECIFIC__ and __CUDA_ARCH_FAMILY_SPECIFIC__

<a id="sec-5-4-7-3"></a>

#### 5.4.7.3 CUDA Feature Testing Macros

<a id="sec-5-4-7-4"></a>

#### 5.4.7.4 __nv_pure__ Attribute

<a id="sec-5-4-8"></a>

### 5.4.8 CUDA-Specific Functions

<a id="sec-5-4-8-1"></a>

#### 5.4.8.1 Address Space Predicate Functions

<a id="sec-5-4-8-2"></a>

#### 5.4.8.2 Address Space Conversion Functions

<a id="sec-5-4-8-3"></a>

#### 5.4.8.3 Low-Level Load and Store Functions

<a id="sec-5-4-8-4"></a>

#### 5.4.8.4 __trap()

<a id="sec-5-4-8-5"></a>

#### 5.4.8.5 __nanosleep()

<a id="sec-5-4-8-6"></a>

#### 5.4.8.6 Dynamic Programming eXtension (DPX) Instructions

<a id="sec-5-4-9"></a>

### 5.4.9 Compiler Optimization Hints

<a id="sec-5-4-9-1"></a>

#### 5.4.9.1 #pragma unroll

<a id="sec-5-4-9-2"></a>

#### 5.4.9.2 __builtin_assume_aligned()

<a id="sec-5-4-9-3"></a>

#### 5.4.9.3 __builtin_assume() and __assume()

<a id="sec-5-4-9-4"></a>

#### 5.4.9.4 __builtin_constant_p()

<a id="sec-5-4-9-5"></a>

#### 5.4.9.5 __builtin_expect()

<a id="sec-5-4-9-6"></a>

#### 5.4.9.6 __builtin_unreachable()

<a id="sec-5-4-9-7"></a>

#### 5.4.9.7 Custom ABI Pragmas

<a id="sec-5-4-9-8"></a>

#### 5.4.9.8 MMA Throughput Pragma

<a id="sec-5-4-10"></a>

### 5.4.10 Debugging and Diagnostics

<a id="sec-5-4-10-1"></a>

#### 5.4.10.1 Assertion

<a id="sec-5-4-10-2"></a>

#### 5.4.10.2 Breakpoint Function

<a id="sec-5-4-10-3"></a>

#### 5.4.10.3 Diagnostic Pragmas

<a id="sec-5-4-11"></a>

### 5.4.11 Warp Matrix Functions

<a id="sec-5-4-11-1"></a>

#### 5.4.11.1 Description

<a id="sec-5-4-11-2"></a>

#### 5.4.11.2 Alternate Floating Point

<a id="sec-5-4-11-3"></a>

#### 5.4.11.3 Double Precision

<a id="sec-5-4-11-4"></a>

#### 5.4.11.4 Sub-byte Operations

<a id="sec-5-4-11-5"></a>

#### 5.4.11.5 Restrictions

<a id="sec-5-4-11-6"></a>

#### 5.4.11.6 Element Types and Matrix Sizes

<a id="sec-5-4-11-7"></a>

#### 5.4.11.7 Example

<a id="sec-5-5"></a>

## 5.5 Floating-Point Computation

<a id="sec-5-5-1"></a>

### 5.5.1 Floating-Point Introduction

<a id="sec-5-5-1-1"></a>

#### 5.5.1.1 Floating-Point Format

<a id="sec-5-5-1-2"></a>

#### 5.5.1.2 Normal and Subnormal Values

<a id="sec-5-5-1-3"></a>

#### 5.5.1.3 Special Values

<a id="sec-5-5-1-4"></a>

#### 5.5.1.4 Associativity

<a id="sec-5-5-1-5"></a>

#### 5.5.1.5 Fused Multiply-Add (FMA)

<a id="sec-5-5-1-6"></a>

#### 5.5.1.6 Dot Product Example

<a id="sec-5-5-1-7"></a>

#### 5.5.1.7 Rounding

<a id="sec-5-5-1-8"></a>

#### 5.5.1.8 Notes on Host/Device Computation Accuracy

<a id="sec-5-5-2"></a>

### 5.5.2 Floating-Point Data Types

<a id="sec-5-5-3"></a>

### 5.5.3 CUDA and IEEE-754 Compliance

<a id="sec-5-5-4"></a>

### 5.5.4 CUDA and C/C++ Compliance

<a id="sec-5-5-5"></a>

### 5.5.5 Floating-Point Functionality Exposure

<a id="sec-5-5-6"></a>

### 5.5.6 Built-In Arithmetic Operators

<a id="sec-5-5-7"></a>

### 5.5.7 CUDA C++ Mathematical Standard Library Functions

<a id="sec-5-5-7-1"></a>

#### 5.5.7.1 Basic Operations

<a id="sec-5-5-7-2"></a>

#### 5.5.7.2 Exponential Functions

<a id="sec-5-5-7-3"></a>

#### 5.5.7.3 Power Functions

<a id="sec-5-5-7-4"></a>

#### 5.5.7.4 Trigonometric Functions

<a id="sec-5-5-7-5"></a>

#### 5.5.7.5 Hyperbolic Functions

<a id="sec-5-5-7-6"></a>

#### 5.5.7.6 Error and Gamma Functions

<a id="sec-5-5-7-7"></a>

#### 5.5.7.7 Nearest Integer Floating-Point Operations

<a id="sec-5-5-7-8"></a>

#### 5.5.7.8 Floating-Point Manipulation Functions

<a id="sec-5-5-7-9"></a>

#### 5.5.7.9 Classification and Comparison

<a id="sec-5-5-8"></a>

### 5.5.8 Non-Standard CUDA Mathematical Functions

<a id="sec-5-5-9"></a>

### 5.5.9 Intrinsic Functions

<a id="sec-5-5-9-1"></a>

#### 5.5.9.1 Basic Intrinsic Functions

<a id="sec-5-5-9-2"></a>

#### 5.5.9.2 Single-Precision-Only Intrinsic Functions

<a id="sec-5-5-9-3"></a>

#### 5.5.9.3 --use_fast_math Effect

<a id="sec-5-5-10"></a>

### 5.5.10 References

<a id="sec-5-6"></a>

## 5.6 Device-Callable APIs and Intrinsics

<a id="sec-5-6-1"></a>

### 5.6.1 Memory Barrier Primitives Interface

<a id="sec-5-6-1-1"></a>

#### 5.6.1.1 Data Types

<a id="sec-5-6-1-2"></a>

#### 5.6.1.2 Memory Barrier Primitives API

<a id="sec-5-6-2"></a>

### 5.6.2 Pipeline Primitives Interface

<a id="sec-5-6-2-1"></a>

#### 5.6.2.1 memcpy_async Primitive

<a id="sec-5-6-2-2"></a>

#### 5.6.2.2 Commit Primitive

<a id="sec-5-6-2-3"></a>

#### 5.6.2.3 Wait Primitive

<a id="sec-5-6-2-4"></a>

#### 5.6.2.4 Arrive On Barrier Primitive

<a id="sec-5-6-3"></a>

### 5.6.3 Cooperative Groups API

<a id="sec-5-6-3-1"></a>

#### 5.6.3.1 cooperative_groups.h

<a id="sec-5-6-3-1-1"></a>

##### 5.6.3.1.1 class thread_block

<a id="sec-5-6-3-1-2"></a>

##### 5.6.3.1.2 class cluster_group

<a id="sec-5-6-3-1-3"></a>

##### 5.6.3.1.3 class grid_group

<a id="sec-5-6-3-1-4"></a>

##### 5.6.3.1.4 class thread_block_tile

<a id="sec-5-6-3-1-5"></a>

##### 5.6.3.1.5 class coalesced_group

<a id="sec-5-6-3-2"></a>

#### 5.6.3.2 cooperative_groups/async.h

<a id="sec-5-6-3-2-1"></a>

##### 5.6.3.2.1 memcpy_async

<a id="sec-5-6-3-2-2"></a>

##### 5.6.3.2.2 wait and wait_prior

<a id="sec-5-6-3-3"></a>

#### 5.6.3.3 cooperative_groups/partition.h

<a id="sec-5-6-3-3-1"></a>

##### 5.6.3.3.1 tiled_partition

<a id="sec-5-6-3-3-2"></a>

##### 5.6.3.3.2 labeled_partition

<a id="sec-5-6-3-3-3"></a>

##### 5.6.3.3.3 binary_partition

<a id="sec-5-6-3-4"></a>

#### 5.6.3.4 cooperative_groups/reduce.h

<a id="sec-5-6-3-4-1"></a>

##### 5.6.3.4.1 Reduce Operators

<a id="sec-5-6-3-4-2"></a>

##### 5.6.3.4.2 reduce

<a id="sec-5-6-3-5"></a>

#### 5.6.3.5 cooperative_groups/scan.h

<a id="sec-5-6-3-5-1"></a>

##### 5.6.3.5.1 inclusive_scan and exclusive_scan

<a id="sec-5-6-3-6"></a>

#### 5.6.3.6 cooperative_groups/sync.h

<a id="sec-5-6-3-6-1"></a>

##### 5.6.3.6.1 barrier_arrive and barrier_wait

<a id="sec-5-6-3-6-2"></a>

##### 5.6.3.6.2 sync

<a id="sec-5-6-3-6-2-1"></a>

###### 5.6.3.6.2.1 Grid Synchronization

<a id="sec-5-6-4"></a>

### 5.6.4 CUDA Device Runtime

<a id="sec-5-6-4-1"></a>

#### 5.6.4.1 Including Device Runtime API in CUDA Code

<a id="sec-5-6-4-2"></a>

#### 5.6.4.2 Memory in the CUDA Device Runtime

<a id="sec-5-6-4-2-1"></a>

##### 5.6.4.2.1 Configuration Options

<a id="sec-5-6-4-2-2"></a>

##### 5.6.4.2.2 Allocation and Lifetime

<a id="sec-5-6-4-2-2-1"></a>

###### 5.6.4.2.2.1 Memory Declarations

<a id="sec-5-6-4-2-2-2"></a>

###### 5.6.4.2.2.2 Device and Constant Memory

<a id="sec-5-6-4-2-2-3"></a>

###### 5.6.4.2.2.3 Textures and Surfaces

<a id="sec-5-6-4-2-2-4"></a>

###### 5.6.4.2.2.4 Shared Memory Variable Declarations

<a id="sec-5-6-4-2-2-5"></a>

###### 5.6.4.2.2.5 Constant Memory

<a id="sec-5-6-4-2-2-6"></a>

###### 5.6.4.2.2.6 Symbol Addresses

<a id="sec-5-6-4-3"></a>

#### 5.6.4.3 SM Id and Warp Id

<a id="sec-5-6-4-4"></a>

#### 5.6.4.4 Launch Setup APIs

<a id="sec-5-6-4-5"></a>

#### 5.6.4.5 Device Management

<a id="sec-5-6-4-6"></a>

#### 5.6.4.6 API Reference

<a id="sec-5-6-4-7"></a>

#### 5.6.4.7 API Errors and Launch Failures

<a id="sec-5-6-4-8"></a>

#### 5.6.4.8 Device Runtime Streams

<a id="sec-5-6-4-8-1"></a>

##### 5.6.4.8.1 The Implicit (NULL) Stream

<a id="sec-5-6-4-8-2"></a>

##### 5.6.4.8.2 The Fire-and-Forget Stream

<a id="sec-5-6-4-8-3"></a>

##### 5.6.4.8.3 The Tail Launch Stream

<a id="sec-5-6-4-9"></a>

#### 5.6.4.9 ECC Errors

<a id="sec-5-7"></a>

## 5.7 CUDA C++ Memory Model

<a id="sec-5-7-1"></a>

### 5.7.1 Thread Scopes

<a id="sec-5-7-1-1"></a>

#### 5.7.1.1 Scope Relationships

<a id="sec-5-7-2"></a>

### 5.7.2 Synchronization primitives

<a id="sec-5-7-3"></a>

### 5.7.3 Atomicity

<a id="sec-5-7-4"></a>

### 5.7.4 Data Races

<a id="sec-5-7-5"></a>

### 5.7.5 Example: Message Passing

<a id="sec-5-8"></a>

## 5.8 CUDA C++ Execution model

<a id="sec-5-8-1"></a>

### 5.8.1 Host threads

<a id="sec-5-8-2"></a>

### 5.8.2 Device threads

<a id="sec-5-8-3"></a>

### 5.8.3 CUDA APIs

<a id="sec-5-8-3-1"></a>

#### 5.8.3.1 Dependencies
