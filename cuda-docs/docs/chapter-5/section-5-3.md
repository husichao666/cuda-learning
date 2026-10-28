---
title: "5.3 C++ Language Support"
description: "CUDA 编程指南 · 第 5 章 技术附录"
---

<a id="sec-5-3"></a>

# 5.3 C++ Language Support

::: info 待翻译
本节已建立目录框架，正文尚未翻译。
:::

<a id="sec-5-3-1"></a>

## 5.3.1 C++11 Language Features

<a id="sec-5-3-2"></a>

## 5.3.2 C++14 Language Features

<a id="sec-5-3-3"></a>

## 5.3.3 C++17 Language Features

<a id="sec-5-3-4"></a>

## 5.3.4 C++20 Language Features

<a id="sec-5-3-5"></a>

## 5.3.5 C++23 Language Features

<a id="sec-5-3-6"></a>

## 5.3.6 CUDA C++ Standard Library

<a id="sec-5-3-7"></a>

## 5.3.7 C Standard Library Functions

<a id="sec-5-3-7-1"></a>

### 5.3.7.1 clock() and clock64()

<a id="sec-5-3-7-2"></a>

### 5.3.7.2 printf()

<a id="sec-5-3-7-3"></a>

### 5.3.7.3 memcpy() and memset()

<a id="sec-5-3-7-4"></a>

### 5.3.7.4 malloc() and free()

<a id="sec-5-3-7-5"></a>

### 5.3.7.5 alloca()

<a id="sec-5-3-8"></a>

## 5.3.8 Lambda Expressions

<a id="sec-5-3-8-1"></a>

### 5.3.8.1 Lambda Expressions and __global__ Function Parameters

<a id="sec-5-3-8-2"></a>

### 5.3.8.2 Extended Lambdas

<a id="sec-5-3-8-3"></a>

### 5.3.8.3 Extended Lambda Type Traits

<a id="sec-5-3-8-4"></a>

### 5.3.8.4 Extended Lambda Restrictions

<a id="sec-5-3-8-5"></a>

### 5.3.8.5 Host-Device Lambda Optimization Notes

<a id="sec-5-3-8-6"></a>

### 5.3.8.6 *this Capture By-Value

<a id="sec-5-3-8-7"></a>

### 5.3.8.7 Argument Dependent Lookup (ADL)

<a id="sec-5-3-9"></a>

## 5.3.9 Polymorphic Function Wrappers

<a id="sec-5-3-10"></a>

## 5.3.10 C/C++ Language Restrictions

<a id="sec-5-3-10-1"></a>

### 5.3.10.1 Unsupported Features

<a id="sec-5-3-10-2"></a>

### 5.3.10.2 Namespace Reservations

<a id="sec-5-3-10-3"></a>

### 5.3.10.3 Pointers and Memory Addresses

<a id="sec-5-3-10-4"></a>

### 5.3.10.4 Variables

<a id="sec-5-3-10-4-1"></a>

#### 5.3.10.4.1 Local Variables

<a id="sec-5-3-10-4-2"></a>

#### 5.3.10.4.2 const-qualified Variables

<a id="sec-5-3-10-4-3"></a>

#### 5.3.10.4.3 volatile-qualified Variables

<a id="sec-5-3-10-4-4"></a>

#### 5.3.10.4.4 static Variables

<a id="sec-5-3-10-4-5"></a>

#### 5.3.10.4.5 extern Variables

<a id="sec-5-3-10-5"></a>

### 5.3.10.5 Functions

<a id="sec-5-3-10-5-1"></a>

#### 5.3.10.5.1 Recursion

<a id="sec-5-3-10-5-2"></a>

#### 5.3.10.5.2 External Linkage

<a id="sec-5-3-10-5-3"></a>

#### 5.3.10.5.3 Formal Parameters

<a id="sec-5-3-10-5-4"></a>

#### 5.3.10.5.4 __global__ Function Parameters

<a id="sec-5-3-10-5-5"></a>

#### 5.3.10.5.5 __global__ Function Arguments Passing

<a id="sec-5-3-10-6"></a>

### 5.3.10.6 Classes

<a id="sec-5-3-10-6-1"></a>

#### 5.3.10.6.1 Class-type Variables

<a id="sec-5-3-10-6-2"></a>

#### 5.3.10.6.2 Data Members

<a id="sec-5-3-10-6-3"></a>

#### 5.3.10.6.3 Function Members

<a id="sec-5-3-10-6-4"></a>

#### 5.3.10.6.4 Implicitly-Declared and Non-Virtual Explicitly-Defaulted functions

<a id="sec-5-3-10-6-5"></a>

#### 5.3.10.6.5 Polymorphic Classes

<a id="sec-5-3-10-6-6"></a>

#### 5.3.10.6.6 Windows-Specific Class Layout

<a id="sec-5-3-10-7"></a>

### 5.3.10.7 Templates

<a id="sec-5-3-10-8"></a>

### 5.3.10.8 Restrictions in Tile Code

<a id="sec-5-3-11"></a>

## 5.3.11 C++11 Restrictions

<a id="sec-5-3-11-1"></a>

### 5.3.11.1 inline Namespaces

<a id="sec-5-3-11-2"></a>

### 5.3.11.2 inline Unnamed Namespaces

<a id="sec-5-3-11-3"></a>

### 5.3.11.3 constexpr Functions

<a id="sec-5-3-11-4"></a>

### 5.3.11.4 constexpr Variables

<a id="sec-5-3-11-5"></a>

### 5.3.11.5 __global__ Variadic Template

<a id="sec-5-3-11-6"></a>

### 5.3.11.6 Defaulted Functions = default

<a id="sec-5-3-11-7"></a>

### 5.3.11.7 [cuda::]std::initializer_list

<a id="sec-5-3-11-8"></a>

### 5.3.11.8 [cuda::]std::move, [cuda::]std::forward

<a id="sec-5-3-12"></a>

## 5.3.12 C++14 Restrictions

<a id="sec-5-3-12-1"></a>

### 5.3.12.1 Functions with Deduced Return Type

<a id="sec-5-3-12-2"></a>

### 5.3.12.2 Variable Templates

<a id="sec-5-3-13"></a>

## 5.3.13 C++17 Restrictions

<a id="sec-5-3-13-1"></a>

### 5.3.13.1 inline Variables

<a id="sec-5-3-13-2"></a>

### 5.3.13.2 Structured Binding

<a id="sec-5-3-14"></a>

## 5.3.14 C++20 Restrictions

<a id="sec-5-3-14-1"></a>

### 5.3.14.1 Three-way Comparison Operator

<a id="sec-5-3-14-2"></a>

### 5.3.14.2 consteval Functions

<a id="sec-5-3-15"></a>

## 5.3.15 C++23 Restrictions

<a id="sec-5-3-15-1"></a>

### 5.3.15.1 Equality Operator (P2468R2)
