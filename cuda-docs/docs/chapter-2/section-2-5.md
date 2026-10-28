---
title: "2.5 异步执行"
description: "CUDA Programming Guide 13.4.2 · 第二章中文译文"
---

<a id="sec-2-5"></a>

# 2.5 异步执行

> 来源：《CUDA Programming Guide》13.4.2，书内第 104–118 页相关内容。正文为中文，常用 CUDA 与 memory 术语保留英文；代码保留原文，恢复 PDF 断行与缩进。


<a id="sec-2-5-1"></a>

## 2.5.1 什么是异步并发执行

CUDA 允许多类任务并发或重叠：host 计算、device 计算、host 到 device 的 memory transfer、device 到 host 的传输、同一 device memory 内的传输，以及 device 之间的传输。

通过异步接口表达并发时，提交函数或 kernel launch 通常在工作完成前返回，甚至可能在工作开始前返回。应用程序随后可执行其他任务；需要结果时，再用同步机制确认操作已经完成。常见用法是让数据传输与计算重叠，减少或隐藏传输开销。

<a id="figure-20"></a>

![图 20：PDF 原图 20](/images/chapter-02/figure-20.png)

*图 20：使用 CUDA stream 表达异步并发执行。*

通常有三类同步方式：

- Blocking：调用函数并等待操作完成。
- Non-blocking 或 polling：立即返回并报告当前完成状态。
- Callback：操作完成后执行预先注册的函数。

接口支持异步，不代表硬件一定能同时执行所有操作。实际并发能力取决于 CUDA 版本、GPU 资源及 [Compute Capability](/chapter-5/section-5-1#sec-5-1)。

[同步 CPU 与 GPU](/chapter-2/section-2-1#sec-2-1-4) 中的 `cudaDeviceSynchronize()` 是 blocking 调用，等待先前提交的 GPU 工作完成。CUDA 同时提供 blocking、non-blocking 及 host callback 机制。

异步执行的核心抽象是 stream 与 event。相关的 CUDA Graphs 可以预先定义操作图，再以较低开销反复执行；入门见 [通过 stream capture 创建 Graph](/chapter-2/section-2-5#sec-2-5-9-2)，完整介绍见 [CUDA Graphs](/chapter-4/section-4-2#sec-4-2)。

<a id="sec-2-5-2"></a>

## 2.5.2 CUDA stream

Stream 表达有序操作序列，可以看作 work queue。程序按顺序将 memory copy、kernel launch 等加入队列；前面的操作完成后，后面的操作才能继续。

应用程序可同时使用多个 stream。Runtime 根据 GPU 资源状态，从有工作可执行的 stream 中调度任务。可以设置 priority 作为调度提示，但它不保证具体执行顺序。

提交到 stream 的 API 和 kernel launch 相对于 host thread 是异步的。可以等待某个 stream 的全部工作完成，也可同步整个 device。

没有显式指定 stream 的操作，会进入 default stream。它有特殊的同步语义，见 [blocking、non-blocking 与 default stream](/chapter-2/section-2-5#sec-2-5-6)。

<a id="sec-2-5-2-1"></a>

### 2.5.2.1 创建与销毁 CUDA stream

`cudaStreamCreate()` 创建 stream 并初始化 handle，后续用该 handle 指定目标 stream。
```cpp
cudaStream_t stream; // Stream handle
cudaStreamCreate(&stream); // Create a new stream
// stream based operations ...
cudaStreamDestroy(stream); // Destroy the stream
```

如果调用 `cudaStreamDestroy()` 时 stream 仍有工作，相关资源会在其中工作完成后再释放。

<a id="sec-2-5-2-2"></a>

### 2.5.2.2 在 stream 中启动 kernel

在三尖括号配置中加入 stream 参数。下面假设已创建 `cudaStream_t` 类型的 `stream`：
```cpp
kernel<<<grid, block, shared_mem_size, stream>>>(...);
```

launch 异步返回。成功提交后，kernel 在该 stream 中执行，应用程序可继续执行 CPU 工作，或向其他 GPU stream 提交任务。

<a id="sec-2-5-2-3"></a>

### 2.5.2.3 在 stream 中启动 memory transfer

使用 `cudaMemcpyAsync()`。它类似 `cudaMemcpy()`，但额外接收 stream 参数。下面将 host 指针 `src` 指向的 `size` 字节复制到 device 指针 `dst`，并排入 `stream`。
```cpp
// Copy `size` bytes from `src` to `dst` in stream `stream`
cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
```

异步调用允许 host 在传输完成前继续执行；访问结果前必须适当同步。`cudaMemcpy2D()` 等其他 memory transfer API 也有对应异步版本。

::: tip 注意
涉及 CPU memory 的真正异步传输，需要 host buffer 为 pinned/page-locked memory。对普通 pageable host memory 调用 `cudaMemcpyAsync()` 仍可工作，但可能退化为同步行为，无法与其他工作重叠。推荐使用 `cudaMallocHost()` 分配 GPU 数据收发 buffer。
:::

<a id="sec-2-5-2-4"></a>

### 2.5.2.4 Stream 同步

最简单的方法是等待 stream 中全部任务完成。`cudaStreamSynchronize()` 会阻塞 host：
```cpp
// Wait for the stream to be empty of tasks
cudaStreamSynchronize(stream);
// At this point the stream is done
// and we can access the results of stream operations safely
```

若只想快速查询、不阻塞等待，则使用 `cudaStreamQuery()`：
```cpp
// Have a peek at the stream
// returns cudaSuccess if the stream is empty
// returns cudaErrorNotReady if the stream is not empty
cudaError_t status = cudaStreamQuery(stream);
switch (status) {
    case cudaSuccess:
        // The stream is empty
        std::cout << "The stream is empty" << std::endl;
        break;
    case cudaErrorNotReady:
        // The stream is not empty
        std::cout << "The stream is not empty" << std::endl;
        break;
    default:
        // An error occurred - we should handle this
        break;
};
```

<a id="sec-2-5-3"></a>

## 2.5.3 CUDA event

Event 是插入 stream 的进度标记。假设一个 stream 顺序执行两个 kernel，而 CPU 或其他 stream 的某个操作只依赖第一个 kernel。若没有 event，只能通过等待整个 stream 为空，确认两个 kernel 都结束。

在第一个 kernel 后、第二个 kernel 前记录 event，CPU 或其他 stream 就可只等待该标记。这样，依赖操作在第一个 kernel 完成后即可启动，不必等待第二个 kernel 完成。这可以构成操作与 stream 之间的依赖图，也为后面的 CUDA Graphs 提供直观理解。

Event 还可以记录时间，用于测量 kernel 和 memory transfer 的耗时。

<a id="sec-2-5-3-1"></a>

### 2.5.3.1 创建与销毁 event

使用 `cudaEventCreate()`、`cudaEventDestroy()`：
```cpp
cudaEvent_t event;
// Create the event
cudaEventCreate(&event);
// do some work involving the event
// Once the work is done and the event is no longer needed
// we can destroy the event
cudaEventDestroy(event);
```

应用程序负责在 event 不再需要时销毁它。

<a id="sec-2-5-3-2"></a>

### 2.5.3.2 在 stream 中记录 event

使用 `cudaEventRecord()`：
```cpp
cudaEvent_t event;
cudaStream_t stream;
// Create the event
cudaEventCreate(&event);
// Insert the event into the stream
cudaEventRecord(event, stream);
```

<a id="sec-2-5-3-3"></a>

### 2.5.3.3 使用 event 计时

Event 到达 stream 中相应执行位置时记录时间戳。将两个 event 放在 kernel 前后，可以测量其在 stream 中的执行耗时。
```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);
cudaEvent_t start;
cudaEvent_t stop;
// create the events
cudaEventCreate(&start);
cudaEventCreate(&stop);
 // record the start event
cudaEventRecord(start, stream);
// launch the kernel
kernel<<<grid, block, 0, stream>>>(...);
// record the stop event
cudaEventRecord(stop, stream);
// wait for the stream to complete
// both events will have been triggered
cudaStreamSynchronize(stream);
// get the timing
float elapsedTime;
cudaEventElapsedTime(&elapsedTime, start, stop);
std::cout << "Kernel execution time: " << elapsedTime << " ms" << std::endl;
// clean up
cudaEventDestroy(start);
cudaEventDestroy(stop);
cudaStreamDestroy(stream);
```

<a id="sec-2-5-3-4"></a>

### 2.5.3.4 检查 event 状态

与 stream 类似，event 可以 blocking 等待，也可以 non-blocking 查询。

`cudaEventSynchronize()` 等待 event 完成。下面在 `kernel1` 与 `kernel2` 之间记录 event，只需等待第一个 kernel 的依赖方可在 event 完成后继续，不必等 `kernel2`。
```cpp
cudaEvent_t event;
cudaStream_t stream;
// create the stream
cudaStreamCreate(&stream);
// create the event
cudaEventCreate(&event);
// launch a kernel into the stream
kernel<<<grid, block, 0, stream>>>(...);
// Record the event
cudaEventRecord(event, stream);
// launch a kernel into the stream
kernel2<<<grid, block, 0, stream>>>(...);
// Wait for the event to complete
// Kernel 1 will be guaranteed to have completed
// and we can launch the dependent task.
cudaEventSynchronize(event);
dependentCPUtask();
// Wait for the stream to be empty
// Kernel 2 is guaranteed to have completed
cudaStreamSynchronize(stream);
// destroy the event
cudaEventDestroy(event);
// destroy the stream
cudaStreamDestroy(stream);
```

`cudaEventQuery()` 则立即查询状态。下面 `kernel1` 在 `stream1` 产生需要复制回 host 的数据，后面还有 `kernel2`。Host 一边执行其他 CPU 工作，一边偶尔查询 event；确认 `kernel1` 完成后，向 `stream2` 提交 device-to-host copy。这样 CPU 工作、GPU kernel 和数据复制可以重叠。
```cpp
cudaEvent_t event;
cudaStream_t stream1;
cudaStream_t stream2;
size_t size = LARGE_NUMBER;
float* d_data;
float* h_data;
// Create some data
cudaMalloc(&d_data, size);
cudaMallocHost(&h_data, size);
// create the streams
cudaStreamCreate(&stream1); // Processing stream
cudaStreamCreate(&stream2); // Copying stream
bool copyStarted = false;
// create the event
cudaEventCreate(&event);
// launch kernel1 into the stream
kernel1<<<grid, block, 0, stream1>>>(d_data, size);
// enqueue an event following kernel1
cudaEventRecord(event, stream1);
// launch kernel2 into the stream
kernel2<<<grid, block, 0, stream1>>>();
// while the kernels are running do some work on the CPU
// but check if kernel1 has completed because then we will start
// a device to host copy in stream2
while ( not allCPUWorkDone() || not copyStarted ) {
    doNextChunkOfCPUWork();
    // peek to see if kernel 1 has completed
    // if so enqueue a non-blocking copy into stream2
    if ( not copyStarted ) {
        if( cudaEventQuery(event) == cudaSuccess ) {
            cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream2);
            copyStarted = true;
        }
    }
}
// wait for both streams to be done
cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);
// destroy the event
cudaEventDestroy(event);
// destroy the streams and free the data
cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
cudaFree(d_data);
free(h_data);
```

<a id="sec-2-5-4"></a>

## 2.5.4 从 stream 调用 host 函数

CUDA 支持在 stream 执行顺序中安排 host 函数。推荐使用 `cudaLaunchHostFunc()`；旧接口 `cudaStreamAddCallback()` 已计划弃用，应迁移到前者。

`cudaLaunchHostFunc()` 的签名如下：
```cpp
cudaError_t cudaLaunchHostFunc(cudaStream_t stream, void (*func)(void *), void *data);
```

其中 `stream` 指定目标 stream，`func` 指定函数，`data` 指向传给函数的用户数据。Host 函数采用以下形式，并自行解释 `data` 指向的结构。
```cpp
void hostFunction(void *data);
```

::: warning 注意
Host callback 内不能调用 CUDA API。
:::

为配合 Unified Memory，接口提供以下执行保证：

- Host 函数执行期间，其 stream 视为空闲，因此函数可以访问附属于该 stream 的 memory。
- 函数开始执行，相当于同步了紧邻其前、在同一 stream 记录的 event，也会同步此前已 join 的 stream。
- 向 stream 添加 device 工作，不会在前面的 host 函数和 callback 完成前使相应工作进入活跃执行。若其他 stream 的工作通过 event 排在该函数之后，函数仍可按规定访问 global-attached memory。
- 函数完成本身不会无条件使 stream 活跃；后面没有 device 工作时仍保持空闲，连续多个 host 函数之间没有 device 工作时也是如此。因此可以用 stream 末尾的 host 函数发送通知。

<a id="sec-2-5-4-1"></a>

### 2.5.4.1 使用 cudaStreamAddCallback()

::: tip 旧接口
本函数计划弃用、移除。这里只为理解已有代码而介绍，新应用程序应使用 `cudaLaunchHostFunc()`。
:::

函数签名：
```cpp
cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void* userData, unsigned int flags);
```

参数分别为目标 `stream`、`callback`、用户数据指针 `userData`，以及为未来兼容必须设为 0 的 `flags`。

Callback 签名与 `cudaLaunchHostFunc` 略有不同：
```cpp
void callbackFunction(cudaStream_t stream, cudaError_t status, void *userData);
```

它接收来源 stream handle、触发 callback 时的 `status` 和用户数据。`status` 包含此前操作可能设置的 stream 错误状态。

与新接口类似，callback 完成前 stream 不继续推进后续任务，callback 内也不能调用 CUDA 函数。

<a id="sec-2-5-4-2"></a>

### 2.5.4.2 异步错误处理

Stream 中的 kernel、memory transfer 等任何操作都可能发生错误。这些错误可能要到 event 等待、`cudaStreamSynchronize()` 等同步位置才传播给应用程序。

可使用以下函数读取已记录的错误：

| API | 行为 |
| --- | --- |
| `cudaGetLastError()` | 返回并清除错误状态；若两次调用间没有新错误，第二次返回 `cudaSuccess`。 |
| `cudaPeekAtLastError()` | 返回错误状态而不清除。 |

二者返回 `cudaError_t`。用 `cudaGetErrorName()`、`cudaGetErrorString()` 获取可打印的名称和说明。结合 [host thread 的错误状态](/chapter-2/section-2-1#sec-2-1-7-1) 理解这些检查的作用。

<strong>代码清单 1：使用错误状态查询函数。</strong>
```cpp
// Some work occurs in streams.
cudaStreamSynchronize(stream);
// Look at the last error but do not clear it
cudaError_t err = cudaPeekAtLastError();
if (err != cudaSuccess) {
    printf("Error with name: %s\n", cudaGetErrorName(err));
    printf("Error description: %s\n", cudaGetErrorString(err));
}
// Look at the last error and clear it
cudaError_t err2 = cudaGetLastError();
if (err2 != cudaSuccess) {
    printf("Error with name: %s\n", cudaGetErrorName(err2));
    printf("Error description: %s\n", cudaGetErrorString(err2));
}
if (err2 != err) {
    printf("As expected, cudaPeekAtLastError() did not clear the error\n");
}
// Check again
cudaError_t err3 = cudaGetLastError();
if (err3 == cudaSuccess) {
    printf("As expected, cudaGetLastError() cleared the error\n");
}
```

::: tip 调试建议
大量操作排在一个 stream 中时，错误常在同步点才暴露，难以定位具体来源。调试时可设置 `CUDA_LAUNCH_BLOCKING=1`，使 kernel launch 采用阻塞行为，帮助找到出错的 kernel。同步可能很昂贵，启用后程序可能明显变慢。
:::

<a id="sec-2-5-5"></a>

## 2.5.5 CUDA stream 的顺序

理解 stream、event、callback 后，需要明确异步操作的排序语义。某些高级机制会在特定条件下放宽执行重叠限制，例如 [Programmatic Dependent Kernel Launch](/chapter-3/section-3-1#sec-3-1-4)，或 [批量 memory transfer](/chapter-3/section-3-1#sec-3-1-5) 中无重叠的 copy。

通常 CUDA stream 是 in-order：执行顺序与入队顺序相同，后面的操作不能越过前面的操作。Memory copy 等操作由 runtime 跟踪，在后续依赖 kernel 使用数据前，会保证相应传输已完成。

<a id="sec-2-5-6"></a>

## 2.5.6 Blocking、non-blocking 与 default stream

这里 blocking/non-blocking 指的是与 legacy default stream 的同步关系，不是说某个 stream 的每次提交是否阻塞 CPU。

`cudaStreamCreate()` 默认创建 blocking stream。要创建 non-blocking stream，使用 `cudaStreamCreateWithFlags()` 并传入 `cudaStreamNonBlocking`。
```cpp
cudaStream_t stream;
cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
```

Non-blocking stream 同样使用 `cudaStreamDestroy()` 销毁。

<a id="sec-2-5-6-1"></a>

### 2.5.6.1 Legacy default stream

Legacy default stream 也称 NULL stream，或 stream ID 0。未指定 stream 的 kernel launch、相关默认操作会使用它；它由多个 host thread 共享。

它与其他 blocking stream 存在隐式同步关系：提交到 legacy default stream 的操作，要等待此前其他 blocking stream 的工作；之后其他 blocking stream 的操作也受该默认操作的排序影响。
```cpp
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);
kernel1<<<grid, block, 0, stream1>>>(...);
kernel2<<<grid, block>>>(...);
kernel3<<<grid, block, 0, stream2>>>(...);
cudaDeviceSynchronize();
```

上例中，`kernel2` 等待 `kernel1`，`kernel3` 等待 `kernel2`，即使硬件原本可能让它们并发。

下面改为两个 non-blocking stream，default stream 不再与它们隐式同步，三个 kernel 原则上可以并发。此时不能假定它们的顺序，若需要确认结束，应显式同步，例如使用较重的 `cudaDeviceSynchronize()`。
```cpp
cudaStream_t stream1, stream2;
cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);
kernel1<<<grid, block, 0, stream1>>>(...);
kernel2<<<grid, block>>>(...);
kernel3<<<grid, block, 0, stream2>>>(...);
cudaDeviceSynchronize();
```

<a id="sec-2-5-6-2"></a>

### 2.5.6.2 Per-thread default stream

从 CUDA 7 开始，可以让每个 host thread 拥有独立 default stream，代替共享的 legacy stream。使用 `nvcc --default-stream per-thread`，或按相应规则定义 `CUDA_API_PER_THREAD_DEFAULT_STREAM`。

启用后，每个 host thread 的 default stream 不再具有 legacy default stream 那样的跨 stream 隐式同步关系；前面的 legacy 示例将具有类似 non-blocking 示例的并发行为。

<a id="sec-2-5-7"></a>

## 2.5.7 显式同步

| API | 等待范围与作用 |
| --- | --- |
| `cudaDeviceSynchronize()` | 等待当前 device 上各 host thread 先前提交的所有 stream 工作完成。 |
| `cudaStreamSynchronize(stream)` | 只等待指定 stream，其他 stream 可继续执行。 |
| `cudaStreamWaitEvent(stream, event)` | 使该调用后加入目标 stream 的操作等待 event 完成，建立 device 侧依赖。 |
| `cudaStreamQuery(stream)` | 查询指定 stream 先前的工作是否全部完成，不阻塞等待。 |

Event 的定义与用法见 [CUDA event](/chapter-2/section-2-5#sec-2-5-3)。

<a id="sec-2-5-8"></a>

## 2.5.8 隐式同步

若两个不同 blocking stream 的操作之间插入 NULL stream 操作，可能因默认 stream 的同步语义而无法并发。使用 `cudaStreamNonBlocking` 创建的 stream 不受这种隐式同步限制。

为提高并发机会，应先提交相互独立的工作，再提交依赖工作；各种同步尽可能推迟到真正需要结果时。

<a id="sec-2-5-9"></a>

## 2.5.9 其他与高级主题

<a id="sec-2-5-9-1"></a>

### 2.5.9.1 Stream priority

通过 `cudaStreamCreateWithPriority()` 创建带 priority 的 stream。通常数值越小，优先级越高；默认优先级为 0。支持范围可由 `cudaDeviceGetStreamPriorityRange()` 查询，具体函数参数见下例。
```cpp
int minPriority, maxPriority;
// Query the priority range for the device
cudaDeviceGetStreamPriorityRange(&minPriority, &maxPriority);
// Create two streams with different priorities
// cudaStreamDefault indicates the stream should be created with default flags
// in other words they will be blocking streams with respect to the legacy default stream
// One could also use the option `cudaStreamNonBlocking` here to create a non-blocking streams
cudaStream_t stream1, stream2;
cudaStreamCreateWithPriority(&stream1, cudaStreamDefault, minPriority); // Lowest priority
cudaStreamCreateWithPriority(&stream2, cudaStreamDefault, maxPriority); // Highest priority
```

Priority 只是 runtime 的提示，主要影响 kernel 调度，未必影响 memory transfer。它不会抢占已经执行的工作，也不保证特定执行顺序。

<a id="sec-2-5-9-2"></a>

### 2.5.9.2 通过 stream capture 初识 CUDA Graphs

Stream 表达顺序操作，多个 stream 配合 `cudaStreamWaitEvent` 可以形成有向无环图（DAG）。某些应用程序会反复执行相同的序列或 DAG。

CUDA Graphs 允许先定义这张图，再多次执行，从而减少 host 重复调用 API 的延迟和 CPU 开销。本节通过 stream capture 演示，更完整的说明见 [CUDA Graphs](/chapter-4/section-4-2#sec-4-2)。

典型流程如下：

1. Capture：首次建立图时捕获操作，也可以通过 Graph API 手工构建。
2. Instantiate：为图创建执行所需的 runtime 结构，尽量降低后续启动成本。
3. Execute：根据需要反复执行已实例化的图，复用现成结构，减少 CPU 开销。

<strong>代码清单 2：简单线性 Graph 的捕获、实例化与执行。原文来源为 CUDA Developer Technical Blog，A. Gray，2019。</strong>
```cpp
#define N 500000 // tuned such that kernel takes a few microseconds
// A very lightweight kernel
__global__ void shortKernel(float * out_d, float * in_d){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<N) out_d[idx]=1.23*in_d[idx];
}
bool graphCreated=false;
cudaGraph_t graph;
cudaGraphExec_t instance;
// The graph will be executed NSTEP times
for(int istep=0; istep<NSTEP; istep++){
    if(!graphCreated){
        // Capture the graph
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        // Launch NKERNEL kernels
        for(int ikrnl=0; ikrnl<NKERNEL; ikrnl++){
            shortKernel<<<blocks, threads, 0, stream>>>(out_d, in_d);
        }
        // End the capture
        cudaStreamEndCapture(stream, &graph);
        // Instantiate the graph
        cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
        graphCreated=true;
    }
    // Launch the graph
    cudaGraphLaunch(instance, stream);
    // Synchronize the stream
    cudaStreamSynchronize(stream);
}
```

更多图更新、依赖、执行方式见 [CUDA Graphs](/chapter-4/section-4-2#sec-4-2)。

<a id="sec-2-5-10"></a>

## 2.5.10 异步执行小结

- 异步 API 可以表达任务并发与操作重叠，但实际并发程度取决于硬件资源及 Compute Capability。
- 核心抽象是 stream、event 和 callback。
- 同步可以在 event、stream 或 device 层级进行。
- Legacy default stream 与其他 blocking stream 隐式同步，但不与 non-blocking stream 进行这种同步。
- 可通过 per-thread default stream 模式避免 legacy 行为。
- Stream priority 是提示，不是执行顺序保证，也未必影响 memory transfer。
- CUDA Graphs、批量 memory transfer、Programmatic Dependent Kernel Launch 等机制可以降低或重叠 launch、传输的开销。
