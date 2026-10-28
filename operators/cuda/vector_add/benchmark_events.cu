#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

void check(cudaError_t err)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void vector_add(
    const float* a, const float* b, float* c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// CPU 端启动函数：空输入不访问指针，也不启动 kernel。
void launch_vector_add(
    const float* a, const float* b, float* c, int n)
{
    if (n == 0) {
        return;
    }

    const int threads = 256;
    const int blocks = n / threads + (n % threads != 0);
    vector_add<<<blocks, threads>>>(a, b, c, n);
    check(cudaGetLastError());
}

void benchmark_vector_add(
    const float* d_a, const float* d_b, float* d_c, int n)
{
    if (n == 0) {
        std::printf("Timing skipped: empty input\n");
        return;
    }

    int device;
    cudaDeviceProp prop;
    check(cudaGetDevice(&device));
    check(cudaGetDeviceProperties(&prop, device));

    const int threads = 256;
    const int blocks = n / threads + (n % threads != 0);
    const int repeats = 1000;
    std::printf("Device: %s, n=%d, blocks=%d, threads=%d\n",
                prop.name, n, blocks, threads);

    cudaEvent_t start, stop;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));

    // 预热，排除在测量区间外。
    for (int k = 0; k < 50; ++k) {
        vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    }
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());

    std::vector<float> samples;
    for (int trial = 0; trial < 5; ++trial) {
        check(cudaEventRecord(start));
        for (int k = 0; k < repeats; ++k) {
            vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
        }
        check(cudaGetLastError());
        check(cudaEventRecord(stop));
        check(cudaEventSynchronize(stop));

        float total_ms;
        check(cudaEventElapsedTime(&total_ms, start, stop));
        const float average_us = total_ms * 1000.0f / repeats;
        samples.push_back(average_us);
        std::printf("Trial %d: %.3f us/launch\n", trial + 1, average_us);
    }

    std::sort(samples.begin(), samples.end());
    std::printf("Median of batch averages: %.3f us/launch\n", samples[2]);
    std::printf("Range: %.3f - %.3f us/launch\n",
                samples.front(), samples.back());

    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));
}

bool run_case(int n)
{
    if (n < 0) {
        std::fprintf(stderr, "n must be nonnegative\n");
        return false;
    }
    if (n == 0) {
        launch_vector_add(nullptr, nullptr, nullptr, 0);
        std::printf("n=0: PASS (empty input)\n");
        return true;
    }

    // 1. 准备 CPU 输入及参考结果。
    std::vector<float> h_a(n), h_b(n), h_c(n), h_ref(n);
    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = 2.0f * static_cast<float>(i);
        h_ref[i] = h_a[i] + h_b[i];
    }
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    // 2. 分配显存。
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    check(cudaMalloc(&d_a, bytes));
    check(cudaMalloc(&d_b, bytes));
    check(cudaMalloc(&d_c, bytes));

    // 3. 输入拷到 GPU。
    check(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    // 4. 启动计算，并等待执行完成。
    launch_vector_add(d_a, d_b, d_c, n);
    check(cudaDeviceSynchronize());

    // 5. 取回结果，逐元素验证。
    check(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    bool passed = true;
    for (int i = 0; i < n; ++i) {
        const float tolerance = 1e-6f + 1e-5f * std::fabs(h_ref[i]);
        if (!std::isfinite(h_c[i]) ||
            std::fabs(h_c[i] - h_ref[i]) > tolerance) {
            std::fprintf(stderr,
                         "Mismatch at %d: got %.9g, expected %.9g\n",
                         i, h_c[i], h_ref[i]);
            passed = false;
            break;
        }
    }

    if (passed) {
        benchmark_vector_add(d_a, d_b, d_c, n);
    }

    // 6. 释放显存。
    check(cudaFree(d_a));
    check(cudaFree(d_b));
    check(cudaFree(d_c));

    std::printf("n=%d: %s\n", n, passed ? "PASS" : "FAIL");
    return passed;
}

int main()
{
    return run_case(1025) ? EXIT_SUCCESS : EXIT_FAILURE;
}
