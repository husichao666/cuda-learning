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

    // 6. 释放显存。
    check(cudaFree(d_a));
    check(cudaFree(d_b));
    check(cudaFree(d_c));

    std::printf("n=%d: %s\n", n, passed ? "PASS" : "FAIL");
    return passed;
}

int main()
{
    const int lengths[] = {0, 1, 31, 32, 33, 255, 256, 257, 1025, 1000003};
    for (int n : lengths) {
        if (!run_case(n)) {
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
