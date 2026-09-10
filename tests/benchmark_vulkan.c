#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include "vulkan/vulkan_context.h"
#include "vulkan/vulkan_buffers.h"
#include "vulkan/vulkan_gemv.h"
#include "vulkan/vulkan_wrapper.h"
#include "qwen38/qwen38_gguf.h"

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

void gemv_f32_cpu(float *output, const float *weights, const float *input, 
                  uint32_t M, uint32_t N) {
    for (uint32_t row = 0; row < M; row++) {
        float sum = 0.0f;
        for (uint32_t col = 0; col < N; col++) {
            sum += weights[row * N + col] * input[col];
        }
        output[row] = sum;
    }
}

typedef struct {
    uint32_t M;
    uint32_t N;
    const char *desc;
} TestSize;

static TestSize test_sizes[] = {
    {256, 256, "Small matrix"},
    {640, 2560, "Expert gate/up projection"},
    {2560, 640, "Expert down projection"},
    {2560, 2560, "Hidden dimension projection"},
};

int main() {
    printf("Vulkan GEMV Benchmark\n");
    printf("=====================\n\n");
    
    // Initialize Vulkan
    Q38VulkanContext *ctx = q38_vulkan_get_context();
    if (!ctx) {
        printf("Failed to initialize Vulkan context\n");
        return 1;
    }
    
    printf("GPU: %s\n", ctx->device_properties.deviceName);
    printf("UMA: %s\n\n", ctx->is_uma ? "Yes" : "No");
    
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    printf("%-30s %10s %10s %8s\n", "Matrix Size", "CPU(ms)", "GPU(ms)", "Speedup");
    printf("%-30s %10s %10s %8s\n", "------------------------------", "----------", "----------", "--------");
    
    for (int t = 0; t < num_tests; t++) {
        uint32_t M = test_sizes[t].M;
        uint32_t N = test_sizes[t].N;
        
        uint64_t weight_size = M * N * sizeof(float);
        float *weights = aligned_alloc(64, weight_size);
        float *input = aligned_alloc(64, N * sizeof(float));
        float *output_cpu = aligned_alloc(64, M * sizeof(float));
        float *output_gpu = aligned_alloc(64, M * sizeof(float));
        
        // Initialize with random data
        srand(42);
        for (uint64_t i = 0; i < M * N; i++) {
            weights[i] = (float)(rand() % 100) / 100.0f - 0.5f;
        }
        for (uint32_t i = 0; i < N; i++) {
            input[i] = (float)(rand() % 100) / 100.0f - 0.5f;
        }
        
        // Create tensor structure
        Q38GGUFTensor tensor = {0};
        tensor.name.data = "test_weights";
        tensor.name.length = 12;
        tensor.n_dims = 2;
        tensor.shape[0] = N;
        tensor.shape[1] = M;
        tensor.type = Q38_GGML_F32;
        tensor.data = (const uint8_t *)weights;
        
        // Warm up CPU
        gemv_f32_cpu(output_cpu, weights, input, M, N);
        
        // Benchmark CPU
        int iterations = 50;
        double start = get_time_ms();
        for (int iter = 0; iter < iterations; iter++) {
            gemv_f32_cpu(output_cpu, weights, input, M, N);
        }
        double cpu_time = (get_time_ms() - start) / iterations;
        
        // Warm up GPU
        q38_vulkan_gemv_f32(ctx, NULL, &tensor, input, output_gpu, M, N);
        
        // Benchmark GPU
        start = get_time_ms();
        for (int iter = 0; iter < iterations; iter++) {
            q38_vulkan_gemv_f32(ctx, NULL, &tensor, input, output_gpu, M, N);
        }
        double gpu_time = (get_time_ms() - start) / iterations;
        
        printf("%-30s %10.3f %10.3f %7.2fx\n", 
               test_sizes[t].desc, cpu_time, gpu_time, cpu_time / gpu_time);
        
        free(weights);
        free(input);
        free(output_cpu);
        free(output_gpu);
    }
    
    printf("\nNote: GPU times include weight upload overhead.\n");
    printf("Weight caching (Phase 3) will eliminate this overhead.\n");
    
    return 0;
}
