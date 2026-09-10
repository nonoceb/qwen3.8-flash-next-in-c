#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vulkan/vulkan_context.h"
#include "vulkan/vulkan_buffers.h"
#include "vulkan/vulkan_gemv.h"
#include "vulkan/vulkan_wrapper.h"
#include "qwen38/qwen38_gguf.h"  // For Q38GGUFTensor definition

// Simple CPU reference implementation
void gemv_cpu(float *output, const float *weights, const float *input, uint32_t M, uint32_t N) {
    for (uint32_t row = 0; row < M; row++) {
        float sum = 0.0f;
        for (uint32_t col = 0; col < N; col++) {
            sum += weights[row * N + col] * input[col];
        }
        output[row] = sum;
    }
}

int main() {
    printf("Testing Vulkan GEMV Implementation\n");
    printf("===================================\n\n");
    
    // Initialize Vulkan
    Q38VulkanContext *ctx = q38_vulkan_get_context();
    if (!ctx) {
        printf("Failed to initialize Vulkan context\n");
        return 1;
    }
    
    printf("Vulkan initialized successfully!\n");
    printf("  UMA support: %s\n", ctx->is_uma ? "Yes" : "No");
    printf("\n");
    
    // Test different matrix sizes
    struct {
        uint32_t M;
        uint32_t N;
    } test_sizes[] = {
        {256, 256},
        {512, 512},
        {1024, 1024},
        {2560, 2560},  // Common size in Qwen3.8
        {640, 2560},   // Expert gate/up projection
        {2560, 640},   // Expert down projection
    };
    
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    int passed = 0;
    int failed = 0;
    
    for (int t = 0; t < num_tests; t++) {
        uint32_t M = test_sizes[t].M;
        uint32_t N = test_sizes[t].N;
        
        printf("Test %d: Matrix size %u x %u\n", t + 1, M, N);
        
        // Allocate test data
        float *weights = malloc(M * N * sizeof(float));
        float *input = malloc(N * sizeof(float));
        float *output_gpu = malloc(M * sizeof(float));
        float *output_cpu = malloc(M * sizeof(float));
        
        // Initialize with random data
        srand(42 + t);
        for (uint32_t i = 0; i < M * N; i++) {
            weights[i] = (float)(rand() % 100) / 100.0f - 0.5f;
        }
        for (uint32_t i = 0; i < N; i++) {
            input[i] = (float)(rand() % 100) / 100.0f - 0.5f;
        }
        
        // Create proper tensor structure matching Q38GGUFTensor layout
        Q38GGUFTensor tensor = {0};
        tensor.name.data = "test_weights";
        tensor.name.length = 12;
        tensor.n_dims = 2;
        tensor.shape[0] = N;
        tensor.shape[1] = M;
        tensor.type = Q38_GGML_F32;
        tensor.data = (const uint8_t *)weights;
        
        // Get GEMV context
        Q38VulkanGEMV *gemv = q38_vulkan_get_gemv();
        
        // Run GPU version
        memset(output_gpu, 0, M * sizeof(float));
        int gpu_result = q38_vulkan_gemv_f32(ctx, gemv, output_gpu, input, &tensor);
        
        // Run CPU reference
        memset(output_cpu, 0, M * sizeof(float));
        gemv_cpu(output_cpu, weights, input, M, N);
        
        // Compare results
        float max_error = 0.0f;
        int errors = 0;
        for (uint32_t i = 0; i < M; i++) {
            float error = fabsf(output_gpu[i] - output_cpu[i]);
            if (error > max_error) max_error = error;
            if (error > 0.01f) errors++;  // Allow small floating point differences
        }
        
        if (gpu_result && errors == 0) {
            printf("  ✓ PASS (max error: %.6f)\n", max_error);
            passed++;
        } else if (!gpu_result) {
            printf("  ✗ FAIL (GPU returned failure)\n");
            failed++;
        } else {
            printf("  ✗ FAIL (%d errors, max error: %.6f)\n", errors, max_error);
            failed++;
            
            // Print first few values for debugging
            printf("    First 5 outputs:\n");
            printf("    GPU: ");
            for (int i = 0; i < 5 && i < M; i++) printf("%.4f ", output_gpu[i]);
            printf("\n    CPU: ");
            for (int i = 0; i < 5 && i < M; i++) printf("%.4f ", output_cpu[i]);
            printf("\n");
        }
        
        free(weights);
        free(input);
        free(output_gpu);
        free(output_cpu);
    }
    
    printf("\n===================================\n");
    printf("Results: %d passed, %d failed\n", passed, failed);
    
    return failed > 0 ? 1 : 0;
}