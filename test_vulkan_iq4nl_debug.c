#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vulkan/vulkan_context.h"
#include "vulkan/vulkan_buffers.h"
#include "vulkan/vulkan_gemv.h"
#include "vulkan/vulkan_wrapper.h"
#include "qwen38/qwen38_gguf.h"

int main() {
    printf("Testing Vulkan IQ4_NL with Debug Output\n");
    printf("========================================\n\n");
    
    // Initialize Vulkan
    Q38VulkanContext *ctx = q38_vulkan_get_context();
    if (!ctx) {
        printf("Failed to initialize Vulkan context\n");
        return 1;
    }
    
    printf("GPU: %s\n", ctx->device_name);
    printf("UMA: %s\n\n", ctx->is_uma ? "Yes" : "No");
    
    // Test different matrix sizes
    struct {
        uint32_t M;
        uint32_t N;
        const char *desc;
    } test_sizes[] = {
        {256, 256, "Small matrix"},
        {640, 2560, "Expert gate/up projection"},
        {2560, 640, "Expert down projection"},
        {2560, 2560, "Hidden dimension projection"},
    };
    
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    int passed = 0;
    int failed = 0;
    
    for (int t = 0; t < num_tests; t++) {
        uint32_t M = test_sizes[t].M;
        uint32_t N = test_sizes[t].N;
        
        printf("Test %d: %s (%u x %u)\n", t + 1, test_sizes[t].desc, M, N);
        
        // Allocate test data
        uint64_t blocks_per_row = N / 32;
        uint64_t weight_size = M * blocks_per_row * 18;  // IQ4_NL block size
        
        uint8_t *weights = malloc(weight_size);
        float *input = malloc(N * sizeof(float));
        float *output_gpu = malloc(M * sizeof(float));
        float *output_cpu = malloc(M * sizeof(float));
        
        // Initialize with simple pattern
        memset(weights, 0, weight_size);
        for (uint64_t i = 0; i < weight_size; i += 18) {
            // Set scale to 1.0 (fp16)
            weights[i] = 0x00;
            weights[i + 1] = 0x3C;
        }
        for (uint32_t i = 0; i < N; i++) {
            input[i] = 1.0f;
        }
        
        // Create tensor
        Q38GGUFTensor tensor = {0};
        tensor.name.data = "test_weights";
        tensor.name.length = 12;
        tensor.n_dims = 2;
        tensor.shape[0] = N;
        tensor.shape[1] = M;
        tensor.type = 20;  // Q38_GGML_IQ4_NL
        tensor.data = weights;
        
        // Get GEMV state
        Q38VulkanGEMV *gemv = q38_vulkan_get_gemv();
        if (!gemv) {
            printf("  Failed to get GEMV state\n");
            failed++;
            free(weights);
            free(input);
            free(output_gpu);
            free(output_cpu);
            continue;
        }
        
        // Run GPU computation
        printf("  Running GPU computation...\n");
        int result = q38_vulkan_gemv_iq4nl(ctx, gemv, output_gpu, input, &tensor);
        
        if (result) {
            // Check for NaN/Inf
            int has_nan = 0;
            int has_inf = 0;
            for (uint32_t i = 0; i < M && i < 10; i++) {
                if (!isfinite(output_gpu[i])) {
                    has_nan = isnan(output_gpu[i]);
                    has_inf = isinf(output_gpu[i]);
                    break;
                }
            }
            
            if (has_nan || has_inf) {
                printf("  ✗ FAIL: Output contains %s\n", has_nan ? "NaN" : "Inf");
                failed++;
            } else {
                printf("  ✓ PASS (output[0]=%.2f, output[1]=%.2f)\n", 
                       output_gpu[0], output_gpu[1]);
                passed++;
            }
        } else {
            printf("  ✗ FAIL: GPU computation returned error\n");
            failed++;
        }
        
        free(weights);
        free(input);
        free(output_gpu);
        free(output_cpu);
    }
    
    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", passed, failed);
    
    return failed > 0 ? 1 : 0;
}
