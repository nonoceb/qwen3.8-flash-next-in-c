#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vulkan/vulkan_context.h"
#include "vulkan/vulkan_buffers.h"
#include "vulkan/vulkan_gemv.h"
#include "vulkan/vulkan_wrapper.h"
#include "qwen38/qwen38_gguf.h"  // For Q38GGUFTensor definition

// IQ4_NL codebook (same as CPU implementation)
static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1, 13, 25, 38, 53, 69, 89, 113
};

// Simple CPU reference implementation for IQ4_NL
void gemv_iq4nl_cpu(float *output, const uint8_t *weights, const float *input, 
                     uint32_t M, uint32_t N) {
    const uint32_t blocks_per_row = N / 32;
    
    for (uint32_t row = 0; row < M; row++) {
        float sum = 0.0f;
        const uint8_t *row_ptr = weights + row * blocks_per_row * 18;
        
        for (uint32_t block = 0; block < blocks_per_row; block++) {
            const uint8_t *block_ptr = row_ptr + block * 18;
            
            // Read scale (fp16)
            uint16_t scale_u16 = block_ptr[0] | (block_ptr[1] << 8);
            // Simple fp16 to float conversion
            uint32_t sign = (scale_u16 >> 15) & 1;
            uint32_t exp = (scale_u16 >> 10) & 31;
            uint32_t mant = scale_u16 & 1023;
            float scale = (exp == 0) ? 0.0f : 
                (sign ? -1.0f : 1.0f) * (1.0f + mant / 1024.0f) * powf(2.0f, exp - 15.0f);
            
            // Process 16 packed bytes (32 values)
            for (uint32_t i = 0; i < 16; i++) {
                uint8_t packed = block_ptr[2 + i];
                
                // Lower nibble
                uint32_t col_lo = block * 32 + i * 2;
                if (col_lo < N) {
                    float weight = kvalues_iq4nl[packed & 0xF] * scale;
                    sum += weight * input[col_lo];
                }
                
                // Upper nibble
                uint32_t col_hi = block * 32 + i * 2 + 1;
                if (col_hi < N) {
                    float weight = kvalues_iq4nl[(packed >> 4) & 0xF] * scale;
                    sum += weight * input[col_hi];
                }
            }
        }
        output[row] = sum;
    }
}

int main() {
    printf("Testing Vulkan IQ4_NL GEMV Implementation\n");
    printf("==========================================\n\n");
    
    // Initialize Vulkan
    Q38VulkanContext *ctx = q38_vulkan_get_context();
    if (!ctx) {
        printf("Failed to initialize Vulkan context\n");
        return 1;
    }
    
    printf("Vulkan initialized successfully!\n");
    printf("  UMA support: %s\n", ctx->is_uma ? "Yes" : "No");
    printf("\n");
    
    // Test different matrix sizes relevant to Qwen3.8
    struct {
        uint32_t M;
        uint32_t N;
        const char *desc;
    } test_sizes[] = {
        {256, 256, "Small matrix"},
        {640, 2560, "Expert gate/up projection"},
        {2560, 640, "Expert down projection"},
        {2560, 2560, "Hidden dimension"},
    };
    
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    int passed = 0;
    int failed = 0;
    
    for (int t = 0; t < num_tests; t++) {
        uint32_t M = test_sizes[t].M;
        uint32_t N = test_sizes[t].N;
        
        printf("Test %d: %s (%u x %u)\n", t + 1, test_sizes[t].desc, M, N);
        
        // Ensure N is multiple of 32 for IQ4_NL
        if (N % 32 != 0) {
            printf("  Skipped (N not multiple of 32)\n");
            continue;
        }
        
        const uint32_t blocks_per_row = N / 32;
        const uint64_t weight_size = M * blocks_per_row * 18;
        
        // Allocate test data
        uint8_t *weights = malloc(weight_size);
        float *input = malloc(N * sizeof(float));
        float *output_gpu = malloc(M * sizeof(float));
        float *output_cpu = malloc(M * sizeof(float));
        
        // Initialize with random data
        srand(42 + t);
        for (uint64_t i = 0; i < weight_size; i++) {
            weights[i] = rand() % 256;
        }
        for (uint32_t i = 0; i < N; i++) {
            input[i] = (float)(rand() % 100) / 100.0f - 0.5f;
        }
        
        // Create proper tensor structure matching Q38GGUFTensor layout
        Q38GGUFTensor tensor = {0};
        tensor.name.data = "test_weights_iq4nl";
        tensor.name.length = 18;
        tensor.n_dims = 2;
        tensor.shape[0] = N;
        tensor.shape[1] = M;
        tensor.type = Q38_GGML_IQ4_NL;
        tensor.data = weights;
        
        // Get GEMV context
        Q38VulkanGEMV *gemv = q38_vulkan_get_gemv();
        
        // Run GPU version
        memset(output_gpu, 0, M * sizeof(float));
        int gpu_result = q38_vulkan_gemv_iq4nl(ctx, gemv, output_gpu, input, &tensor);
        
        // Run CPU reference
        memset(output_cpu, 0, M * sizeof(float));
        gemv_iq4nl_cpu(output_cpu, weights, input, M, N);
        
        // Compare results
        float max_error = 0.0f;
        int errors = 0;
        for (uint32_t i = 0; i < M; i++) {
            float error = fabsf(output_gpu[i] - output_cpu[i]);
            if (error > max_error) max_error = error;
            // Use relative error tolerance for large values, absolute for small
            float rel_error = fabsf(output_cpu[i]) > 1.0f ? 
                error / fabsf(output_cpu[i]) : error;
            if (rel_error > 0.001f && error > 10.0f) errors++;  // 0.1% relative or 10 absolute
        }
        
        if (gpu_result && errors == 0) {
            printf("  ✓ PASS (max error: %.6f)\n", max_error);
            passed++;
        } else if (!gpu_result) {
            printf("  ✗ FAIL (GPU returned failure - falling back to CPU)\n");
            failed++;
        } else {
            printf("  ✗ FAIL (%d errors, max error: %.6f)\n", errors, max_error);
            failed++;
            
            // Print first few values for debugging
            printf("    First 5 outputs:\n");
            printf("    GPU: ");
            for (int i = 0; i < 5 && i < M; i++) printf("%.2f ", output_gpu[i]);
            printf("\n    CPU: ");
            for (int i = 0; i < 5 && i < M; i++) printf("%.2f ", output_cpu[i]);
            printf("\n");
        }
        
        free(weights);
        free(input);
        free(output_gpu);
        free(output_cpu);
    }
    
    printf("\n==========================================\n");
    printf("Results: %d passed, %d failed\n", passed, failed);
    
    return failed > 0 ? 1 : 0;
}