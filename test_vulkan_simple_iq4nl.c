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
    printf("Simple Vulkan IQ4_NL Test\n");
    printf("========================\n\n");
    
    // Initialize Vulkan
    Q38VulkanContext *ctx = q38_vulkan_get_context();
    if (!ctx) {
        printf("Failed to initialize Vulkan context\n");
        return 1;
    }
    
    printf("Vulkan initialized (UMA: %s)\n\n", ctx->is_uma ? "Yes" : "No");
    
    // Very simple test: 32x32 matrix (smallest possible)
    uint32_t M = 32;  // Must be >= 16
    uint32_t N = 32;  // Must be multiple of 32
    
    printf("Testing smallest matrix: %u x %u\n", M, N);
    
    // Create simple weight data
    uint64_t blocks_per_row = N / 32;  // 1 block
    uint64_t weight_size = M * blocks_per_row * 18;  // 18 bytes per block
    
    uint8_t *weights = malloc(weight_size);
    memset(weights, 0, weight_size);
    
    // Set all scales to 1.0 (fp16 = 0x3C00)
    for (uint64_t i = 0; i < M * blocks_per_row; i++) {
        weights[i * 18] = 0x00;
        weights[i * 18 + 1] = 0x3C;
    }
    
    // Create input vector (all ones)
    float *input = malloc(N * sizeof(float));
    for (uint32_t i = 0; i < N; i++) {
        input[i] = 1.0f;
    }
    
    // Output buffer
    float *output = malloc(M * sizeof(float));
    memset(output, 0, M * sizeof(float));
    
    // Create tensor
    Q38GGUFTensor tensor = {0};
    tensor.name.data = "test";
    tensor.name.length = 4;
    tensor.n_dims = 2;
    tensor.shape[0] = N;
    tensor.shape[1] = M;
    tensor.type = 20;  // Q38_GGML_IQ4_NL
    tensor.data = weights;
    
    // Get GEMV state
    Q38VulkanGEMV *gemv = q38_vulkan_get_gemv();
    if (!gemv) {
        printf("Failed to get GEMV state\n");
        free(weights);
        free(input);
        free(output);
        return 1;
    }
    
    printf("Running GPU computation...\n");
    int result = q38_vulkan_gemv_iq4nl(ctx, gemv, output, input, &tensor);
    
    if (result) {
        printf("Success! Output values:\n");
        for (int i = 0; i < 10 && i < M; i++) {
            printf("  output[%d] = %.4f\n", i, output[i]);
        }
        
        // Check for NaN/Inf
        int has_error = 0;
        for (uint32_t i = 0; i < M; i++) {
            if (!isfinite(output[i])) {
                printf("ERROR: output[%u] is not finite!\n", i);
                has_error = 1;
                break;
            }
        }
        
        if (!has_error) {
            printf("\n✓ All values are finite - TEST PASSED\n");
        } else {
            printf("\n✗ Found non-finite values - TEST FAILED\n");
        }
    } else {
        printf("✗ GPU computation failed\n");
    }
    
    free(weights);
    free(input);
    free(output);
    
    return 0;
}
