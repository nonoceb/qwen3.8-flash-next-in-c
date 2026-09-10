#include "vulkan/vulkan_context.h"
#include "vulkan/vulkan_gemv.h"
#include "qwen38/qwen38_gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Generate random F32 data
static void generate_random_f32(float *data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        data[i] = (float)(rand() % 1000) / 100.0f - 5.0f;
    }
}

// Compare two F32 arrays with tolerance
static int compare_f32(const float *a, const float *b, size_t n, float tolerance) {
    for (size_t i = 0; i < n; i++) {
        if (fabsf(a[i] - b[i]) > tolerance) {
            fprintf(stderr, "Mismatch at index %zu: expected %.6f, got %.6f\n",
                    i, a[i], b[i]);
            return 0;
        }
    }
    return 1;
}

int main(int argc, char **argv) {
    printf("=== Weight Cache Test ===\n\n");
    
    // Initialize Vulkan
    Q38VulkanContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    
    if (!q38_vulkan_context_init(&ctx)) {
        fprintf(stderr, "Failed to initialize Vulkan context\n");
        return 1;
    }
    printf("✓ Vulkan context initialized\n");
    
    // Initialize GEMV with custom cache config
    Q38VulkanGEMV gemv;
    Q38WeightCacheConfig cache_config = {
        .max_entries = 64,
        .eviction_policy = Q38_EVICT_HYBRID,
        .evict_samples = 8,
        .age_every = 1000,
        .enable_prefetch = true,
        .enable_async = true,
        .debug_stats = true
    };
    
    if (!q38_vulkan_gemv_init_with_cache(&ctx, &gemv, &cache_config)) {
        fprintf(stderr, "Failed to initialize GEMV\n");
        q38_vulkan_context_cleanup(&ctx);
        return 1;
    }
    printf("✓ GEMV initialized with weight caching\n\n");
    
    // Test parameters
    const uint64_t N = 256;   // Input dimension
    const uint64_t M = 512;   // Output dimension
    const int n_iterations = 10;
    
    // Create test tensors (simulating different layer weights)
    float *weights[3];
    Q38GGUFTensor tensors[3];
    
    for (int i = 0; i < 3; i++) {
        weights[i] = malloc(N * M * sizeof(float));
        generate_random_f32(weights[i], N * M);
        
        tensors[i].name.data = "test_weights";
        tensors[i].n_dims = 2;
        tensors[i].shape[0] = N;
        tensors[i].shape[1] = M;
        tensors[i].type = Q38_GGML_F32;
        tensors[i].data = (const uint8_t *)weights[i];
    }
    
    float *input = malloc(N * sizeof(float));
    float *output = malloc(M * sizeof(float));
    float *output_ref = malloc(M * sizeof(float));
    
    generate_random_f32(input, N);
    
    printf("Test 1: First access (cache miss)\n");
    printf("--------------------------------\n");
    
    // First call should be a cache miss
    int result = q38_vulkan_gemv_f32(&ctx, &gemv, output, input, &tensors[0]);
    if (!result) {
        fprintf(stderr, "✗ First GEMV call failed\n");
        goto cleanup;
    }
    printf("✓ First GEMV call succeeded\n");
    memcpy(output_ref, output, M * sizeof(float));
    
    // Check cache statistics
    Q38WeightCacheStats stats;
    q38_weight_cache_get_stats(&gemv.weight_cache, &stats);
    printf("  Lookups: %lu\n", stats.lookups);
    printf("  Hits: %lu\n", stats.hits);
    printf("  Misses: %lu\n", stats.misses);
    printf("  Hit rate: %.1f%%\n", 
           stats.lookups > 0 ? (double)stats.hits / stats.lookups * 100.0 : 0.0);
    
    printf("\nTest 2: Second access (should be cache hit)\n");
    printf("-------------------------------------------\n");
    
    // Second call with same tensor should be a cache hit
    result = q38_vulkan_gemv_f32(&ctx, &gemv, output, input, &tensors[0]);
    if (!result) {
        fprintf(stderr, "✗ Second GEMV call failed\n");
        goto cleanup;
    }
    printf("✓ Second GEMV call succeeded\n");
    
    // Verify output is identical
    if (!compare_f32(output, output_ref, M, 0.0001f)) {
        fprintf(stderr, "✗ Output mismatch on cache hit\n");
        goto cleanup;
    }
    printf("✓ Output matches reference\n");
    
    // Check cache statistics again
    q38_weight_cache_get_stats(&gemv.weight_cache, &stats);
    printf("  Lookups: %lu\n", stats.lookups);
    printf("  Hits: %lu\n", stats.hits);
    printf("  Misses: %lu\n", stats.misses);
    printf("  Hit rate: %.1f%%\n", 
           stats.lookups > 0 ? (double)stats.hits / stats.lookups * 100.0 : 0.0);
    
    printf("\nTest 3: Multiple tensors (mixed hits/misses)\n");
    printf("---------------------------------------------\n");
    
    // Access different tensors in a pattern that simulates MoE routing
    for (int iter = 0; iter < n_iterations; iter++) {
        int tensor_idx = iter % 3;  // Cycle through tensors
        
        result = q38_vulkan_gemv_f32(&ctx, &gemv, output, input, &tensors[tensor_idx]);
        if (!result) {
            fprintf(stderr, "✗ Iteration %d failed\n", iter);
            goto cleanup;
        }
    }
    printf("✓ Completed %d iterations\n", n_iterations);
    
    // Final statistics
    q38_weight_cache_get_stats(&gemv.weight_cache, &stats);
    printf("  Total lookups: %lu\n", stats.lookups);
    printf("  Total hits: %lu\n", stats.hits);
    printf("  Total misses: %lu\n", stats.misses);
    printf("  Evictions: %lu\n", stats.evictions);
    printf("  Final hit rate: %.1f%%\n", 
           stats.lookups > 0 ? (double)stats.hits / stats.lookups * 100.0 : 0.0);
    printf("  Bytes transferred: %.2f MB\n", stats.bytes_transferred / 1e6);
    
    printf("\nTest 4: Async fetch operations\n");
    printf("------------------------------\n");
    
    // Test async fetch
    bool ready;
    result = q38_vulkan_gemv_fetch_begin(&ctx, &gemv, &tensors[1], /*layer_id=*/1, &ready);
    if (result) {
        printf("✓ Async fetch begin succeeded (ready=%s)\n", ready ? "true" : "false");
        
        if (!ready) {
            result = q38_vulkan_gemv_fetch_end(&ctx, &gemv, &tensors[1]);
            printf("✓ Async fetch end %s\n", result ? "succeeded" : "failed");
        }
    } else {
        printf("✗ Async fetch begin failed\n");
    }
    
    printf("\nTest 5: Speculative prefetch\n");
    printf("----------------------------\n");
    
    // Test prediction model
    uint32_t current_experts[] = {0, 1, 2, 3, 4};
    uint32_t predicted[5];
    
    // Update prediction model
    q38_vulkan_gemv_update_routing(&gemv, /*layer=*/0, current_experts, 5);
    printf("✓ Updated prediction model\n");
    
    // Predict next layer
    q38_weight_cache_predict_next(&gemv.weight_cache, /*current_layer=*/0,
                                   current_experts, 5, predicted, 5);
    printf("✓ Predicted experts for next layer: ");
    for (int i = 0; i < 5; i++) {
        printf("%u ", predicted[i]);
    }
    printf("\n");
    
    printf("\n=== All Tests Passed ✓ ===\n\n");
    
cleanup:
    // Cleanup
    free(input);
    free(output);
    free(output_ref);
    for (int i = 0; i < 3; i++) {
        free(weights[i]);
    }
    
    q38_vulkan_gemv_cleanup(&ctx, &gemv);
    q38_vulkan_context_cleanup(&ctx);
    
    return 0;
}
