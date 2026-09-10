#ifndef VULKAN_GEMV_H
#define VULKAN_GEMV_H

#include "vulkan_context.h"
#include "vulkan_buffers.h"
#include "vulkan_weight_cache.h"  // Weight caching system
#include "../qwen38/qwen38_gguf.h"  // For Q38GGUFTensor definition
#include <stdint.h>

// Vulkan-accelerated GEMV (General Matrix-Vector Multiplication)
// Primary compute hotspot in LLM inference

// Pipeline types for different quantization formats
typedef enum {
    Q38_VK_PIPELINE_GEMV_F32 = 0,
    Q38_VK_PIPELINE_GEMV_IQ4_NL,
    Q38_VK_PIPELINE_GEMV_Q8_0,
    Q38_VK_PIPELINE_GEMV_Q3_K,
    Q38_VK_PIPELINE_COUNT
} Q38VulkanPipelineType;

// GEMV kernel state
struct Q38VulkanGEMV {
    VkPipeline pipelines[Q38_VK_PIPELINE_COUNT];
    VkPipelineLayout pipeline_layouts[Q38_VK_PIPELINE_COUNT];
    VkDescriptorSetLayout descriptor_set_layouts[Q38_VK_PIPELINE_COUNT];
    
    // Pre-allocated descriptor pool
    VkDescriptorPool descriptor_pool;
    
    // Staging buffers for weights (if not unified memory)
    Q38VulkanBuffer weight_buffer;
    Q38VulkanBuffer input_buffer;
    Q38VulkanBuffer output_buffer;
    
    // Persistent codebook buffer for IQ4_NL (never destroyed during inference)
    Q38VulkanBuffer codebook_buffer;
    int codebook_initialized;
    
    // === WEIGHT CACHING SYSTEM ===
    Q38WeightCache weight_cache;  // Cache for reusing weight buffers
    bool use_cache;               // Whether caching is enabled
};

typedef struct Q38VulkanGEMV Q38VulkanGEMV;

// Initialize GEMV kernels with optional weight caching
int q38_vulkan_gemv_init(
    const Q38VulkanContext *ctx,
    Q38VulkanGEMV *gemv);

// Initialize GEMV with custom cache configuration
int q38_vulkan_gemv_init_with_cache(
    const Q38VulkanContext *ctx,
    Q38VulkanGEMV *gemv,
    const Q38WeightCacheConfig *cache_config);

// Cleanup GEMV resources
void q38_vulkan_gemv_cleanup(
    const Q38VulkanContext *ctx,
    Q38VulkanGEMV *gemv);

// Execute F32 GEMV: output = weights * input
// weights: M x N matrix (row-major)
// input: N-dimensional vector
// output: M-dimensional vector
int q38_vulkan_gemv_f32(
    const Q38VulkanContext *ctx,
    Q38VulkanGEMV *gemv,
    float *output,
    const float *input,
    const Q38GGUFTensor *tensor);

// Execute IQ4_NL quantized GEMV
int q38_vulkan_gemv_iq4nl(
    const Q38VulkanContext *ctx,
    Q38VulkanGEMV *gemv,
    float *output,
    const float *input,
    const Q38GGUFTensor *tensor);

// Execute Q8_0 quantized GEMV
int q38_vulkan_gemv_q8_0(
    const Q38VulkanContext *ctx,
    Q38VulkanGEMV *gemv,
    float *output,
    const float *input,
    const Q38GGUFTensor *tensor);

// Execute Q3_K quantized GEMV
int q38_vulkan_gemv_q3_k(
    const Q38VulkanContext *ctx,
    Q38VulkanGEMV *gemv,
    float *output,
    const float *input,
    const Q38GGUFTensor *tensor);

// === ASYNC OPERATIONS ===

// Begin async fetch of weights (returns immediately, may not be ready)
int q38_vulkan_gemv_fetch_begin(
    const Q38VulkanContext *ctx,
    Q38VulkanGEMV *gemv,
    const Q38GGUFTensor *tensor,
    uint32_t layer_id,
    bool *ready);

// Wait for async fetch to complete
int q38_vulkan_gemv_fetch_end(
    const Q38VulkanContext *ctx,
    Q38VulkanGEMV *gemv,
    const Q38GGUFTensor *tensor);

// === SPECULATIVE PREFETCH ===

// Prefetch predicted weights for next layer
void q38_vulkan_gemv_prefetch_next_layer(
    const Q38VulkanContext *ctx,
    Q38VulkanGEMV *gemv,
    uint32_t current_layer,
    const uint32_t *predicted_experts,
    uint32_t n_experts);

// Update prediction model with actual routing
void q38_vulkan_gemv_update_routing(
    Q38VulkanGEMV *gemv,
    uint32_t layer,
    const uint32_t *experts,
    uint32_t n_experts);

#endif // VULKAN_GEMV_H