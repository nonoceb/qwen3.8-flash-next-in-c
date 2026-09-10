#ifndef VULKAN_CONTEXT_H
#define VULKAN_CONTEXT_H

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdint.h>

// Vulkan context for AMD 780M iGPU compute offload
// Target: Unified Memory Architecture (UMA) with zero-copy data sharing

struct Q38VulkanContext {
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue compute_queue;
    VkCommandPool command_pool;
    uint32_t compute_queue_family_index;
    
    // Memory properties for unified memory detection
    VkPhysicalDeviceMemoryProperties memory_properties;
    bool is_uma;  // True if unified memory architecture
    
    // Pipeline cache
    VkPipelineCache pipeline_cache;
};

typedef struct Q38VulkanContext Q38VulkanContext;

// Initialize Vulkan context for compute
int q38_vulkan_init(Q38VulkanContext *ctx);

// Cleanup Vulkan resources
void q38_vulkan_cleanup(Q38VulkanContext *ctx);

// Check if device supports unified memory
bool q38_vulkan_is_uma(const Q38VulkanContext *ctx);

// Find suitable memory type for unified memory
uint32_t q38_vulkan_find_unified_memory_type(
    const Q38VulkanContext *ctx,
    VkMemoryPropertyFlags required_flags);

#endif // VULKAN_CONTEXT_H