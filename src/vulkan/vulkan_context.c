#include "vulkan/vulkan_context.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Check Vulkan result and log error
#define VK_CHECK(call, msg) \
    do { \
        VkResult result = call; \
        if (result != VK_SUCCESS) { \
            fprintf(stderr, "Vulkan error %d at %s:%d: %s\n", \
                    result, __FILE__, __LINE__, msg); \
            return 0; \
        } \
    } while(0)

int q38_vulkan_init(Q38VulkanContext *ctx) {
    if (!ctx) return 0;
    memset(ctx, 0, sizeof(*ctx));
    
    // Create Vulkan instance
    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "Qwen3.8-Vulkan",
        .applicationVersion = 1,
        .pEngineName = "Qwen38-Compute",
        .engineVersion = 1,
        .apiVersion = VK_API_VERSION_1_2
    };
    
    VkInstanceCreateInfo instance_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info
    };
    
    VK_CHECK(vkCreateInstance(&instance_info, NULL, &ctx->instance), 
             "Failed to create Vulkan instance");
    
    // Enumerate physical devices
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(ctx->instance, &device_count, NULL);
    if (device_count == 0) {
        fprintf(stderr, "No Vulkan-capable GPUs found\n");
        q38_vulkan_cleanup(ctx);
        return 0;
    }
    
    VkPhysicalDevice *devices = malloc(device_count * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(ctx->instance, &device_count, devices);
    
    // Select first discrete GPU or integrated GPU with compute support
    ctx->physical_device = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < device_count; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devices[i], &props);
        
        // Look for AMD GPU (prefer AMD APU/iGPU for this project)
        if (props.vendorID == 0x1002 ||  // AMD
            props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ||
            props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            ctx->physical_device = devices[i];
            printf("Selected GPU: %s\n", props.deviceName);
            break;
        }
    }
    
    if (!ctx->physical_device && device_count > 0) {
        ctx->physical_device = devices[0];  // Fallback to first device
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(ctx->physical_device, &props);
        printf("Fallback to GPU: %s\n", props.deviceName);
    }
    
    free(devices);
    
    if (!ctx->physical_device) {
        fprintf(stderr, "No suitable GPU found\n");
        q38_vulkan_cleanup(ctx);
        return 0;
    }
    
    // Get memory properties
    vkGetPhysicalDeviceMemoryProperties(ctx->physical_device, 
                                        &ctx->memory_properties);
    
    // Detect UMA (Unified Memory Architecture)
    ctx->is_uma = q38_vulkan_is_uma(ctx);
    printf("Unified Memory Architecture: %s\n", ctx->is_uma ? "Yes" : "No");
    
    // Find compute queue family
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx->physical_device, 
                                              &queue_family_count, NULL);
    VkQueueFamilyProperties *queue_families = 
        malloc(queue_family_count * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(ctx->physical_device, 
                                              &queue_family_count, queue_families);
    
    ctx->compute_queue_family_index = UINT32_MAX;
    for (uint32_t i = 0; i < queue_family_count; i++) {
        if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            ctx->compute_queue_family_index = i;
            break;
        }
    }
    free(queue_families);
    
    if (ctx->compute_queue_family_index == UINT32_MAX) {
        fprintf(stderr, "No compute queue found\n");
        q38_vulkan_cleanup(ctx);
        return 0;
    }
    
    // Create logical device with compute queue
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = ctx->compute_queue_family_index,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority
    };
    
    VkDeviceCreateInfo device_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_info
    };
    
    VK_CHECK(vkCreateDevice(ctx->physical_device, &device_info, NULL, &ctx->device),
             "Failed to create logical device");
    
    // Get compute queue
    vkGetDeviceQueue(ctx->device, ctx->compute_queue_family_index, 0, 
                     &ctx->compute_queue);
    
    // Create command pool
    VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = ctx->compute_queue_family_index
    };
    
    VK_CHECK(vkCreateCommandPool(ctx->device, &pool_info, NULL, &ctx->command_pool),
             "Failed to create command pool");
    
    // Create pipeline cache
    VkPipelineCacheCreateInfo cache_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO
    };
    
    vkCreatePipelineCache(ctx->device, &cache_info, NULL, &ctx->pipeline_cache);
    
    return 1;
}

void q38_vulkan_cleanup(Q38VulkanContext *ctx) {
    if (!ctx) return;
    
    if (ctx->pipeline_cache) {
        vkDestroyPipelineCache(ctx->device, ctx->pipeline_cache, NULL);
    }
    if (ctx->command_pool) {
        vkDestroyCommandPool(ctx->device, ctx->command_pool, NULL);
    }
    if (ctx->device) {
        vkDestroyDevice(ctx->device, NULL);
    }
    if (ctx->instance) {
        vkDestroyInstance(ctx->instance, NULL);
    }
    memset(ctx, 0, sizeof(*ctx));
}

bool q38_vulkan_is_uma(const Q38VulkanContext *ctx) {
    if (!ctx || !ctx->physical_device) return false;
    
    // Check if all memory types are device-local and host-visible
    // This indicates unified memory architecture
    for (uint32_t i = 0; i < ctx->memory_properties.memoryTypeCount; i++) {
        VkMemoryPropertyFlags flags = 
            ctx->memory_properties.memoryTypes[i].propertyFlags;
        
        // UMA typically has memory that is both device-local and host-visible
        if ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
            (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
            return true;
        }
    }
    
    return false;
}

uint32_t q38_vulkan_find_unified_memory_type(
    const Q38VulkanContext *ctx,
    VkMemoryPropertyFlags required_flags) {
    
    if (!ctx || !ctx->physical_device) return UINT32_MAX;
    
    // For UMA, prefer memory that is:
    // - Device local (fast GPU access)
    // - Host visible (CPU can access)
    // - Host coherent (no explicit cache management)
    VkMemoryPropertyFlags uma_flags = 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    
    // Try to find UMA memory first
    for (uint32_t i = 0; i < ctx->memory_properties.memoryTypeCount; i++) {
        VkMemoryPropertyFlags flags = 
            ctx->memory_properties.memoryTypes[i].propertyFlags;
        
        if ((flags & uma_flags) == uma_flags &&
            (flags & required_flags) == required_flags) {
            return i;
        }
    }
    
    // Fallback to any host-visible memory
    for (uint32_t i = 0; i < ctx->memory_properties.memoryTypeCount; i++) {
        VkMemoryPropertyFlags flags = 
            ctx->memory_properties.memoryTypes[i].propertyFlags;
        
        if ((flags & required_flags) == required_flags) {
            return i;
        }
    }
    
    return UINT32_MAX;
}
