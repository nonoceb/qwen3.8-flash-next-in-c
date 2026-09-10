#ifndef VULKAN_WRAPPER_H
#define VULKAN_WRAPPER_H

#include <stdbool.h>

// Global Vulkan context wrapper for Qwen inference
// Manages lazy initialization and cleanup

typedef struct Q38VulkanContext Q38VulkanContext;
typedef struct Q38VulkanGEMV Q38VulkanGEMV;

// Check if Vulkan is available and enabled
bool q38_vulkan_is_enabled(void);

// Enable or disable Vulkan (can be called before model load)
void q38_vulkan_set_enabled(bool enabled);

// Get global Vulkan context (initializes on first call)
Q38VulkanContext* q38_vulkan_get_context(void);

// Get global GEMV state (initializes on first call)
Q38VulkanGEMV* q38_vulkan_get_gemv(void);

// Cleanup Vulkan resources (call at program exit)
void q38_vulkan_shutdown(void);

#endif // VULKAN_WRAPPER_H