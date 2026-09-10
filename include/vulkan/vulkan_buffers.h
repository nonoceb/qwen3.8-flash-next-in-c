#ifndef VULKAN_BUFFERS_H
#define VULKAN_BUFFERS_H

#include "vulkan_context.h"
#include <stdint.h>
#include <stddef.h>

// Unified memory buffer management for UMA systems
// On AMD 780M, we can share memory between CPU and GPU without copies

typedef struct {
    VkBuffer buffer;
    VkDeviceMemory memory;
    void *mapped_ptr;  // CPU-accessible pointer (for UMA)
    uint64_t size;
    bool is_unified;   // True if using unified memory
} Q38VulkanBuffer;

// Create a buffer with unified memory (zero-copy on UMA)
int q38_vulkan_buffer_create(
    const Q38VulkanContext *ctx,
    Q38VulkanBuffer *buf,
    uint64_t size,
    VkBufferUsageFlags usage,
    bool prefer_unified);

// Map buffer to CPU address space (no-op for unified memory)
int q38_vulkan_buffer_map(
    const Q38VulkanContext *ctx,
    Q38VulkanBuffer *buf);

// Unmap buffer from CPU address space
void q38_vulkan_buffer_unmap(
    const Q38VulkanContext *ctx,
    Q38VulkanBuffer *buf);

// Destroy buffer and free memory
void q38_vulkan_buffer_destroy(
    const Q38VulkanContext *ctx,
    Q38VulkanBuffer *buf);

// Write data to buffer (direct memcpy for unified memory)
int q38_vulkan_buffer_write(
    const Q38VulkanContext *ctx,
    Q38VulkanBuffer *buf,
    const void *data,
    uint64_t offset,
    uint64_t size);

// Read data from buffer (direct memcpy for unified memory)
int q38_vulkan_buffer_read(
    const Q38VulkanContext *ctx,
    Q38VulkanBuffer *buf,
    void *data,
    uint64_t offset,
    uint64_t size);

#endif // VULKAN_BUFFERS_H