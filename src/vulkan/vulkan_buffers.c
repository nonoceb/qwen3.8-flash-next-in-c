#include "vulkan/vulkan_buffers.h"
#include <stdio.h>
#include <string.h>

int q38_vulkan_buffer_create(
    const Q38VulkanContext *ctx,
    Q38VulkanBuffer *buf,
    uint64_t size,
    VkBufferUsageFlags usage,
    bool prefer_unified) {
    
    if (!ctx || !buf || !size) return 0;
    memset(buf, 0, sizeof(*buf));
    buf->size = size;
    
    // Create buffer
    VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE
    };
    
    VkResult result = vkCreateBuffer(ctx->device, &buffer_info, NULL, &buf->buffer);
    if (result != VK_SUCCESS) {
        fprintf(stderr, "Failed to create Vulkan buffer\n");
        return 0;
    }
    
    // Get memory requirements
    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(ctx->device, buf->buffer, &mem_reqs);
    
    // Find suitable memory type
    VkMemoryPropertyFlags required_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    
    if (prefer_unified && ctx->is_uma) {
        // For UMA, request device-local as well for optimal performance
        required_flags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    }
    
    uint32_t memory_type = q38_vulkan_find_unified_memory_type(ctx, required_flags);
    
    if (memory_type == UINT32_MAX) {
        // Fallback without device-local requirement
        required_flags &= ~VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        memory_type = q38_vulkan_find_unified_memory_type(ctx, required_flags);
    }
    
    if (memory_type == UINT32_MAX) {
        fprintf(stderr, "Failed to find suitable memory type\n");
        vkDestroyBuffer(ctx->device, buf->buffer, NULL);
        buf->buffer = VK_NULL_HANDLE;
        return 0;
    }
    
    // Check if we got unified memory
    VkMemoryPropertyFlags memory_props = 
        ctx->memory_properties.memoryTypes[memory_type].propertyFlags;
    buf->is_unified = (memory_props & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
                      (memory_props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    
    // Allocate memory
    VkMemoryAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mem_reqs.size,
        .memoryTypeIndex = memory_type
    };
    
    result = vkAllocateMemory(ctx->device, &alloc_info, NULL, &buf->memory);
    if (result != VK_SUCCESS) {
        fprintf(stderr, "Failed to allocate buffer memory\n");
        vkDestroyBuffer(ctx->device, buf->buffer, NULL);
        buf->buffer = VK_NULL_HANDLE;
        return 0;
    }
    
    // Bind memory to buffer
    vkBindBufferMemory(ctx->device, buf->buffer, buf->memory, 0);
    
    return 1;
}

int q38_vulkan_buffer_map(
    const Q38VulkanContext *ctx,
    Q38VulkanBuffer *buf) {
    
    if (!ctx || !buf || !buf->memory) return 0;
    
    if (buf->mapped_ptr) return 1;  // Already mapped
    
    VkResult result = vkMapMemory(ctx->device, buf->memory, 0, buf->size, 0, 
                                   &buf->mapped_ptr);
    if (result != VK_SUCCESS) {
        fprintf(stderr, "Failed to map buffer memory\n");
        return 0;
    }
    
    return 1;
}

void q38_vulkan_buffer_unmap(
    const Q38VulkanContext *ctx,
    Q38VulkanBuffer *buf) {
    
    if (!ctx || !buf) return;
    
    if (buf->mapped_ptr) {
        vkUnmapMemory(ctx->device, buf->memory);
        buf->mapped_ptr = NULL;
    }
}

void q38_vulkan_buffer_destroy(
    const Q38VulkanContext *ctx,
    Q38VulkanBuffer *buf) {
    
    if (!ctx || !buf) return;
    
    if (buf->mapped_ptr) {
        q38_vulkan_buffer_unmap(ctx, buf);
    }
    if (buf->memory) {
        vkFreeMemory(ctx->device, buf->memory, NULL);
    }
    if (buf->buffer) {
        vkDestroyBuffer(ctx->device, buf->buffer, NULL);
    }
    memset(buf, 0, sizeof(*buf));
}

int q38_vulkan_buffer_write(
    const Q38VulkanContext *ctx,
    Q38VulkanBuffer *buf,
    const void *data,
    uint64_t offset,
    uint64_t size) {
    
    if (!ctx || !buf || !data || offset + size > buf->size) return 0;
    
    // Map if not already mapped
    if (!buf->mapped_ptr) {
        if (!q38_vulkan_buffer_map(ctx, buf)) return 0;
    }
    
    // Direct memcpy for host-visible memory
    memcpy((char*)buf->mapped_ptr + offset, data, size);
    
    return 1;
}

int q38_vulkan_buffer_read(
    const Q38VulkanContext *ctx,
    Q38VulkanBuffer *buf,
    void *data,
    uint64_t offset,
    uint64_t size) {
    
    if (!ctx || !buf || !data || offset + size > buf->size) return 0;
    
    // Map if not already mapped
    if (!buf->mapped_ptr) {
        if (!q38_vulkan_buffer_map(ctx, buf)) return 0;
    }
    
    // Direct memcpy for host-visible memory
    memcpy(data, (char*)buf->mapped_ptr + offset, size);
    
    return 1;
}
