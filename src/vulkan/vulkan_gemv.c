#ifndef _GNU_SOURCE
#define _GNU_SOURCE  // For readlink on some systems
#endif
#include "vulkan/vulkan_gemv.h"
#include "qwen38/qwen38_gguf.h"  // For Q38GGUFTensor definition
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <unistd.h>
#ifdef __linux__
#include <linux/limits.h>
#endif

// Get the directory containing the executable
static int get_executable_dir(char *buf, size_t bufsize) {
#if defined(__linux__) || defined(__APPLE__)
    ssize_t len = readlink("/proc/self/exe", buf, bufsize - 1);
    if (len == -1) return 0;
    buf[len] = '\0';
#elif defined(_WIN32)
    DWORD len = GetModuleFileNameA(NULL, buf, bufsize - 1);
    if (len == 0) return 0;
    buf[len] = '\0';
#else
    return 0;  // Unsupported platform
#endif
    
    // Remove executable name to get directory
    char *last_slash = strrchr(buf, '/');
    if (last_slash) *last_slash = '\0';
    return 1;
}

// Try multiple paths to find a shader file
static FILE* find_shader_file(const char *shader_name) {
    char path[PATH_MAX];
    
    // List of paths to try (in order of priority)
    const char *search_paths[] = {
        "shaders/spv",                    // Relative to CWD (project root)
        "../shaders/spv",                 // Relative to bin/ directory
        "../../shaders/spv",              // Relative to build/bin/ directory
        NULL
    };
    
    // Try each search path
    for (int i = 0; search_paths[i]; i++) {
        snprintf(path, sizeof(path), "%s/%s", search_paths[i], shader_name);
        FILE *f = fopen(path, "rb");
        if (f) {
            // fprintf(stderr, "DEBUG: Found shader at: %s\n", path);
            return f;
        }
    }
    
    // Try relative to executable location as last resort
    char exec_dir[PATH_MAX];
    if (get_executable_dir(exec_dir, sizeof(exec_dir))) {
        snprintf(path, sizeof(path), "%s/../shaders/spv/%s", exec_dir, shader_name);
        FILE *f = fopen(path, "rb");
        if (f) {
            // fprintf(stderr, "DEBUG: Found shader at: %s\n", path);
            return f;
        }
    }
    
    return NULL;
}

// Helper to load SPIR-V shader from file
static uint32_t* load_spirv_shader(const char *filename, size_t *size) {
    // Extract just the shader name if a path was provided
    const char *shader_name = strrchr(filename, '/');
    if (shader_name) {
        shader_name++;  // Skip the slash
    } else {
        shader_name = filename;
    }
    
    FILE *f = find_shader_file(shader_name);
    if (!f) {
        fprintf(stderr, "ERROR: Failed to find shader file: %s\n", shader_name);
        fprintf(stderr, "  Searched in: shaders/spv/, ../shaders/spv/, ../../shaders/spv/\n");
        return NULL;
    }
    
    fseek(f, 0, SEEK_END);
    *size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    uint32_t *code = malloc(*size);
    if (!code) {
        fclose(f);
        return NULL;
    }
    
    if (fread(code, 1, *size, f) != *size) {
        free(code);
        fclose(f);
        return NULL;
    }
    
    fclose(f);
    return code;
}

// Create compute pipeline from SPIR-V shader (currently unused - inline creation in gemv functions)
/*
static int create_compute_pipeline(
    const Q38VulkanContext *ctx,
    VkPipeline *pipeline,
    VkPipelineLayout *layout,
    VkDescriptorSetLayout *desc_layout,
    const char *shader_file,
    uint32_t push_constant_size) {
    
    // Load shader
    size_t shader_size;
    uint32_t *shader_code = load_spirv_shader(shader_file, &shader_size);
    if (!shader_code) return 0;
    
    // Create shader module
    VkShaderModuleCreateInfo shader_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = shader_size,
        .pCode = shader_code
    };
    
    VkShaderModule shader_module;
    VkResult result = vkCreateShaderModule(ctx->device, &shader_info, NULL, &shader_module);
    free(shader_code);
    
    if (result != VK_SUCCESS) {
        fprintf(stderr, "Failed to create shader module from %s\n", shader_file);
        return 0;
    }
    
    // Create descriptor set layout (3 bindings: weights, input, output)
    VkDescriptorSetLayoutBinding bindings[] = {
        {  // Weights buffer
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
        },
        {  // Input buffer
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
        },
        {  // Output buffer
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
        }
    };
    
    VkDescriptorSetLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = bindings
    };
    
    result = vkCreateDescriptorSetLayout(ctx->device, &layout_info, NULL, desc_layout);
    if (result != VK_SUCCESS) {
        vkDestroyShaderModule(ctx->device, shader_module, NULL);
        fprintf(stderr, "Failed to create descriptor set layout\n");
        return 0;
    }
    
    // Create pipeline layout
    VkPushConstantRange push_constant_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = push_constant_size
    };
    
    VkPipelineLayoutCreateInfo pipeline_layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = desc_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_constant_range
    };
    
    result = vkCreatePipelineLayout(ctx->device, &pipeline_layout_info, NULL, layout);
    if (result != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(ctx->device, *desc_layout, NULL);
        vkDestroyShaderModule(ctx->device, shader_module, NULL);
        fprintf(stderr, "Failed to create pipeline layout\n");
        return 0;
    }
    
    // Create compute pipeline
    VkComputePipelineCreateInfo pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .layout = *layout,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader_module,
            .pName = "main"
        }
    };
    
    result = vkCreateComputePipelines(ctx->device, ctx->pipeline_cache, 1, 
                                       &pipeline_info, NULL, pipeline);
    
    vkDestroyShaderModule(ctx->device, shader_module, NULL);
    
    if (result != VK_SUCCESS) {
        vkDestroyPipelineLayout(ctx->device, *layout, NULL);
        vkDestroyDescriptorSetLayout(ctx->device, *desc_layout, NULL);
        fprintf(stderr, "Failed to create compute pipeline\n");
        return 0;
    }
    
    return 1;
}
*/

int q38_vulkan_gemv_init(
    const Q38VulkanContext *ctx,
    Q38VulkanGEMV *gemv) {
    
    // Initialize with default cache configuration
    return q38_vulkan_gemv_init_with_cache(ctx, gemv, NULL);
}

int q38_vulkan_gemv_init_with_cache(
    const Q38VulkanContext *ctx,
    Q38VulkanGEMV *gemv,
    const Q38WeightCacheConfig *cache_config) {
    
    if (!ctx || !gemv) return 0;
    memset(gemv, 0, sizeof(*gemv));
    
    // Create descriptor pool
    VkDescriptorPoolSize pool_sizes[] = {
        {
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 4096  // Enough for thousands of GEMV calls
        }
    };
    
    VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,  // Allow freeing
        .maxSets = 1024,
        .poolSizeCount = 1,
        .pPoolSizes = pool_sizes
    };
    
    VkResult result = vkCreateDescriptorPool(ctx->device, &pool_info, NULL, 
                                              &gemv->descriptor_pool);
    if (result != VK_SUCCESS) {
        fprintf(stderr, "Failed to create descriptor pool\n");
        return 0;
    }
    
    // Initialize weight cache
    if (q38_weight_cache_init(ctx, &gemv->weight_cache, cache_config)) {
        gemv->use_cache = true;
        fprintf(stderr, "[gemv] Weight caching enabled\n");
    } else {
        gemv->use_cache = false;
        fprintf(stderr, "[gemv] Weight caching disabled (init failed)\n");
    }
    
    // Check environment variable to disable cache
    if (getenv("Q38_DISABLE_CACHE")) {
        gemv->use_cache = false;
        fprintf(stderr, "[gemv] Weight caching disabled by environment variable\n");
    }
    
    printf("Vulkan GEMV initialized (pipelines will be created on-demand)\n");
    return 1;
}

void q38_vulkan_gemv_cleanup(
    const Q38VulkanContext *ctx,
    Q38VulkanGEMV *gemv) {
    
    if (!ctx || !gemv) return;
    
    // Clean up weight cache first
    if (gemv->use_cache) {
        q38_weight_cache_cleanup(ctx, &gemv->weight_cache);
    }
    
    for (int i = 0; i < Q38_VK_PIPELINE_COUNT; i++) {
        if (gemv->pipelines[i]) {
            vkDestroyPipeline(ctx->device, gemv->pipelines[i], NULL);
        }
        if (gemv->pipeline_layouts[i]) {
            vkDestroyPipelineLayout(ctx->device, gemv->pipeline_layouts[i], NULL);
        }
        if (gemv->descriptor_set_layouts[i]) {
            vkDestroyDescriptorSetLayout(ctx->device, gemv->descriptor_set_layouts[i], NULL);
        }
    }
    
    if (gemv->descriptor_pool) {
        vkDestroyDescriptorPool(ctx->device, gemv->descriptor_pool, NULL);
    }
    
    q38_vulkan_buffer_destroy(ctx, &gemv->weight_buffer);
    q38_vulkan_buffer_destroy(ctx, &gemv->input_buffer);
    q38_vulkan_buffer_destroy(ctx, &gemv->output_buffer);
    q38_vulkan_buffer_destroy(ctx, &gemv->codebook_buffer);  // Clean up persistent codebook
    
    memset(gemv, 0, sizeof(*gemv));
}

int q38_vulkan_gemv_f32(
    const Q38VulkanContext *ctx,
    Q38VulkanGEMV *gemv,
    float *output,
    const float *input,
    const Q38GGUFTensor *tensor) {
    
    // fprintf(stderr, "DEBUG: q38_vulkan_gemv_f32 called\n");
    
    if (!ctx || !gemv || !output || !input || !tensor) {
        fprintf(stderr, "ERROR: Null parameter check failed\n");
        return 0;
    }
    if (tensor->n_dims != 2) {
        fprintf(stderr, "ERROR: Tensor is not 2D (n_dims=%d)\n", tensor->n_dims);
        return 0;
    }
    
    const uint64_t N = tensor->shape[0];  // Input dimension
    const uint64_t M = tensor->shape[1];  // Output dimension (rows)
    
    // fprintf(stderr, "DEBUG: Matrix dimensions: M=%lu, N=%lu\n", M, N);
    
    // Skip very small matrices - not worth GPU overhead
    if (M < 16 || N < 16) {
        // fprintf(stderr, "DEBUG: Matrix too small, falling back to CPU\n");
        return 0;  // Fall back to CPU
    }
    
    // Validate input data
    const float *weights = (const float *)tensor->data;
    bool has_nan = false;
    for (int i = 0; i < 10 && i < M * N; i++) {
        if (!isfinite(weights[i])) {
            has_nan = true;
        }
    }
    if (has_nan) return 0;
    
    for (int i = 0; i < 10 && i < N; i++) {
        if (!isfinite(input[i])) {
            has_nan = true;
        }
    }
    if (has_nan) return 0;
    
    // === WEIGHT CACHING ===
    Q38VulkanBuffer *weight_buf = NULL;
    bool using_cache = false;
    
    const uint64_t weight_size = N * M * sizeof(float);
    
    if (gemv->use_cache) {
        // Try to get cached weight buffer
        // fprintf(stderr, "DEBUG: Looking up tensor_data=%p, size=%lu\n", tensor->data, weight_size);
        weight_buf = q38_weight_cache_get_ext(ctx, &gemv->weight_cache, 
                                               tensor->data, weight_size,
                                               Q38_GGML_F32, /*layer_id=*/0,
                                               (uint32_t)M, (uint32_t)N);  // Pass dimensions for validation
        if (weight_buf) {
            // Verify the cached buffer is large enough
            if (weight_buf->size >= weight_size) {
                using_cache = true;
                // fprintf(stderr, "DEBUG: Using cached weight buffer (size=%lu, need=%lu)\n", weight_buf->size, weight_size);            } else {
                // fprintf(stderr, "DEBUG: Cached buffer too small (%lu < %lu), reallocating\n", weight_buf->size, weight_size);                weight_buf = NULL;
            }
        } else {
            // fprintf(stderr, "DEBUG: Cache miss for weights\n");
        }
    }
    
    // Create pipeline if not already created
    if (!gemv->pipelines[Q38_VK_PIPELINE_GEMV_F32]) {
        // fprintf(stderr, "DEBUG: Creating F32 GEMV pipeline (M=%lu, N=%lu)\n", M, N);
        
        // Try to load precompiled SPIR-V shader
        const char *shader_paths[] = {
            "shaders/spv/gemv_f32.spv",  // Try simple version first
            "shaders/spv/gemv_f32_optimized.spv",
            NULL
        };
        
        for (int i = 0; shader_paths[i]; i++) {
            // fprintf(stderr, "DEBUG: Trying shader: %s\n", shader_paths[i]);
            size_t shader_size;
            uint32_t *shader_code = load_spirv_shader(shader_paths[i], &shader_size);
            if (shader_code) {
                // fprintf(stderr, "DEBUG: Loaded shader, size=%zu bytes\n", shader_size);
                // Create shader module
                VkShaderModuleCreateInfo shader_info = {
                    .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                    .codeSize = shader_size,
                    .pCode = shader_code
                };
                
                VkShaderModule shader_module;
                VkResult result = vkCreateShaderModule(ctx->device, &shader_info, 
                                                        NULL, &shader_module);
                free(shader_code);
                
                if (result == VK_SUCCESS) {
                    // fprintf(stderr, "DEBUG: Shader module created successfully\n");
                    // Create descriptor set layout
                    VkDescriptorSetLayoutBinding bindings[] = {
                        {  // Weights buffer
                            .binding = 0,
                            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            .descriptorCount = 1,
                            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
                        },
                        {  // Input buffer
                            .binding = 1,
                            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            .descriptorCount = 1,
                            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
                        },
                        {  // Output buffer
                            .binding = 2,
                            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            .descriptorCount = 1,
                            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
                        }
                    };
                    
                    VkDescriptorSetLayoutCreateInfo layout_info = {
                        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                        .bindingCount = 3,
                        .pBindings = bindings
                    };
                    
                    vkCreateDescriptorSetLayout(ctx->device, &layout_info, NULL,
                        &gemv->descriptor_set_layouts[Q38_VK_PIPELINE_GEMV_F32]);
                    
                    // Create pipeline layout with push constants
                    typedef struct {
                        uint32_t M;
                        uint32_t N;
                    } PushConstants;
                    
                    VkPushConstantRange push_range = {
                        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                        .offset = 0,
                        .size = sizeof(PushConstants)
                    };
                    
                    VkPipelineLayoutCreateInfo pipeline_layout_info = {
                        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                        .setLayoutCount = 1,
                        .pSetLayouts = &gemv->descriptor_set_layouts[Q38_VK_PIPELINE_GEMV_F32],
                        .pushConstantRangeCount = 1,
                        .pPushConstantRanges = &push_range
                    };
                    
                    vkCreatePipelineLayout(ctx->device, &pipeline_layout_info, NULL,
                        &gemv->pipeline_layouts[Q38_VK_PIPELINE_GEMV_F32]);
                    
                    // Create compute pipeline
                    VkComputePipelineCreateInfo pipeline_info = {
                        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                        .layout = gemv->pipeline_layouts[Q38_VK_PIPELINE_GEMV_F32],
                        .stage = {
                            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                            .module = shader_module,
                            .pName = "main"
                        }
                    };
                    
                    vkCreateComputePipelines(ctx->device, ctx->pipeline_cache, 1,
                        &pipeline_info, NULL, &gemv->pipelines[Q38_VK_PIPELINE_GEMV_F32]);
                    
                    // fprintf(stderr, "DEBUG: Pipeline created: %p\n", (void*)gemv->pipelines[Q38_VK_PIPELINE_GEMV_F32]);
                    
                    vkDestroyShaderModule(ctx->device, shader_module, NULL);
                    break;
                }
            }
        }
    }
    
    if (!gemv->pipelines[Q38_VK_PIPELINE_GEMV_F32]) {
        fprintf(stderr, "ERROR: Failed to create F32 GEMV pipeline\n");
        return 0;
    }
    
    // fprintf(stderr, "DEBUG: Pipeline ready, allocating buffers\n");
    
    // Allocate or resize buffers
    const uint64_t input_size = N * sizeof(float);
    const uint64_t output_size = M * sizeof(float);
    
    // Use cached weight buffer if available, otherwise allocate/resize
    if (!using_cache) {
        if (gemv->weight_buffer.size < weight_size) {
            q38_vulkan_buffer_destroy(ctx, &gemv->weight_buffer);
            if (!q38_vulkan_buffer_create(ctx, &gemv->weight_buffer, weight_size,
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                           VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                           true)) {
                return 0;
            }
        }
        weight_buf = &gemv->weight_buffer;
    }
    
    if (gemv->input_buffer.size < input_size) {
        q38_vulkan_buffer_destroy(ctx, &gemv->input_buffer);
        if (!q38_vulkan_buffer_create(ctx, &gemv->input_buffer, input_size,
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                       true)) {
            return 0;
        }
    }
    
    if (gemv->output_buffer.size < output_size) {
        q38_vulkan_buffer_destroy(ctx, &gemv->output_buffer);
        if (!q38_vulkan_buffer_create(ctx, &gemv->output_buffer, output_size,
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                       true)) {
            return 0;
        }
    }
    
    // Upload weights and input data
    if (!using_cache) {
        // Upload weights (not cached)
        if (!q38_vulkan_buffer_write(ctx, weight_buf, tensor->data, 0, weight_size)) {
            return 0;
        }
    }
    if (!q38_vulkan_buffer_write(ctx, &gemv->input_buffer, input, 0, input_size)) {
        return 0;
    }
    
    // Allocate descriptor set
    VkDescriptorSetAllocateInfo desc_alloc = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = gemv->descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &gemv->descriptor_set_layouts[Q38_VK_PIPELINE_GEMV_F32]
    };
    
    VkDescriptorSet descriptor_set;
    VkResult result = vkAllocateDescriptorSets(ctx->device, &desc_alloc, &descriptor_set);
    if (result != VK_SUCCESS) {
        fprintf(stderr, "ERROR: Failed to allocate descriptor set (result=%d)\n", result);
        return 0;
    }
    // fprintf(stderr, "DEBUG: Descriptor set allocated\n");
    
    // Update descriptor sets
    VkDescriptorBufferInfo weight_desc = {
        .buffer = weight_buf->buffer,
        .offset = 0,
        .range = weight_size
    };
    VkDescriptorBufferInfo input_desc = {
        .buffer = gemv->input_buffer.buffer,
        .offset = 0,
        .range = input_size
    };
    VkDescriptorBufferInfo output_desc = {
        .buffer = gemv->output_buffer.buffer,
        .offset = 0,
        .range = output_size
    };
    
    VkWriteDescriptorSet writes[] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &weight_desc
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &input_desc
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &output_desc
        }
    };
    
    vkUpdateDescriptorSets(ctx->device, 3, writes, 0, NULL);
    
    // Create command buffer
    VkCommandBufferAllocateInfo cmd_alloc = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = ctx->command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };
    
    VkCommandBuffer cmd_buffer;
    vkAllocateCommandBuffers(ctx->device, &cmd_alloc, &cmd_buffer);
    
    // Begin command buffer
    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    
    vkBeginCommandBuffer(cmd_buffer, &begin_info);
    
    // Bind pipeline and descriptor set
    vkCmdBindPipeline(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      gemv->pipelines[Q38_VK_PIPELINE_GEMV_F32]);
    vkCmdBindDescriptorSets(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           gemv->pipeline_layouts[Q38_VK_PIPELINE_GEMV_F32],
                           0, 1, &descriptor_set, 0, NULL);
    
    // Push constants
    struct {
        uint32_t M;
        uint32_t N;
    } push_constants = { (uint32_t)M, (uint32_t)N };
    
    vkCmdPushConstants(cmd_buffer, gemv->pipeline_layouts[Q38_VK_PIPELINE_GEMV_F32],
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants),
                       &push_constants);
    
    // Dispatch compute shader
    uint32_t workgroups = (M + 255) / 256;
    vkCmdDispatch(cmd_buffer, workgroups, 1, 1);
    
    // Add memory barrier to ensure shader writes are visible
    VkMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT
    };
    vkCmdPipelineBarrier(cmd_buffer,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT,
                         0, 1, &barrier, 0, NULL, 0, NULL);
    
    vkEndCommandBuffer(cmd_buffer);
    
    // Submit and wait
    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd_buffer
    };
    
    VkFence fence;
    VkFenceCreateInfo fence_info = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO
    };
    vkCreateFence(ctx->device, &fence_info, NULL, &fence);
    
    vkQueueSubmit(ctx->compute_queue, 1, &submit_info, fence);
    vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);
    
    // Read back results
    q38_vulkan_buffer_read(ctx, &gemv->output_buffer, output, 0, output_size);
    
    // Validate output (silently fall back to CPU if NaN detected)
    bool output_has_nan = false;
    for (int i = 0; i < 10 && i < M; i++) {
        if (!isfinite(output[i])) {
            output_has_nan = true;
        }
    }
    if (output_has_nan) {
        memset(output, 0, output_size);
        return 0;
    }
    
    // Cleanup
    vkDestroyFence(ctx->device, fence, NULL);
    vkFreeCommandBuffers(ctx->device, ctx->command_pool, 1, &cmd_buffer);
    vkFreeDescriptorSets(ctx->device, gemv->descriptor_pool, 1, &descriptor_set);
    
    return 1;
}

// IQ4_NL codebook (same as CPU implementation)
static const float kIQ4NLCodebook[16] = {
    -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
    1.0f, 13.0f, 25.0f, 38.0f, 53.0f, 69.0f, 89.0f, 113.0f
};

int q38_vulkan_gemv_iq4nl(
    const Q38VulkanContext *ctx,
    Q38VulkanGEMV *gemv,
    float *output,
    const float *input,
    const Q38GGUFTensor *tensor) {
    
    if (!ctx || !gemv || !output || !input || !tensor) return 0;
    if (tensor->n_dims != 2) return 0;
    if (tensor->type != 20) return 0;  // Q38_GGML_IQ4_NL
    
    const uint64_t N = tensor->shape[0];  // Input dimension
    const uint64_t M = tensor->shape[1];  // Output dimension (rows)
    
    // Skip very small matrices
    if (M < 16 || N < 32) return 0;
    
    // IQ4_NL block size is 32 elements
    if (N % 32 != 0) return 0;
    const uint64_t blocks_per_row = N / 32;
    
    // Block format: 2 bytes scale (fp16) + 16 bytes packed weights = 18 bytes per block
    // We pad to 20 bytes (5 uints) for optimal GPU memory access on UMA systems
    const uint64_t padded_weight_size = M * blocks_per_row * 20;  // 20 bytes per padded block
    const uint64_t input_size = N * sizeof(float);
    const uint64_t output_size = M * sizeof(float);
    
    // Create pipeline if not already created
    if (!gemv->pipelines[Q38_VK_PIPELINE_GEMV_IQ4_NL]) {
        size_t shader_size;
        uint32_t *shader_code = load_spirv_shader("shaders/spv/gemv_iq4nl.spv", &shader_size);
        if (!shader_code) {
            fprintf(stderr, "Failed to load IQ4_NL shader\n");
            return 0;
        }
        
        VkShaderModuleCreateInfo shader_info = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = shader_size,
            .pCode = shader_code
        };
        
        VkShaderModule shader_module;
        VkResult result = vkCreateShaderModule(ctx->device, &shader_info, NULL, &shader_module);
        free(shader_code);
        
        if (result != VK_SUCCESS) {
            fprintf(stderr, "Failed to create IQ4_NL shader module\n");
            return 0;
        }
        
        // Create descriptor set layout (4 bindings: weights, input, output, codebook)
        VkDescriptorSetLayoutBinding bindings[] = {
            {  // Weights buffer (packed IQ4_NL blocks)
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
            },
            {  // Input buffer
                .binding = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
            },
            {  // Output buffer
                .binding = 2,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
            },
            {  // Codebook buffer
                .binding = 3,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
            }
        };
        
        VkDescriptorSetLayoutCreateInfo layout_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 4,
            .pBindings = bindings
        };
        
        vkCreateDescriptorSetLayout(ctx->device, &layout_info, NULL,
            &gemv->descriptor_set_layouts[Q38_VK_PIPELINE_GEMV_IQ4_NL]);
        
        // Push constants: M, N, blocks_per_row
        typedef struct {
            uint32_t M;
            uint32_t N;
            uint32_t blocks_per_row;
        } PushConstants;
        
        VkPushConstantRange push_range = {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(PushConstants)
        };
        
        VkPipelineLayoutCreateInfo pipeline_layout_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &gemv->descriptor_set_layouts[Q38_VK_PIPELINE_GEMV_IQ4_NL],
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_range
        };
        
        vkCreatePipelineLayout(ctx->device, &pipeline_layout_info, NULL,
            &gemv->pipeline_layouts[Q38_VK_PIPELINE_GEMV_IQ4_NL]);
        
        VkComputePipelineCreateInfo pipeline_info = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .layout = gemv->pipeline_layouts[Q38_VK_PIPELINE_GEMV_IQ4_NL],
            .stage = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = shader_module,
                .pName = "main"
            }
        };
        
        vkCreateComputePipelines(ctx->device, ctx->pipeline_cache, 1,
            &pipeline_info, NULL, &gemv->pipelines[Q38_VK_PIPELINE_GEMV_IQ4_NL]);
        
        vkDestroyShaderModule(ctx->device, shader_module, NULL);
    }
    
    if (!gemv->pipelines[Q38_VK_PIPELINE_GEMV_IQ4_NL]) {
        fprintf(stderr, "Failed to create IQ4_NL pipeline\n");
        return 0;
    }
    
    // Allocate or resize buffers (use padded size for weights)
    if (gemv->weight_buffer.size < padded_weight_size) {
        q38_vulkan_buffer_destroy(ctx, &gemv->weight_buffer);
        if (!q38_vulkan_buffer_create(ctx, &gemv->weight_buffer, padded_weight_size,
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true)) {
            return 0;
        }
    }
    
    if (gemv->input_buffer.size < input_size) {
        q38_vulkan_buffer_destroy(ctx, &gemv->input_buffer);
        if (!q38_vulkan_buffer_create(ctx, &gemv->input_buffer, input_size,
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true)) {
            return 0;
        }
    }
    
    if (gemv->output_buffer.size < output_size) {
        q38_vulkan_buffer_destroy(ctx, &gemv->output_buffer);
        if (!q38_vulkan_buffer_create(ctx, &gemv->output_buffer, output_size,
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true)) {
            return 0;
        }
    }
    
    // Initialize persistent codebook buffer (once)
    if (!gemv->codebook_initialized) {
        const uint64_t codebook_size = 16 * sizeof(float);
        if (!q38_vulkan_buffer_create(ctx, &gemv->codebook_buffer, codebook_size,
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true)) {
            return 0;
        }
        q38_vulkan_buffer_write(ctx, &gemv->codebook_buffer, kIQ4NLCodebook, 0, codebook_size);
        gemv->codebook_initialized = 1;
    }
    
    // Pad weight data from 18 bytes to 20 bytes per block for optimal GPU access
    // This is critical for UMA systems where memory bandwidth is shared
    uint8_t *padded_weights = malloc(padded_weight_size);
    if (!padded_weights) {
        return 0;
    }
    
    const uint8_t *src = tensor->data;
    for (uint64_t i = 0; i < M * blocks_per_row; i++) {
        // Copy 18 bytes of block data
        memcpy(padded_weights + i * 20, src + i * 18, 18);
        // Zero out 2 padding bytes (not strictly necessary but cleaner)
        memset(padded_weights + i * 20 + 18, 0, 2);
    }
    
    // Upload padded data to GPU
    q38_vulkan_buffer_write(ctx, &gemv->weight_buffer, padded_weights, 0, padded_weight_size);
    free(padded_weights);  // Free temporary buffer after upload
    
    q38_vulkan_buffer_write(ctx, &gemv->input_buffer, input, 0, input_size);
    // Codebook is already uploaded during initialization
    
    // Allocate descriptor set
    VkDescriptorSetAllocateInfo desc_alloc = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = gemv->descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &gemv->descriptor_set_layouts[Q38_VK_PIPELINE_GEMV_IQ4_NL]
    };
    
    VkDescriptorSet descriptor_set;
    VkResult result = vkAllocateDescriptorSets(ctx->device, &desc_alloc, &descriptor_set);
    if (result != VK_SUCCESS) {
        return 0;  // No need to destroy codebook - it's persistent
    }
    
    VkDescriptorBufferInfo input_desc = {
        .buffer = gemv->input_buffer.buffer,
        .offset = 0,
        .range = input_size
    };
    VkDescriptorBufferInfo output_desc = {
        .buffer = gemv->output_buffer.buffer,
        .offset = 0,
        .range = output_size
    };
    VkDescriptorBufferInfo codebook_desc = {
        .buffer = gemv->codebook_buffer.buffer,  // Use persistent buffer
        .offset = 0,
        .range = 16 * sizeof(float)
    };
    
    // Update descriptor sets (use padded weight size)
    VkDescriptorBufferInfo weight_desc = {
        .buffer = gemv->weight_buffer.buffer,
        .offset = 0,
        .range = padded_weight_size
    };
    
    VkWriteDescriptorSet writes[] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &weight_desc
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &input_desc
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &output_desc
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 3,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &codebook_desc
        }
    };
    
    vkUpdateDescriptorSets(ctx->device, 4, writes, 0, NULL);
    
    // Create command buffer
    VkCommandBufferAllocateInfo cmd_alloc = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = ctx->command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };
    
    VkCommandBuffer cmd_buffer;
    vkAllocateCommandBuffers(ctx->device, &cmd_alloc, &cmd_buffer);
    
    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    
    vkBeginCommandBuffer(cmd_buffer, &begin_info);
    
    vkCmdBindPipeline(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      gemv->pipelines[Q38_VK_PIPELINE_GEMV_IQ4_NL]);
    vkCmdBindDescriptorSets(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           gemv->pipeline_layouts[Q38_VK_PIPELINE_GEMV_IQ4_NL],
                           0, 1, &descriptor_set, 0, NULL);
    
    // Push constants
    struct {
        uint32_t M;
        uint32_t N;
        uint32_t blocks_per_row;
    } push_constants = { (uint32_t)M, (uint32_t)N, (uint32_t)blocks_per_row };
    
    vkCmdPushConstants(cmd_buffer, gemv->pipeline_layouts[Q38_VK_PIPELINE_GEMV_IQ4_NL],
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants),
                       &push_constants);
    
    // Dispatch
    uint32_t workgroups = (M + 255) / 256;
    vkCmdDispatch(cmd_buffer, workgroups, 1, 1);
    
    // Memory barrier
    VkMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT
    };
    vkCmdPipelineBarrier(cmd_buffer,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT,
                         0, 1, &barrier, 0, NULL, 0, NULL);
    
    vkEndCommandBuffer(cmd_buffer);
    
    // Submit and wait
    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd_buffer
    };
    
    VkFence fence;
    VkFenceCreateInfo fence_info = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO
    };
    vkCreateFence(ctx->device, &fence_info, NULL, &fence);
    
    vkQueueSubmit(ctx->compute_queue, 1, &submit_info, fence);
    vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);
    
    // Read back results
    q38_vulkan_buffer_read(ctx, &gemv->output_buffer, output, 0, output_size);
    
    // Validate output
    bool output_has_nan = false;
    for (int i = 0; i < 10 && i < M; i++) {
        if (!isfinite(output[i])) {
            output_has_nan = true;
            break;
        }
    }
    
    if (output_has_nan) {
        memset(output, 0, output_size);
        // Don't destroy codebook - it's persistent
        vkDestroyFence(ctx->device, fence, NULL);
        vkFreeCommandBuffers(ctx->device, ctx->command_pool, 1, &cmd_buffer);
        vkFreeDescriptorSets(ctx->device, gemv->descriptor_pool, 1, &descriptor_set);
        return 0;
    }
    
    // Cleanup
    // Don't destroy codebook - it's persistent and reused
    vkDestroyFence(ctx->device, fence, NULL);
    vkFreeCommandBuffers(ctx->device, ctx->command_pool, 1, &cmd_buffer);
    vkFreeDescriptorSets(ctx->device, gemv->descriptor_pool, 1, &descriptor_set);
    
    return 1;
}

// === ASYNC OPERATIONS ===

int q38_vulkan_gemv_fetch_begin(
    const Q38VulkanContext *ctx,
    Q38VulkanGEMV *gemv,
    const Q38GGUFTensor *tensor,
    uint32_t layer_id,
    bool *ready) {
    
    if (!ctx || !gemv || !tensor || !ready) return 0;
    if (!gemv->use_cache) return 0;
    
    const uint64_t tensor_size = tensor->n_dims == 2 
        ? tensor->shape[0] * tensor->shape[1] * sizeof(float)
        : 0;
    
    Q38VulkanBuffer *buf = q38_weight_cache_fetch_begin(
        ctx, &gemv->weight_cache, tensor->data, tensor_size,
        tensor->type, layer_id, ready);
    
    return buf != NULL;
}

int q38_vulkan_gemv_fetch_end(
    const Q38VulkanContext *ctx,
    Q38VulkanGEMV *gemv,
    const Q38GGUFTensor *tensor) {
    
    if (!ctx || !gemv || !tensor) return 0;
    if (!gemv->use_cache) return 0;
    
    return q38_weight_cache_fetch_end(ctx, &gemv->weight_cache, tensor->data);
}

// === SPECULATIVE PREFETCH ===

void q38_vulkan_gemv_prefetch_next_layer(
    const Q38VulkanContext *ctx,
    Q38VulkanGEMV *gemv,
    uint32_t current_layer,
    const uint32_t *predicted_experts,
    uint32_t n_experts) {
    
    if (!ctx || !gemv || !predicted_experts || n_experts == 0) return;
    if (!gemv->use_cache) return;
    
    // Predict next layer's experts
    (void)current_layer;  // Will be used in full implementation
    uint32_t predicted[MAX_EXPERTS_PER_LAYER];
    
    q38_weight_cache_predict_next(&gemv->weight_cache, current_layer,
                                   predicted_experts, n_experts,
                                   predicted, n_experts);
    
    // Prefetch each predicted expert's weights
    // Note: In a real implementation, you'd need to map expert IDs to actual tensor data
    // This is a placeholder showing the pattern
    
    for (uint32_t i = 0; i < n_experts && i < MAX_EXPERTS_PER_LAYER; i++) {
        // TODO: Get actual tensor data pointer for this expert
        // const void *expert_data = get_expert_weights(next_layer, predicted[i]);
        // q38_weight_cache_prefetch(ctx, &gemv->weight_cache, expert_data, ...);
        (void)predicted[i];  // Suppress unused warning
    }
}

void q38_vulkan_gemv_update_routing(
    Q38VulkanGEMV *gemv,
    uint32_t layer,
    const uint32_t *experts,
    uint32_t n_experts) {
    
    if (!gemv || !experts || n_experts == 0) return;
    if (!gemv->use_cache) return;
    
    q38_weight_cache_update_prediction(&gemv->weight_cache, layer, experts, n_experts);
}

// ============================================================================
// Q3_K GEMV Implementation
// ============================================================================

int q38_vulkan_gemv_q3_k(
    const Q38VulkanContext *ctx,
    Q38VulkanGEMV *gemv,
    float *output,
    const float *input,
    const Q38GGUFTensor *tensor) {
    
    if (!ctx || !gemv || !output || !input || !tensor) return 0;
    if (tensor->n_dims != 2) return 0;
    if (tensor->type != 11) return 0;  // Q38_GGML_Q3_K
    
    const uint64_t N = tensor->shape[0];  // Input dimension
    const uint64_t M = tensor->shape[1];  // Output dimension (rows)
    
    // Q3_K requires N to be multiple of 256
    if (N % 256 != 0) return 0;
    
    // Skip very small matrices - not worth GPU overhead
    if (M < 4 || N < 256) return 0;
    
    // Validate input data
    for (int i = 0; i < 10 && i < N; i++) {
        if (!isfinite(input[i])) return 0;
    }
    
    // === CREATE PIPELINE IF NEEDED ===
    if (!gemv->pipelines[Q38_VK_PIPELINE_GEMV_Q3_K]) {
        size_t shader_size;
        uint32_t *shader_code = load_spirv_shader("shaders/spv/gemv_q3_k.spv", &shader_size);
        if (!shader_code) {
            fprintf(stderr, "Failed to load Q3_K shader\n");
            return 0;
        }
        
        VkShaderModuleCreateInfo shader_info = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = shader_size,
            .pCode = shader_code
        };
        
        VkShaderModule shader_module;
        VkResult result = vkCreateShaderModule(ctx->device, &shader_info, NULL, &shader_module);
        free(shader_code);
        
        if (result != VK_SUCCESS) {
            fprintf(stderr, "Failed to create Q3_K shader module\n");
            return 0;
        }
        
        // Create descriptor set layout (same as F32: weights, input, output)
        VkDescriptorSetLayoutBinding bindings[] = {
            {  // Weights buffer (binding 0)
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
            },
            {  // Input buffer (binding 1)
                .binding = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
            },
            {  // Output buffer (binding 2)
                .binding = 2,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
            }
        };
        
        VkDescriptorSetLayoutCreateInfo layout_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 3,
            .pBindings = bindings
        };
        
        vkCreateDescriptorSetLayout(ctx->device, &layout_info, NULL,
            &gemv->descriptor_set_layouts[Q38_VK_PIPELINE_GEMV_Q3_K]);
        
        // Create pipeline layout with push constants
        typedef struct {
            uint32_t M;
            uint32_t N;
        } PushConstants;
        
        VkPushConstantRange push_range = {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(PushConstants)
        };
        
        VkPipelineLayoutCreateInfo pipeline_layout_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &gemv->descriptor_set_layouts[Q38_VK_PIPELINE_GEMV_Q3_K],
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_range
        };
        
        vkCreatePipelineLayout(ctx->device, &pipeline_layout_info, NULL,
            &gemv->pipeline_layouts[Q38_VK_PIPELINE_GEMV_Q3_K]);
        
        // Create compute pipeline
        VkComputePipelineCreateInfo pipeline_info = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .layout = gemv->pipeline_layouts[Q38_VK_PIPELINE_GEMV_Q3_K],
            .stage = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = shader_module,
                .pName = "main"
            }
        };
        
        vkCreateComputePipelines(ctx->device, ctx->pipeline_cache, 1,
            &pipeline_info, NULL, &gemv->pipelines[Q38_VK_PIPELINE_GEMV_Q3_K]);
        
        vkDestroyShaderModule(ctx->device, shader_module, NULL);
        
        if (!gemv->pipelines[Q38_VK_PIPELINE_GEMV_Q3_K]) {
            fprintf(stderr, "Failed to create Q3_K pipeline\n");
            return 0;
        }
    }
    
    // === PREPARE WEIGHTS WITH PADDING ===
    // Q3_K block size is 110 bytes, pad to 112 for alignment
    const uint64_t blocks_per_row = N / 256;
    const uint64_t padded_block_size = 112;
    const uint64_t weight_size = M * blocks_per_row * padded_block_size;
    const uint64_t input_size = N * sizeof(float);
    const uint64_t output_size = M * sizeof(float);
    
    // Check cache first
    Q38VulkanBuffer *weight_buf = NULL;
    bool using_cache = false;
    
    if (gemv->use_cache) {
        weight_buf = q38_weight_cache_get_ext(ctx, &gemv->weight_cache,
            tensor->data, weight_size, Q38_GGML_Q3_K, 0, (uint32_t)M, (uint32_t)N);
        if (weight_buf && weight_buf->size >= weight_size) {
            using_cache = true;
        }
    }
    
    if (!using_cache) {
        // Create weight buffer and upload padded data
        if (!q38_vulkan_buffer_create(ctx, &gemv->weight_buffer, weight_size,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true)) {
            return 0;
        }
        weight_buf = &gemv->weight_buffer;
        
        // Pad and upload weight data
        uint8_t *padded_weights = malloc(weight_size);
        if (!padded_weights) return 0;
        
        const uint8_t *src = tensor->data;
        for (uint64_t i = 0; i < M * blocks_per_row; i++) {
            memcpy(padded_weights + i * padded_block_size, src + i * 110, 110);
            memset(padded_weights + i * padded_block_size + 110, 0, 2);
        }
        
        q38_vulkan_buffer_write(ctx, weight_buf, padded_weights, 0, weight_size);
        free(padded_weights);
    }
    
    // === CREATE INPUT/OUTPUT BUFFERS ===
    if (!q38_vulkan_buffer_create(ctx, &gemv->input_buffer, input_size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true)) {
        return 0;
    }
    
    q38_vulkan_buffer_write(ctx, &gemv->input_buffer, input, 0, input_size);
    
    if (!q38_vulkan_buffer_create(ctx, &gemv->output_buffer, output_size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true)) {
        return 0;
    }
    
    // === ALLOCATE DESCRIPTOR SET ===
    VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = gemv->descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &gemv->descriptor_set_layouts[Q38_VK_PIPELINE_GEMV_Q3_K]
    };
    
    VkDescriptorSet desc_set;
    if (vkAllocateDescriptorSets(ctx->device, &alloc_info, &desc_set) != VK_SUCCESS) {
        return 0;
    }
    
    // === UPDATE DESCRIPTOR SET ===
    VkDescriptorBufferInfo buffer_infos[3] = {
        { .buffer = weight_buf->buffer, .offset = 0, .range = weight_size },
        { .buffer = gemv->input_buffer.buffer, .offset = 0, .range = input_size },
        { .buffer = gemv->output_buffer.buffer, .offset = 0, .range = output_size }
    };
    
    VkWriteDescriptorSet writes[3] = {0};
    for (int i = 0; i < 3; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = desc_set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buffer_infos[i];
    }
    
    vkUpdateDescriptorSets(ctx->device, 3, writes, 0, NULL);
    
    // === EXECUTE COMPUTE SHADER ===
    VkCommandBufferAllocateInfo cmd_alloc = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = ctx->command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };
    
    VkCommandBuffer cmd_buffer;
    vkAllocateCommandBuffers(ctx->device, &cmd_alloc, &cmd_buffer);
    
    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    
    vkBeginCommandBuffer(cmd_buffer, &begin_info);
    
    vkCmdBindPipeline(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, 
        gemv->pipelines[Q38_VK_PIPELINE_GEMV_Q3_K]);
    vkCmdBindDescriptorSets(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
        gemv->pipeline_layouts[Q38_VK_PIPELINE_GEMV_Q3_K], 0, 1, &desc_set, 0, NULL);
    
    // Push constants: M, N
    struct {
        uint32_t M;
        uint32_t N;
    } push_constants = { (uint32_t)M, (uint32_t)N };
    
    vkCmdPushConstants(cmd_buffer, gemv->pipeline_layouts[Q38_VK_PIPELINE_GEMV_Q3_K],
        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);
    
    // Dispatch: one workgroup per row
    vkCmdDispatch(cmd_buffer, (uint32_t)M, 1, 1);
    
    // Add memory barrier
    VkMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT
    };
    vkCmdPipelineBarrier(cmd_buffer,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT,
                         0, 1, &barrier, 0, NULL, 0, NULL);
    
    vkEndCommandBuffer(cmd_buffer);
    
    // Submit and wait
    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd_buffer
    };
    
    VkFence fence;
    VkFenceCreateInfo fence_info = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO
    };
    vkCreateFence(ctx->device, &fence_info, NULL, &fence);
    
    vkQueueSubmit(ctx->compute_queue, 1, &submit_info, fence);
    vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);
    
    // === READ BACK RESULTS ===
    q38_vulkan_buffer_read(ctx, &gemv->output_buffer, output, 0, output_size);
    
    // Validate output
    for (int i = 0; i < 10 && i < (int)M; i++) {
        if (!isfinite(output[i])) {
            memset(output, 0, output_size);
            vkDestroyFence(ctx->device, fence, NULL);
            vkFreeCommandBuffers(ctx->device, ctx->command_pool, 1, &cmd_buffer);
            vkFreeDescriptorSets(ctx->device, gemv->descriptor_pool, 1, &desc_set);
            return 0;
        }
    }
    
    // Cleanup
    vkDestroyFence(ctx->device, fence, NULL);
    vkFreeCommandBuffers(ctx->device, ctx->command_pool, 1, &cmd_buffer);
    vkFreeDescriptorSets(ctx->device, gemv->descriptor_pool, 1, &desc_set);
    
    return 1;
}

