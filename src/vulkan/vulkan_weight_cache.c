#include "vulkan/vulkan_weight_cache.h"
#include "qwen38/qwen38_gguf.h"  // For Q38_GGML_IQ4_NL
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

// === INTERNAL HELPERS ===

static uint64_t get_timestamp_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

// Simple xorshift PRNG for sampling
static uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

// Find entry index by tensor data pointer
static int find_entry_by_data(const Q38WeightCache *cache, const void *tensor_data, uint64_t tensor_size) {
    for (uint32_t i = 0; i < cache->n_entries; i++) {
        if (cache->entries[i].valid && 
            cache->entries[i].tensor_data == tensor_data &&
            cache->entries[i].tensor_size == tensor_size) {
            return (int)i;
        }
    }
    return -1;
}

// Find an empty slot in the cache
static int find_empty_slot(const Q38WeightCache *cache) {
    for (uint32_t i = 0; i < cache->max_entries; i++) {
        if (!cache->entries[i].valid) {
            return (int)i;
        }
    }
    return -1;
}

// Sample-based eviction victim selection
static int find_eviction_victim(const Q38WeightCache *cache) {
    if (cache->n_entries == 0) return -1;
    
    uint32_t rng_state = (uint32_t)(cache->timestamp & 0xFFFFFFFF);
    int victim = -1;
    uint64_t worst_score = UINT64_MAX;
    
    // Sample N entries instead of scanning all
    const uint32_t samples = cache->config.evict_samples;
    
    for (uint32_t i = 0; i < samples; i++) {
        uint32_t idx = xorshift32(&rng_state) % cache->max_entries;
        Q38WeightCacheEntry *entry = &cache->entries[idx];
        
        if (!entry->valid) continue;
        if (entry->transfer_in_flight) continue;  // Don't evict during transfer
        
        uint64_t score;
        switch (cache->config.eviction_policy) {
            case Q38_EVICT_LRU:
                score = entry->last_used;  // Lower = older = better victim
                break;
            case Q38_EVICT_LFU:
                score = entry->freq;  // Lower = less used = better victim
                break;
            case Q38_EVICT_HYBRID:
            default:
                // Combine recency and frequency
                // Higher timestamp diff + lower freq = better victim
                score = (cache->timestamp - entry->last_used) / (entry->freq + 1);
                break;
        }
        
        if (score < worst_score) {
            worst_score = score;
            victim = (int)idx;
        }
    }
    
    return victim;
}

// Create padded weight data for IQ4_NL (18 -> 20 bytes per block)
static uint8_t* create_padded_weights(const void *tensor_data, uint64_t tensor_size,
                                       int tensor_type, uint64_t *padded_size) {
    if (tensor_type != Q38_GGML_IQ4_NL) {
        // No padding needed for other types
        *padded_size = tensor_size;
        return NULL;  // Caller should use original data
    }
    
    // Calculate number of blocks (each block is 32 weights = 18 bytes originally)
    const uint64_t blocks = tensor_size / 18;
    *padded_size = blocks * 20;  // Padded to 20 bytes per block
    
    uint8_t *padded = malloc(*padded_size);
    if (!padded) return NULL;
    
    const uint8_t *src = (const uint8_t *)tensor_data;
    uint8_t *dst = padded;
    
    for (uint64_t i = 0; i < blocks; i++) {
        memcpy(dst, src, 18);     // Copy original 18 bytes
        memset(dst + 18, 0, 2);   // Zero-fill 2 bytes padding
        src += 18;
        dst += 20;
    }
    
    return padded;
}

// === INITIALIZATION ===

int q38_weight_cache_init(
    const Q38VulkanContext *ctx,
    Q38WeightCache *cache,
    const Q38WeightCacheConfig *config) {
    
    if (!ctx || !cache) return 0;
    
    memset(cache, 0, sizeof(Q38WeightCache));
    cache->ctx = ctx;
    
    // Apply configuration
    if (config) {
        cache->config = *config;
    } else {
        // Defaults
        cache->config.max_entries = 256;
        cache->config.eviction_policy = Q38_EVICT_HYBRID;
        cache->config.evict_samples = 16;
        cache->config.age_every = 65536;
        cache->config.max_promotions_per_layer = 2;
        cache->config.enable_prefetch = true;
        cache->config.enable_async = true;
        cache->config.debug_stats = true;
    }
    
    // Allocate entries
    cache->max_entries = cache->config.max_entries;
    cache->entries = calloc(cache->max_entries, sizeof(Q38WeightCacheEntry));
    if (!cache->entries) {
        fprintf(stderr, "[weight_cache] Failed to allocate %u entries\n", cache->max_entries);
        return 0;
    }
    
    cache->timestamp = 1;  // Start at 1 so 0 means "never used"
    
    fprintf(stderr, "[weight_cache] Initialized with %u entries, policy=%d\n",
            cache->max_entries, cache->config.eviction_policy);
    
    return 1;
}

void q38_weight_cache_cleanup(
    const Q38VulkanContext *ctx,
    Q38WeightCache *cache) {
    
    if (!cache) return;
    
    // Print final statistics
    if (cache->config.debug_stats) {
        q38_weight_cache_print_stats(cache);
    }
    
    // Free all buffers
    for (uint32_t i = 0; i < cache->max_entries; i++) {
        Q38WeightCacheEntry *entry = &cache->entries[i];
        if (entry->valid) {
            if (entry->buffer.buffer != VK_NULL_HANDLE) {
                q38_vulkan_buffer_destroy(ctx, &entry->buffer);
            }
            if (entry->transfer_fence != VK_NULL_HANDLE) {
                vkDestroyFence(ctx->device, entry->transfer_fence, NULL);
            }
        }
    }
    
    free(cache->entries);
    memset(cache, 0, sizeof(Q38WeightCache));
}

// === CORE CACHE OPERATIONS ===

Q38VulkanBuffer* q38_weight_cache_get(
    const Q38VulkanContext *ctx,
    Q38WeightCache *cache,
    const void *tensor_data,
    uint64_t tensor_size,
    int tensor_type,
    uint32_t layer_id) {
    
    // Call extended version with dimensions set to 0 (skip validation)
    return q38_weight_cache_get_ext(ctx, cache, tensor_data, tensor_size, tensor_type, layer_id, 0, 0);
}

Q38VulkanBuffer* q38_weight_cache_get_ext(
    const Q38VulkanContext *ctx,
    Q38WeightCache *cache,
    const void *tensor_data,
    uint64_t tensor_size,
    int tensor_type,
    uint32_t layer_id,
    uint32_t rows,
    uint32_t cols) {
    
    if (!ctx || !cache || !tensor_data) return NULL;
    
    cache->stats.lookups++;
    
    // Age counters periodically
    if (cache->config.age_every > 0 && 
        (cache->stats.lookups % cache->config.age_every) == 0) {
        q38_weight_cache_age(cache);
    }
    
    // Check if already cached
    int idx = find_entry_by_data(cache, tensor_data, tensor_size);
    if (idx >= 0) {
        // Cache HIT - validate dimensions if provided
        Q38WeightCacheEntry *entry = &cache->entries[idx];
        
        // If dimensions are provided, verify they match
        if (rows > 0 && cols > 0 && (entry->rows != rows || entry->cols != cols)) {
            // Dimension mismatch - invalidate this entry
            fprintf(stderr, "DEBUG: Cache dimension mismatch (cached: %ux%u, requested: %ux%u), invalidating\n",
                    entry->rows, entry->cols, rows, cols);
            q38_weight_cache_invalidate(ctx, cache, tensor_data);
            // Fall through to create new entry
        } else {
            entry->last_used = ++cache->timestamp;
            entry->freq++;
            cache->stats.hits++;
            
            // Wait for any pending async transfer
            if (entry->transfer_in_flight && entry->transfer_fence != VK_NULL_HANDLE) {
                vkWaitForFences(ctx->device, 1, &entry->transfer_fence, VK_TRUE, UINT64_MAX);
                entry->transfer_in_flight = false;
            }
            
            return &entry->buffer;
        }
    }
    
    // Cache MISS
    cache->stats.misses++;
    
    // Find a slot (empty or evict)
    int slot = find_empty_slot(cache);
    if (slot < 0) {
        slot = find_eviction_victim(cache);
        if (slot < 0) {
            fprintf(stderr, "[weight_cache] No eviction victim found\n");
            return NULL;
        }
        
        // Evict old entry
        Q38WeightCacheEntry *old = &cache->entries[slot];
        if (old->buffer.buffer != VK_NULL_HANDLE) {
            q38_vulkan_buffer_destroy(ctx, &old->buffer);
        }
        if (old->transfer_fence != VK_NULL_HANDLE) {
            vkDestroyFence(ctx->device, old->transfer_fence, NULL);
        }
        cache->stats.evictions++;
        
        // Mark as empty
        memset(old, 0, sizeof(Q38WeightCacheEntry));
    }
    
    // Create new entry
    Q38WeightCacheEntry *entry = &cache->entries[slot];
    entry->tensor_data = tensor_data;
    entry->tensor_size = tensor_size;
    entry->tensor_type = tensor_type;
    entry->layer_id = layer_id;
    entry->rows = rows;  // Store dimensions for validation
    entry->cols = cols;
    entry->last_used = ++cache->timestamp;
    entry->freq = 1;
    entry->tier = Q38_CACHE_TIER_VRAM;
    entry->on_gpu = true;
    entry->late = false;
    entry->transfer_in_flight = false;
    
    // Prepare weight data (with padding for IQ4_NL)
    uint64_t padded_size;
    uint8_t *padded_data = create_padded_weights(tensor_data, tensor_size, tensor_type, &padded_size);
    const void *data_to_upload = padded_data ? padded_data : tensor_data;
    entry->padded_size = padded_size;
    
    // Create GPU buffer
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (!q38_vulkan_buffer_create(ctx, &entry->buffer, padded_size, usage, true)) {
        fprintf(stderr, "[weight_cache] Failed to create buffer (%lu bytes)\n", padded_size);
        if (padded_data) free(padded_data);
        return NULL;
    }
    
    // Upload weights synchronously (blocking)
    double start_time = (double)get_timestamp_ns() / 1000.0;
    if (!q38_vulkan_buffer_write(ctx, &entry->buffer, data_to_upload, 0, padded_size)) {
        fprintf(stderr, "[weight_cache] Failed to write buffer\n");
        q38_vulkan_buffer_destroy(ctx, &entry->buffer);
        if (padded_data) free(padded_data);
        return NULL;
    }
    double end_time = (double)get_timestamp_ns() / 1000.0;
    
    cache->stats.bytes_transferred += padded_size;
    cache->stats.total_transfer_time_us += (end_time - start_time);
    
    entry->valid = true;
    if (cache->n_entries <= (uint32_t)slot) {
        cache->n_entries = (uint32_t)slot + 1;
    }
    
    if (padded_data) free(padded_data);
    
    return &entry->buffer;
}

// === ASYNC OPERATIONS ===

Q38VulkanBuffer* q38_weight_cache_fetch_begin(
    const Q38VulkanContext *ctx,
    Q38WeightCache *cache,
    const void *tensor_data,
    uint64_t tensor_size,
    int tensor_type,
    uint32_t layer_id,
    bool *ready) {
    
    if (!ctx || !cache || !tensor_data || !ready) return NULL;
    *ready = false;
    
    cache->stats.lookups++;
    
    // Check if already cached
    int idx = find_entry_by_data(cache, tensor_data, tensor_size);
    if (idx >= 0) {
        Q38WeightCacheEntry *entry = &cache->entries[idx];
        entry->last_used = ++cache->timestamp;
        entry->freq++;
        cache->stats.hits++;
        
        // Check if transfer is still in flight
        if (entry->transfer_in_flight && entry->transfer_fence != VK_NULL_HANDLE) {
            VkResult result = vkGetFenceStatus(ctx->device, entry->transfer_fence);
            if (result == VK_SUCCESS) {
                // Transfer complete
                entry->transfer_in_flight = false;
                *ready = true;
            }
            // If not ready, return buffer anyway (caller can poll or wait)
        } else {
            *ready = true;
        }
        
        return &entry->buffer;
    }
    
    // Not in cache - need to upload
    if (!cache->config.enable_async) {
        // Fall back to synchronous get
        return q38_weight_cache_get(ctx, cache, tensor_data, tensor_size, tensor_type, layer_id);
    }
    
    cache->stats.misses++;
    
    // Find slot
    int slot = find_empty_slot(cache);
    if (slot < 0) {
        slot = find_eviction_victim(cache);
        if (slot < 0) return NULL;
        
        // Evict
        Q38WeightCacheEntry *old = &cache->entries[slot];
        if (old->buffer.buffer != VK_NULL_HANDLE) {
            q38_vulkan_buffer_destroy(ctx, &old->buffer);
        }
        if (old->transfer_fence != VK_NULL_HANDLE) {
            vkDestroyFence(ctx->device, old->transfer_fence, NULL);
        }
        cache->stats.evictions++;
        memset(old, 0, sizeof(Q38WeightCacheEntry));
    }
    
    // Create entry
    Q38WeightCacheEntry *entry = &cache->entries[slot];
    entry->tensor_data = tensor_data;
    entry->tensor_size = tensor_size;
    entry->tensor_type = tensor_type;
    entry->layer_id = layer_id;
    entry->last_used = ++cache->timestamp;
    entry->freq = 1;
    entry->tier = Q38_CACHE_TIER_VRAM;
    entry->on_gpu = true;
    entry->late = true;  // Promoted during this fetch
    
    // Prepare data
    uint64_t padded_size;
    uint8_t *padded_data = create_padded_weights(tensor_data, tensor_size, tensor_type, &padded_size);
    const void *data_to_upload = padded_data ? padded_data : tensor_data;
    entry->padded_size = padded_size;
    
    // Create buffer
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (!q38_vulkan_buffer_create(ctx, &entry->buffer, padded_size, usage, true)) {
        if (padded_data) free(padded_data);
        return NULL;
    }
    
    // Create fence for async transfer
    VkFenceCreateInfo fence_info = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO
    };
    vkCreateFence(ctx->device, &fence_info, NULL, &entry->transfer_fence);
    
    // TODO: Implement actual async transfer using command buffer
    // For now, do synchronous write but mark as async for API compatibility
    if (!q38_vulkan_buffer_write(ctx, &entry->buffer, data_to_upload, 0, padded_size)) {
        q38_vulkan_buffer_destroy(ctx, &entry->buffer);
        if (padded_data) free(padded_data);
        return NULL;
    }
    
    entry->transfer_in_flight = true;  // Would be true in real async impl
    entry->valid = true;
    cache->n_pending_transfers++;
    
    if (cache->n_entries <= (uint32_t)slot) {
        cache->n_entries = (uint32_t)slot + 1;
    }
    
    if (padded_data) free(padded_data);
    
    *ready = false;  // Transfer in progress
    return &entry->buffer;
}

int q38_weight_cache_fetch_end(
    const Q38VulkanContext *ctx,
    Q38WeightCache *cache,
    const void *tensor_data) {
    
    if (!ctx || !cache || !tensor_data) return 0;
    
    int idx = find_entry_by_data(cache, tensor_data, 0);  // Size not needed for invalidate
    if (idx < 0) return 0;
    
    Q38WeightCacheEntry *entry = &cache->entries[idx];
    
    if (entry->transfer_in_flight && entry->transfer_fence != VK_NULL_HANDLE) {
        double start_time = (double)get_timestamp_ns() / 1000.0;
        vkWaitForFences(ctx->device, 1, &entry->transfer_fence, VK_TRUE, UINT64_MAX);
        double end_time = (double)get_timestamp_ns() / 1000.0;
        
        entry->transfer_in_flight = false;
        cache->n_pending_transfers--;
        cache->stats.total_wait_time_us += (end_time - start_time);
    }
    
    return 1;
}

void q38_weight_cache_settle(
    const Q38VulkanContext *ctx,
    Q38WeightCache *cache) {
    
    if (!ctx || !cache) return;
    
    // Wait for all pending transfers
    for (uint32_t i = 0; i < cache->n_entries; i++) {
        Q38WeightCacheEntry *entry = &cache->entries[i];
        if (entry->valid && entry->transfer_in_flight && entry->transfer_fence != VK_NULL_HANDLE) {
            vkWaitForFences(ctx->device, 1, &entry->transfer_fence, VK_TRUE, UINT64_MAX);
            entry->transfer_in_flight = false;
        }
    }
    
    cache->n_pending_transfers = 0;
}

// === SPECULATIVE PREFETCH ===

void q38_weight_cache_prefetch(
    const Q38VulkanContext *ctx,
    Q38WeightCache *cache,
    const void *tensor_data,
    uint64_t tensor_size,
    int tensor_type,
    uint32_t layer_id) {
    
    if (!ctx || !cache || !tensor_data) return;
    if (!cache->config.enable_prefetch) return;
    
    // Check if already cached
    if (find_entry_by_data(cache, tensor_data, tensor_size) >= 0) {
        return;  // Already present, nothing to do
    }
    
    cache->stats.prefetch_issued++;
    
    // Use async fetch for prefetch
    bool ready;
    q38_weight_cache_fetch_begin(ctx, cache, tensor_data, tensor_size, tensor_type, layer_id, &ready);
}

void q38_weight_cache_predict_next(
    Q38WeightCache *cache,
    uint32_t current_layer,
    const uint32_t *current_experts,
    uint32_t n_experts,
    uint32_t *predicted_experts,
    uint32_t n_predict) {
    
    if (!cache || !current_experts || !predicted_experts || n_predict == 0) return;
    
    // Simple prediction: look up what experts historically followed the current ones
    Q38PredictionModel *model = &cache->prediction;
    
    // Count occurrences of successors
    uint32_t successor_counts[MAX_EXPERTS_PER_LAYER] = {0};
    
    for (uint32_t i = 0; i < n_experts; i++) {
        uint32_t expert = current_experts[i];
        if (expert < model->n_patterns) {
            Q38ExpertPattern *pattern = &model->patterns[expert];
            for (uint32_t j = 0; j < MAX_EXPERTS_PER_LAYER; j++) {
                successor_counts[j] += pattern->successor_count[j];
            }
        }
    }
    
    // Select top-N most common successors
    for (uint32_t i = 0; i < n_predict; i++) {
        uint32_t best_expert = 0;
        uint32_t best_count = 0;
        
        for (uint32_t j = 0; j < MAX_EXPERTS_PER_LAYER; j++) {
            if (successor_counts[j] > best_count) {
                best_count = successor_counts[j];
                best_expert = j;
            }
        }
        
        predicted_experts[i] = best_expert;
        successor_counts[best_expert] = 0;  // Remove from consideration
    }
}

void q38_weight_cache_update_prediction(
    Q38WeightCache *cache,
    uint32_t layer,
    const uint32_t *experts,
    uint32_t n_experts) {
    
    if (!cache || !experts || n_experts == 0) return;
    
    Q38PredictionModel *model = &cache->prediction;
    
    // Update patterns: expert[i] -> expert[i+1] transitions
    for (uint32_t i = 0; i < n_experts - 1; i++) {
        uint32_t from = experts[i];
        uint32_t to = experts[i + 1];
        
        if (from < MAX_PREDICTION_HISTORY && to < MAX_EXPERTS_PER_LAYER) {
            model->patterns[from].successor_count[to]++;
            model->patterns[from].total_successors++;
            
            if (model->n_patterns <= from) {
                model->n_patterns = from + 1;
            }
        }
    }
}

// === STATISTICS ===

void q38_weight_cache_get_stats(
    const Q38WeightCache *cache,
    Q38WeightCacheStats *stats) {
    
    if (!cache || !stats) return;
    *stats = cache->stats;
}

void q38_weight_cache_print_stats(
    const Q38WeightCache *cache) {
    
    if (!cache) return;
    
    double hit_rate = cache->stats.lookups > 0 
        ? (double)cache->stats.hits / (double)cache->stats.lookups * 100.0 
        : 0.0;
    
    double avg_transfer_time = cache->stats.hits > 0
        ? cache->stats.total_transfer_time_us / (double)cache->stats.hits
        : 0.0;
    
    fprintf(stderr, "\n[weight_cache] Statistics:\n");
    fprintf(stderr, "  Lookups: %lu\n", cache->stats.lookups);
    fprintf(stderr, "  Hits: %lu (%.1f%%)\n", cache->stats.hits, hit_rate);
    fprintf(stderr, "  Misses: %lu\n", cache->stats.misses);
    fprintf(stderr, "  Evictions: %lu\n", cache->stats.evictions);
    fprintf(stderr, "  Bytes transferred: %.2f MB\n", cache->stats.bytes_transferred / 1e6);
    fprintf(stderr, "  Avg transfer time: %.2f µs\n", avg_transfer_time);
    fprintf(stderr, "  Total wait time: %.2f ms\n", cache->stats.total_wait_time_us / 1000.0);
    
    if (cache->config.enable_prefetch) {
        double prefetch_accuracy = cache->stats.prefetch_issued > 0
            ? (double)cache->stats.prefetch_used / (double)cache->stats.prefetch_issued * 100.0
            : 0.0;
        fprintf(stderr, "  Prefetch issued: %lu\n", cache->stats.prefetch_issued);
        fprintf(stderr, "  Prefetch used: %lu (%.1f%% accuracy)\n", 
                cache->stats.prefetch_used, prefetch_accuracy);
        fprintf(stderr, "  Prefetch wasted: %lu\n", cache->stats.prefetch_wasted);
    }
    
    double prediction_accuracy = cache->prediction.predictions_made > 0
        ? (double)cache->prediction.predictions_correct / (double)cache->prediction.predictions_made * 100.0
        : 0.0;
    fprintf(stderr, "  Prediction accuracy: %.1f%% (%u/%u)\n",
            prediction_accuracy, cache->prediction.predictions_correct, cache->prediction.predictions_made);
}

void q38_weight_cache_reset_stats(
    Q38WeightCache *cache) {
    
    if (!cache) return;
    memset(&cache->stats, 0, sizeof(Q38WeightCacheStats));
}

// === UTILITY ===

bool q38_weight_cache_contains(
    const Q38WeightCache *cache,
    const void *tensor_data) {
    
    return cache && tensor_data && find_entry_by_data(cache, tensor_data, 0) >= 0;
}

void q38_weight_cache_invalidate(
    const Q38VulkanContext *ctx,
    Q38WeightCache *cache,
    const void *tensor_data) {
    
    if (!ctx || !cache || !tensor_data) return;
    
    int idx = find_entry_by_data(cache, tensor_data, 0);  // Size not needed for invalidate
    if (idx < 0) return;
    
    Q38WeightCacheEntry *entry = &cache->entries[idx];
    
    // Wait for any pending transfer
    if (entry->transfer_in_flight && entry->transfer_fence != VK_NULL_HANDLE) {
        vkWaitForFences(ctx->device, 1, &entry->transfer_fence, VK_TRUE, UINT64_MAX);
    }
    
    // Destroy buffer
    if (entry->buffer.buffer != VK_NULL_HANDLE) {
        q38_vulkan_buffer_destroy(ctx, &entry->buffer);
    }
    if (entry->transfer_fence != VK_NULL_HANDLE) {
        vkDestroyFence(ctx->device, entry->transfer_fence, NULL);
    }
    
    memset(entry, 0, sizeof(Q38WeightCacheEntry));
}

void q38_weight_cache_clear(
    const Q38VulkanContext *ctx,
    Q38WeightCache *cache) {
    
    if (!ctx || !cache) return;
    
    q38_weight_cache_settle(ctx, cache);
    
    for (uint32_t i = 0; i < cache->max_entries; i++) {
        Q38WeightCacheEntry *entry = &cache->entries[i];
        if (entry->valid) {
            if (entry->buffer.buffer != VK_NULL_HANDLE) {
                q38_vulkan_buffer_destroy(ctx, &entry->buffer);
            }
            if (entry->transfer_fence != VK_NULL_HANDLE) {
                vkDestroyFence(ctx->device, entry->transfer_fence, NULL);
            }
        }
    }
    
    memset(cache->entries, 0, cache->max_entries * sizeof(Q38WeightCacheEntry));
    cache->n_entries = 0;
}

void q38_weight_cache_age(
    Q38WeightCache *cache) {
    
    if (!cache) return;
    
    // Halve all frequency counters to prevent early-hot items from pinning slots forever
    for (uint32_t i = 0; i < cache->n_entries; i++) {
        cache->entries[i].freq /= 2;
    }
}
