#include "vulkan/vulkan_expert_cache.h"
#include "qwen38/qwen38_gguf.h"  // For Q38GGUFTensor
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>

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

// Find eviction victim using sampling
static int find_victim_slot(Q38ExpertCache *cache, uint32_t layer, bool in_vram) {
    Q38ExpertLayerInfo *info = &cache->layers[layer];
    if (info->n_slots == 0) return -1;
    
    uint32_t rng_state = (uint32_t)(cache->timestamp & 0xFFFFFFFF);
    int victim = -1;
    uint64_t worst_score = UINT64_MAX;
    
    uint32_t n_slots = in_vram ? info->n_vram_slots : info->n_ram_slots;
    uint32_t samples = cache->config.evict_samples;
    if (samples > n_slots) samples = n_slots;
    
    for (uint32_t i = 0; i < samples; i++) {
        uint32_t idx = xorshift32(&rng_state) % n_slots;
        
        if (!info->slot_valid[idx]) continue;
        
        // Hybrid LRU/LFU score
        uint64_t score = (cache->timestamp - info->slot_used[idx]) / (info->slot_freq[idx] + 1);
        
        if (score < worst_score) {
            worst_score = score;
            victim = (int)idx;
        }
    }
    
    return victim;
}

// === INITIALIZATION ===

int q38_expert_cache_init_from_tensors(
    const Q38VulkanContext *ctx,
    Q38ExpertCache *cache,
    uint32_t n_layers,
    uint32_t n_experts,
    uint32_t n_embd,
    uint32_t n_ff,
    const Q38GGUFTensor **gate_tensors,
    const Q38GGUFTensor **up_tensors,
    const Q38GGUFTensor **down_tensors,
    const Q38ExpertCacheConfig *config) {
    
    if (!ctx || !cache || !gate_tensors || !up_tensors || !down_tensors) return 0;
    memset(cache, 0, sizeof(*cache));
    
    cache->ctx = ctx;
    cache->n_layers = n_layers;
    cache->n_experts = n_experts;
    cache->n_embd = n_embd;
    cache->n_ff = n_ff;
    
    // Set configuration
    if (config) {
        cache->config = *config;
    } else {
        // Default configuration for AMD 780M
        cache->config.vram_budget = 2ull * 1024 * 1024 * 1024;  // 2 GB (conservative for iGPU)
        cache->config.ram_budget = 8ull * 1024 * 1024 * 1024;  // 8 GB
        cache->config.max_promotions_per_layer = 2;
        cache->config.evict_samples = 16;
        cache->config.age_every = 65536;
        cache->config.enable_prefetch = true;
        cache->config.debug_stats = true;
    }
    
    fprintf(stderr, "[expert_cache] Initializing three-tier cache\n");
    fprintf(stderr, "[expert_cache] Model: n_embd=%u, n_ff=%u, n_experts=%u, n_layers=%u\n",
            cache->n_embd, cache->n_ff, cache->n_experts, cache->n_layers);
    
    // Calculate expert block size from actual tensors
    // Each expert is a slice of the 3D tensor [rows, cols, n_experts]
    const Q38GGUFTensor *gate0 = gate_tensors[0];
    if (!gate0 || gate0->n_dims != 3) {
        fprintf(stderr, "[expert_cache] Invalid gate tensor dimensions\n");
        return 0;
    }
    
    // Size per expert (assuming all layers have same structure)
    uint64_t expert_bytes = gate0->nbytes / n_experts;
    fprintf(stderr, "[expert_cache] Expert size: ~%lu MB per expert part\n", 
            expert_bytes / (1024 * 1024));
    
    // Total expert size = gate + up + down
    uint64_t total_expert_bytes = expert_bytes * 3;
    fprintf(stderr, "[expert_cache] Total expert size: ~%lu MB\n", 
            total_expert_bytes / (1024 * 1024));
    
    // Store tensor references for later use
    // We'll access these directly instead of copying
    cache->ram_base = NULL;  // Not using separate RAM buffer - using mmap'd data
    cache->ram_capacity = 0;
    cache->ram_slots_total = 0;
    
    // Allocate T0: VRAM tier for hot experts
    // For now, skip VRAM tier and just use the existing memory-mapped data
    // This simplifies initial integration
    cache->vram_capacity = 0;
    cache->vram_slots_total = 0;
    
    fprintf(stderr, "[expert_cache] Using direct memory-mapped access (no VRAM tier yet)\n");
    
    // Initialize per-layer structures
    for (uint32_t layer = 0; layer < cache->n_layers; layer++) {
        Q38ExpertLayerInfo *info = &cache->layers[layer];
        
        info->block_bytes = total_expert_bytes;
        info->part_bytes[0] = expert_bytes;  // gate
        info->part_bytes[1] = expert_bytes;  // up
        info->part_bytes[2] = expert_bytes;  // down
        
        info->part_offset[0] = 0;
        info->part_offset[1] = expert_bytes;
        info->part_offset[2] = expert_bytes * 2;
        
        // No slots - we're not caching yet, just providing direct access
        info->n_vram_slots = 0;
        info->n_ram_slots = 0;
        info->n_slots = 0;
        
        // Allocate minimal tracking arrays
        info->slot_expert = NULL;
        info->slot_valid = NULL;
        info->slot_used = NULL;
        info->slot_freq = NULL;
        info->expert_slot = (int32_t*)malloc(n_experts * sizeof(int32_t));
        
        if (!info->expert_slot) {
            fprintf(stderr, "[expert_cache] Failed to allocate layer %u structures\n", layer);
            q38_expert_cache_shutdown(ctx, cache);
            return 0;
        }
        
        // Mark all experts as available (in mapped memory)
        for (uint32_t i = 0; i < n_experts; i++) {
            info->expert_slot[i] = 0;  // 0 means "directly accessible in mapped memory"
        }
    }
    
    // Store tensor pointers for direct access
    // We'll add these to the cache structure
    
    fprintf(stderr, "[expert_cache] Initialization complete\n");
    fprintf(stderr, "[expert_cache] Mode: Direct memory-mapped access (Phase 2a)\n");
    
    return 1;
}

// Legacy init function - redirects to tensor-based init
int q38_expert_cache_init(
    const Q38VulkanContext *ctx,
    Q38ExpertCache *cache,
    const char *gguf_path,
    const Q38ExpertCacheConfig *config) {
    
    fprintf(stderr, "[expert_cache] Legacy init not supported - use init_from_tensors()\n");
    return 0;
}

void q38_expert_cache_shutdown(
    const Q38VulkanContext *ctx,
    Q38ExpertCache *cache) {
    
    if (!ctx || !cache) return;
    
    // Print final statistics
    if (cache->config.debug_stats) {
        q38_expert_cache_print_stats(cache);
    }
    
    // Free per-layer structures
    for (uint32_t layer = 0; layer < cache->n_layers; layer++) {
        Q38ExpertLayerInfo *info = &cache->layers[layer];
        free(info->slot_expert);
        free(info->slot_valid);
        free(info->slot_used);
        free(info->slot_freq);
        free(info->expert_slot);
    }
    
    // Free VRAM
    if (cache->vram_buffer) {
        // Note: memory was mapped, need to unmap first
        // But we don't have the VkDeviceMemory handle stored separately
        // In production code, we'd store it
        vkDestroyBuffer(ctx->device, cache->vram_buffer, NULL);
    }
    
    // Free RAM
    free(cache->ram_base);
    
    // Close file descriptors
    for (int i = 0; i < Q38_EXPERT_NPARTS; i++) {
        if (cache->fd_shard[i] >= 0) {
            close(cache->fd_shard[i]);
        }
    }
    
    memset(cache, 0, sizeof(*cache));
}

// === CORE OPERATIONS ===

bool q38_expert_cache_fetch(
    Q38ExpertCache *cache,
    uint32_t layer,
    const uint32_t *expert_ids,
    uint32_t n,
    Q38ExpertHandle *handles) {
    
    bool ready[Q38_MAX_EXPERTS_PER_TOKEN];
    if (!q38_expert_cache_fetch_begin(cache, layer, expert_ids, n, handles, ready)) {
        return false;
    }
    
    // Wait for all to be ready
    if (!q38_expert_cache_fetch_end(cache)) {
        return false;
    }
    
    return true;
}

bool q38_expert_cache_fetch_begin(
    Q38ExpertCache *cache,
    uint32_t layer,
    const uint32_t *expert_ids,
    uint32_t n,
    Q38ExpertHandle *handles,
    bool *ready) {
    
    if (!cache || !expert_ids || !handles || !ready) return false;
    if (layer >= cache->n_layers) return false;
    if (n > Q38_MAX_EXPERTS_PER_TOKEN) return false;
    
    Q38ExpertLayerInfo *info = &cache->layers[layer];
    cache->timestamp++;
    
    // Age counters periodically
    if (cache->timestamp % cache->config.age_every == 0) {
        for (uint32_t i = 0; i < info->n_slots; i++) {
            info->slot_freq[i] /= 2;
        }
    }
    
    for (uint32_t i = 0; i < n; i++) {
        uint32_t expert_id = expert_ids[i];
        cache->stats.lookups++;
        
        if (expert_id >= cache->n_experts) {
            // Invalid expert ID
            handles[i].valid = false;
            ready[i] = false;
            continue;
        }
        
        // Check if expert is already cached
        int32_t slot = info->expert_slot[expert_id];
        
        if (slot >= 0 && info->slot_valid[slot]) {
            // Cache hit!
            cache->stats.hits++;
            
            // Update tracking
            info->slot_used[slot] = cache->timestamp;
            info->slot_freq[slot]++;
            
            // Fill handle
            handles[i].valid = true;
            handles[i].slot = slot;
            handles[i].late = false;
            
            if (slot < (int32_t)info->n_vram_slots) {
                // In VRAM - best case!
                cache->stats.gpu_hits++;
                handles[i].on_gpu = true;
                handles[i].buffer = cache->vram_buffer;
                
                // Calculate pointers to each part
                uint8_t *base = cache->vram_mapped + slot * info->block_bytes;
                for (int p = 0; p < Q38_EXPERT_NPARTS; p++) {
                    handles[i].parts[p] = base + info->part_offset[p];
                    handles[i].quant_type[p] = 11;  // Q3_K
                }
            } else {
                // In RAM - still good
                handles[i].on_gpu = false;
                handles[i].buffer = VK_NULL_HANDLE;
                
                uint32_t ram_slot = slot - info->n_vram_slots;
                uint8_t *base = cache->ram_base + ram_slot * info->block_bytes;
                for (int p = 0; p < Q38_EXPERT_NPARTS; p++) {
                    handles[i].parts[p] = base + info->part_offset[p];
                    handles[i].quant_type[p] = 11;  // Q3_K
                }
            }
            
            ready[i] = true;
        } else {
            // Cache miss - need to load from disk
            cache->stats.misses++;
            
            handles[i].valid = false;
            handles[i].on_gpu = false;
            ready[i] = false;
            
            // For now, just mark as not ready
            // TODO: Implement actual loading from disk
            fprintf(stderr, "[expert_cache] Layer %u expert %u not cached (miss rate %.1f%%)\n",
                    layer, expert_id, 100.0 * cache->stats.misses / cache->stats.lookups);
        }
    }
    
    return true;
}

bool q38_expert_cache_fetch_end(
    Q38ExpertCache *cache) {
    
    // For now, no async operations pending
    // TODO: Wait for any pending disk reads
    
    return true;
}

// === SPECULATIVE PREFETCH ===

void q38_expert_cache_prefetch(
    Q38ExpertCache *cache,
    uint32_t layer,
    const uint32_t *expert_ids,
    uint32_t n) {
    
    if (!cache || !expert_ids) return;
    if (!cache->config.enable_prefetch) return;
    
    cache->stats.prefetch_issued += n;
    
    // TODO: Issue async reads for predicted experts
    // For now, this is a no-op
}

void q38_expert_cache_settle(
    Q38ExpertCache *cache) {
    
    // TODO: Wait for all pending async operations
}

// === STATISTICS ===

void q38_expert_cache_get_stats(
    const Q38ExpertCache *cache,
    Q38ExpertCacheStats *stats) {
    
    if (!cache || !stats) return;
    *stats = cache->stats;
}

void q38_expert_cache_print_stats(
    const Q38ExpertCache *cache) {
    
    if (!cache) return;
    
    double hit_rate = cache->stats.lookups > 0 ? 
        (double)cache->stats.hits / cache->stats.lookups : 0.0;
    double gpu_rate = cache->stats.lookups > 0 ?
        (double)cache->stats.gpu_hits / cache->stats.lookups : 0.0;
    
    fprintf(stderr, "\n[expert_cache] === Statistics ===\n");
    fprintf(stderr, "[expert_cache] Lookups: %lu\n", cache->stats.lookups);
    fprintf(stderr, "[expert_cache] Hits: %lu (%.1f%%)\n", cache->stats.hits, hit_rate * 100.0);
    fprintf(stderr, "[expert_cache] GPU hits: %lu (%.1f%%)\n", cache->stats.gpu_hits, gpu_rate * 100.0);
    fprintf(stderr, "[expert_cache] Misses: %lu\n", cache->stats.misses);
    fprintf(stderr, "[expert_cache] Promotions: %lu\n", cache->stats.promotions);
    fprintf(stderr, "[expert_cache] Evictions: %lu\n", cache->stats.evictions);
    fprintf(stderr, "[expert_cache] Bytes read: %lu MB\n", cache->stats.bytes_read / (1024 * 1024));
}

// === GLOBAL EXPERT CACHE FOR TESTING ===
// In production, this would be part of Q4Model

static Q38ExpertCache g_expert_cache;
static bool g_expert_cache_initialized = false;

// Initialize global expert cache (call once during model load)
int q38_init_global_expert_cache(
    const Q38VulkanContext *ctx,
    uint32_t n_layers,
    uint32_t n_experts,
    uint32_t n_embd,
    uint32_t n_ff,
    const Q38GGUFTensor **gate_tensors,
    const Q38GGUFTensor **up_tensors,
    const Q38GGUFTensor **down_tensors) {
    
    if (g_expert_cache_initialized) {
        fprintf(stderr, "[expert_cache] Already initialized\n");
        return 1;
    }
    
    if (!q38_expert_cache_init_from_tensors(ctx, &g_expert_cache,
        n_layers, n_experts, n_embd, n_ff,
        gate_tensors, up_tensors, down_tensors, NULL)) {
        fprintf(stderr, "[expert_cache] Failed to initialize\n");
        return 0;
    }
    
    g_expert_cache_initialized = true;
    return 1;
}

Q38ExpertCache* q38_get_global_expert_cache(void) {
    return g_expert_cache_initialized ? &g_expert_cache : NULL;
}

void q38_shutdown_global_expert_cache(const Q38VulkanContext *ctx) {
    if (g_expert_cache_initialized) {
        q38_expert_cache_shutdown(ctx, &g_expert_cache);
        g_expert_cache_initialized = false;
    }
}
