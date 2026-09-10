#include "vulkan/vulkan_expert_cache.h"
#include "qwen38/qwen38_gguf.h"  // For Q38GGUFTensor
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>

// Store tensor pointers for direct access during cache misses
typedef struct {
    const Q38GGUFTensor **gate_tensors;
    const Q38GGUFTensor **up_tensors;
    const Q38GGUFTensor **down_tensors;
    uint32_t n_layers;
    bool initialized;
} Q38ExpertTensorStore;

static Q38ExpertTensorStore g_tensor_store = {0};

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

// Find or create a slot for an expert
static int32_t find_or_create_slot(Q38ExpertCache *cache, uint32_t layer, uint32_t expert_id) {
    Q38ExpertLayerInfo *info = &cache->layers[layer];
    
    if (info->n_slots == 0) return -1;
    
    // Check if already cached
    if (info->expert_slot[expert_id] >= 0) {
        return info->expert_slot[expert_id];
    }
    
    // Find empty slot
    for (uint32_t i = 0; i < info->n_slots; i++) {
        if (!info->slot_valid[i]) {
            info->expert_slot[expert_id] = (int32_t)i;
            info->slot_expert[i] = (uint16_t)expert_id;
            return (int32_t)i;
        }
    }
    
    // Need to evict - use hybrid LRU/LFU
    int victim = find_victim_slot(cache, layer, true);
    if (victim < 0) return -1;
    
    // Mark old expert as not cached
    uint16_t old_expert = info->slot_expert[victim];
    if (old_expert < cache->n_experts) {
        info->expert_slot[old_expert] = -1;
    }
    
    cache->stats.evictions++;
    
    // Assign new expert to slot
    info->expert_slot[expert_id] = (int32_t)victim;
    info->slot_expert[victim] = (uint16_t)expert_id;
    info->slot_valid[victim] = 0;  // Will be set after load
    
    return (int32_t)victim;
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
        cache->config.vram_budget = 4ull * 1024 * 1024 * 1024;  // 4 GB (conservative for iGPU)
        cache->config.ram_budget = 16ull * 1024 * 1024 * 1024;  // 16 GB
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
    if (!gate0) {
        fprintf(stderr, "[expert_cache] Gate tensor is NULL\n");
        return 0;
    }
    fprintf(stderr, "[expert_cache] Gate tensor: n_dims=%u, nbytes=%lu, shape=[%lu,%lu,%lu]\n",
            gate0->n_dims, gate0->nbytes, 
            gate0->n_dims > 0 ? gate0->shape[0] : 0,
            gate0->n_dims > 1 ? gate0->shape[1] : 0,
            gate0->n_dims > 2 ? gate0->shape[2] : 0);
    
    if (gate0->n_dims != 3) {
        fprintf(stderr, "[expert_cache] Invalid gate tensor dimensions (expected 3)\n");
        return 0;
    }
    
    // Size per expert (assuming all layers have same structure)
    uint64_t expert_bytes = gate0->nbytes / n_experts;
    fprintf(stderr, "[expert_cache] Expert size: %lu KB per expert part (%lu bytes)\n", 
            expert_bytes / 1024, expert_bytes);
    
    // Total expert size = gate + up + down
    uint64_t total_expert_bytes = expert_bytes * 3;
    fprintf(stderr, "[expert_cache] Total expert size: %lu KB (%.2f MB)\n", 
            total_expert_bytes / 1024, total_expert_bytes / (1024.0 * 1024.0));
    
    // Allocate persistent storage for tensor pointer arrays
    // (The arrays passed in are stack-allocated and become invalid after this function returns)
    g_tensor_store.gate_tensors = malloc(n_layers * sizeof(Q38GGUFTensor*));
    g_tensor_store.up_tensors = malloc(n_layers * sizeof(Q38GGUFTensor*));
    g_tensor_store.down_tensors = malloc(n_layers * sizeof(Q38GGUFTensor*));
    
    if (!g_tensor_store.gate_tensors || !g_tensor_store.up_tensors || !g_tensor_store.down_tensors) {
        fprintf(stderr, "[expert_cache] Failed to allocate tensor storage\n");
        free(g_tensor_store.gate_tensors);
        free(g_tensor_store.up_tensors);
        free(g_tensor_store.down_tensors);
        q38_expert_cache_shutdown(ctx, cache);
        return 0;
    }
    
    // Copy the tensor pointers to persistent storage
    memcpy(g_tensor_store.gate_tensors, gate_tensors, n_layers * sizeof(Q38GGUFTensor*));
    memcpy(g_tensor_store.up_tensors, up_tensors, n_layers * sizeof(Q38GGUFTensor*));
    memcpy(g_tensor_store.down_tensors, down_tensors, n_layers * sizeof(Q38GGUFTensor*));
    g_tensor_store.n_layers = n_layers;
    g_tensor_store.initialized = true;
    
    // Calculate how many experts fit in VRAM budget per layer
    size_t vram_budget = cache->config.vram_budget;
    uint32_t slots_per_layer = (uint32_t)(vram_budget / total_expert_bytes / n_layers);
    
    // Clamp to reasonable limits
    if (slots_per_layer > n_experts) slots_per_layer = n_experts;
    if (slots_per_layer < 1) slots_per_layer = 1;
    
    fprintf(stderr, "[expert_cache] VRAM budget: %lu MB\n", vram_budget / (1024 * 1024));
    fprintf(stderr, "[expert_cache] Slots per layer: %u experts\n", slots_per_layer);
    
    // Allocate T0: VRAM tier for hot experts
    size_t vram_needed = (size_t)slots_per_layer * total_expert_bytes * n_layers;
    
    // Add padding for each layer to prevent over-read issues
    const size_t LAYER_PAD = 4096;
    vram_needed += LAYER_PAD * n_layers;
    
    fprintf(stderr, "[expert_cache] Allocating VRAM buffer: %lu MB\n", vram_needed / (1024 * 1024));
    
    // Create Vulkan buffer for expert cache
    Q38VulkanBuffer vram_buf;
    if (!q38_vulkan_buffer_create(ctx, &vram_buf, vram_needed,
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true)) {
        fprintf(stderr, "[expert_cache] Failed to allocate VRAM tier - falling back to CPU-only\n");
        cache->vram_capacity = 0;
        cache->vram_slots_total = 0;
    } else {
        // Map the buffer for host access (UMA architecture)
        if (!q38_vulkan_buffer_map(ctx, &vram_buf)) {
            fprintf(stderr, "[expert_cache] Failed to map VRAM buffer\n");
            q38_vulkan_buffer_destroy(ctx, &vram_buf);
            cache->vram_capacity = 0;
            cache->vram_slots_total = 0;
        } else {
            cache->vram_buffer = vram_buf.buffer;
            cache->vram_memory = vram_buf.memory;
            cache->vram_mapped = vram_buf.mapped_ptr;
            cache->vram_capacity = vram_needed;
            cache->vram_slots_total = slots_per_layer * n_layers;
            
            fprintf(stderr, "[expert_cache] VRAM tier allocated successfully\n");
            fprintf(stderr, "[expert_cache] Buffer is %s\n", 
                    vram_buf.is_unified ? "unified (zero-copy)" : "host-visible");
        }
    }
    
    // RAM tier not used - we rely on memory-mapped GGUF files as T2
    cache->ram_base = NULL;
    cache->ram_capacity = 0;
    cache->ram_slots_total = 0;
    
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
        
        // Set up VRAM slots for this layer
        info->n_vram_slots = slots_per_layer;
        info->n_ram_slots = 0;  // Not using RAM tier
        info->n_slots = slots_per_layer;
        
        // Allocate tracking arrays
        info->slot_expert = (uint16_t*)calloc(slots_per_layer, sizeof(uint16_t));
        info->slot_valid = (uint8_t*)calloc(slots_per_layer, sizeof(uint8_t));
        info->slot_used = (uint64_t*)calloc(slots_per_layer, sizeof(uint64_t));
        info->slot_freq = (uint32_t*)calloc(slots_per_layer, sizeof(uint32_t));
        info->expert_slot = (int32_t*)malloc(n_experts * sizeof(int32_t));
        
        if (!info->slot_expert || !info->slot_valid || !info->slot_used || 
            !info->slot_freq || !info->expert_slot) {
            fprintf(stderr, "[expert_cache] Failed to allocate layer %u structures\n", layer);
            q38_expert_cache_shutdown(ctx, cache);
            return 0;
        }
        
        // Mark all slots as empty
        for (uint32_t i = 0; i < slots_per_layer; i++) {
            info->slot_expert[i] = 0xFFFF;  // Invalid expert ID
            info->slot_valid[i] = 0;
            info->slot_used[i] = 0;
            info->slot_freq[i] = 0;
        }
        
        // Mark all experts as NOT in cache
        for (uint32_t i = 0; i < n_experts; i++) {
            info->expert_slot[i] = -1;
        }
        
        // Calculate base pointer in VRAM for this layer
        if (cache->vram_mapped) {
            info->vram_base = cache->vram_mapped + layer * slots_per_layer * total_expert_bytes;
        } else {
            info->vram_base = NULL;
        }
    }
    
    // Store tensor pointers for direct access
    // We'll add these to the cache structure
    
    fprintf(stderr, "[expert_cache] Initialization complete\n");
    if (cache->vram_mapped) {
        fprintf(stderr, "[expert_cache] Mode: VRAM caching enabled (Phase 2b)\n");
    } else {
        fprintf(stderr, "[expert_cache] Mode: Direct memory-mapped access (fallback)\n");
    }
    
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
    
    // Free VRAM buffer using proper Vulkan buffer management
    if (cache->vram_buffer) {
        Q38VulkanBuffer buf = {
            .buffer = cache->vram_buffer,
            .memory = cache->vram_memory,
            .mapped_ptr = cache->vram_mapped,
            .size = cache->vram_capacity
        };
        q38_vulkan_buffer_destroy(ctx, &buf);
    }
    
    // Free RAM
    free(cache->ram_base);
    
    // Close file descriptors
    for (int i = 0; i < Q38_EXPERT_NPARTS; i++) {
        if (cache->fd_shard[i] >= 0) {
            close(cache->fd_shard[i]);
        }
    }
    
    // Free global tensor store
    if (g_tensor_store.initialized) {
        free(g_tensor_store.gate_tensors);
        free(g_tensor_store.up_tensors);
        free(g_tensor_store.down_tensors);
        memset(&g_tensor_store, 0, sizeof(g_tensor_store));
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
            // Cache miss - need to load from memory-mapped data
            cache->stats.misses++;
            
            // Check if we have VRAM caching available
            if (cache->vram_mapped && info->n_slots > 0) {
                // Find or create a slot for this expert
                int32_t new_slot = find_or_create_slot(cache, layer, expert_id);
                
                if (new_slot >= 0 && info->vram_base) {
                    // Load expert weights from GGUF tensors to VRAM
                    const Q38GGUFTensor *gate = g_tensor_store.gate_tensors[layer];
                    const Q38GGUFTensor *up = g_tensor_store.up_tensors[layer];
                    const Q38GGUFTensor *down = g_tensor_store.down_tensors[layer];
                    
                    if (gate && up && down && gate->data && up->data && down->data) {
                        // Calculate source pointers (expert slice in each tensor)
                        const uint8_t *src_gate = gate->data + expert_id * info->part_bytes[0];
                        const uint8_t *src_up = up->data + expert_id * info->part_bytes[1];
                        const uint8_t *src_down = down->data + expert_id * info->part_bytes[2];
                        
                        // Calculate destination pointer in VRAM
                        uint8_t *dst = info->vram_base + new_slot * info->block_bytes;
                        
                        // Copy all three parts
                        memcpy(dst + info->part_offset[0], src_gate, info->part_bytes[0]);
                        memcpy(dst + info->part_offset[1], src_up, info->part_bytes[1]);
                        memcpy(dst + info->part_offset[2], src_down, info->part_bytes[2]);
                        
                        // Update tracking
                        info->slot_valid[new_slot] = 1;
                        info->slot_used[new_slot] = cache->timestamp;
                        info->slot_freq[new_slot] = 1;
                        
                        // Fill handle
                        handles[i].valid = true;
                        handles[i].on_gpu = true;
                        handles[i].slot = new_slot;
                        handles[i].late = true;  // Promoted during this fetch
                        handles[i].buffer = cache->vram_buffer;
                        
                        // Set part pointers
                        for (int p = 0; p < Q38_EXPERT_NPARTS; p++) {
                            handles[i].parts[p] = dst + info->part_offset[p];
                            handles[i].quant_type[p] = 11;  // Q3_K
                        }
                        
                        ready[i] = true;
                        cache->stats.promotions++;
                        cache->stats.bytes_read += info->block_bytes;
                    } else {
                        // No tensor data available
                        handles[i].valid = false;
                        ready[i] = false;
                    }
                } else {
                    // No slot available
                    handles[i].valid = false;
                    ready[i] = false;
                }
            } else {
                // No VRAM caching - mark as not ready (will use CPU path)
                handles[i].valid = false;
                handles[i].on_gpu = false;
                ready[i] = false;
            }
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
