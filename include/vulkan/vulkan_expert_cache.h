#ifndef VULKAN_EXPERT_CACHE_H
#define VULKAN_EXPERT_CACHE_H

#include "vulkan_context.h"
#include "vulkan_buffers.h"
#include "../qwen38/qwen38_gguf.h"  // For Q38GGUFTensor definition
#include <stdint.h>
#include <stdbool.h>

// Three-tier expert cache for MoE models
// Manages expert weights across VRAM (T0), RAM (T1), and Disk (T2)
//
// Architecture:
// - Model has 512 experts × 48 layers = 55.8 GB total
// - Each token touches ~480 experts (~1.05 GB)
// - Cache keeps hot experts in VRAM, warm in RAM, cold on disk
//
// Performance target: 91-97% hit rate, 57-86% GPU hits

#ifdef __cplusplus
extern "C" {
#endif

// Maximum values
#define Q38_MAX_LAYERS 64
#define Q38_MAX_EXPERTS 512
#define Q38_MAX_EXPERTS_PER_TOKEN 16
#define Q38_EXPERT_NPARTS 3  // gate, up, down projections

// Expert handle returned to caller
typedef struct {
    const uint8_t *parts[Q38_EXPERT_NPARTS];  // Pointers to weight data
    uint32_t quant_type[Q38_EXPERT_NPARTS];   // Quantization type per part
    VkBuffer buffer;                          // Which tier holds this expert
    bool on_gpu;                              // Currently in VRAM?
    int32_t slot;                             // Slot index in tier
    bool late;                                // Promoted during this fetch
    bool valid;                               // Handle contains valid data
} Q38ExpertHandle;

// Per-layer slot allocation
typedef struct {
    uint32_t block_bytes;                     // Size of one expert block
    uint32_t part_bytes[Q38_EXPERT_NPARTS];   // Size per part
    uint32_t part_offset[Q38_EXPERT_NPARTS];  // Offset within block
    
    // Slot management
    uint16_t *slot_expert;                    // Expert ID in each slot
    uint8_t *slot_valid;                      // Data loaded?
    uint64_t *slot_used;                      // Last use timestamp
    uint32_t *slot_freq;                      // Access frequency
    
    // Lookup table: expert_id -> slot (-1 if not cached)
    int32_t *expert_slot;
    
    uint32_t n_slots;                         // Total slots in this layer
    uint32_t n_vram_slots;                    // Slots in VRAM tier
    uint32_t n_ram_slots;                     // Slots in RAM tier
} Q38ExpertLayerInfo;

// Statistics
typedef struct {
    uint64_t lookups;
    uint64_t hits;
    uint64_t misses;
    uint64_t gpu_hits;                        // Served from VRAM
    uint64_t promotions;                      // RAM -> VRAM copies
    uint64_t evictions;
    uint64_t bytes_read;
    uint64_t prefetch_issued;
    uint64_t prefetch_used;
    double total_time_us;
} Q38ExpertCacheStats;

// Configuration
typedef struct {
    size_t vram_budget;                       // Bytes for T0 (VRAM)
    size_t ram_budget;                        // Bytes for T1 (RAM)
    uint32_t max_promotions_per_layer;        // Limit H2D traffic
    uint32_t evict_samples;                   // Sample size for eviction
    uint64_t age_every;                       // Halve counters every N lookups
    bool enable_prefetch;                     // Speculative prefetch
    bool debug_stats;                         // Print stats on cleanup
} Q38ExpertCacheConfig;

// Main cache structure
typedef struct {
    // T0: VRAM tier
    VkBuffer vram_buffer;
    uint8_t *vram_mapped;                     // Host-visible mapping
    size_t vram_capacity;
    uint32_t vram_slots_total;
    
    // T1: RAM tier (pinned memory)
    uint8_t *ram_base;
    size_t ram_capacity;
    uint32_t ram_slots_total;
    bool ram_pinned;                          // Using pinned memory
    
    // T2: Disk tier (file descriptors)
    int fd_shard[Q38_EXPERT_NPARTS];          // GGUF file descriptors
    uint64_t shard_offsets[Q38_EXPERT_NPARTS]; // Base offsets in files
    
    // Per-layer pools
    Q38ExpertLayerInfo layers[Q38_MAX_LAYERS];
    uint32_t n_layers;
    
    // Expert dimensions (from model)
    uint32_t n_embd;                          // Embedding dimension
    uint32_t n_ff;                            // Feed-forward dimension
    uint32_t n_experts;                       // Number of experts
    
    // Timing
    uint64_t timestamp;                       // Monotonic counter
    
    // Configuration
    Q38ExpertCacheConfig config;
    
    // Statistics
    Q38ExpertCacheStats stats;
    
    // Context reference
    const Q38VulkanContext *ctx;
} Q38ExpertCache;

// Initialize expert cache with pre-loaded tensors
int q38_expert_cache_init_from_tensors(
    const Q38VulkanContext *ctx,
    Q38ExpertCache *cache,
    uint32_t n_layers,
    uint32_t n_experts,
    uint32_t n_embd,
    uint32_t n_ff,
    const Q38GGUFTensor **gate_tensors,   // Array of n_layers tensors
    const Q38GGUFTensor **up_tensors,
    const Q38GGUFTensor **down_tensors,
    const Q38ExpertCacheConfig *config);

// Cleanup expert cache
void q38_expert_cache_shutdown(
    const Q38VulkanContext *ctx,
    Q38ExpertCache *cache);

// === CORE OPERATIONS ===

// Blocking fetch of n experts on one layer
// Returns handles for all requested experts
bool q38_expert_cache_fetch(
    Q38ExpertCache *cache,
    uint32_t layer,
    const uint32_t *expert_ids,
    uint32_t n,
    Q38ExpertHandle *handles);

// Split fetch for overlap:
// fetch_begin() returns immediately with ready flags
// Caller computes ready experts while others load
// fetch_end() waits for remaining experts
bool q38_expert_cache_fetch_begin(
    Q38ExpertCache *cache,
    uint32_t layer,
    const uint32_t *expert_ids,
    uint32_t n,
    Q38ExpertHandle *handles,
    bool *ready);

bool q38_expert_cache_fetch_end(
    Q38ExpertCache *cache);

// === SPECULATIVE PREFETCH ===

// Prefetch predicted experts for next layer
void q38_expert_cache_prefetch(
    Q38ExpertCache *cache,
    uint32_t layer,
    const uint32_t *expert_ids,
    uint32_t n);

// Settle all pending async operations
void q38_expert_cache_settle(
    Q38ExpertCache *cache);

// === STATISTICS ===

void q38_expert_cache_get_stats(
    const Q38ExpertCache *cache,
    Q38ExpertCacheStats *stats);

void q38_expert_cache_print_stats(
    const Q38ExpertCache *cache);

// === GLOBAL CACHE MANAGEMENT ===

// Initialize global expert cache (simplified interface for testing)
int q38_init_global_expert_cache(
    const Q38VulkanContext *ctx,
    uint32_t n_layers,
    uint32_t n_experts,
    uint32_t n_embd,
    uint32_t n_ff,
    const Q38GGUFTensor **gate_tensors,
    const Q38GGUFTensor **up_tensors,
    const Q38GGUFTensor **down_tensors);

// Get global cache instance
Q38ExpertCache* q38_get_global_expert_cache(void);

// Shutdown global cache
void q38_shutdown_global_expert_cache(const Q38VulkanContext *ctx);

#ifdef __cplusplus
}
#endif

#endif // VULKAN_EXPERT_CACHE_H