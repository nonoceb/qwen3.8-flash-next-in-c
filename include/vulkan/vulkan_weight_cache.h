#ifndef VULKAN_WEIGHT_CACHE_H
#define VULKAN_WEIGHT_CACHE_H

#include "vulkan_context.h"
#include "vulkan_buffers.h"
#include <stdint.h>
#include <stdbool.h>

// Weight caching system for Vulkan GEMV operations
// Implements three-tier caching with LRU eviction and async transfers

#ifdef __cplusplus
extern "C" {
#endif

// Cache tier levels
typedef enum {
    Q38_CACHE_TIER_VRAM = 0,   // GPU memory (fastest)
    Q38_CACHE_TIER_RAM  = 1,   // System RAM (pinned if available)
    Q38_CACHE_TIER_DISK = 2    // Not yet loaded
} Q38CacheTier;

// Eviction policy
typedef enum {
    Q38_EVICT_LRU,     // Least recently used
    Q38_EVICT_LFU,     // Least frequently used  
    Q38_EVICT_HYBRID   // Combine recency + frequency
} Q38EvictPolicy;

// Forward declaration
typedef struct Q38WeightCache Q38WeightCache;

// Single cache entry
typedef struct {
    const void *tensor_data;      // Pointer identity for lookup (key)
    uint64_t tensor_size;         // Size in bytes
    uint64_t padded_size;         // Padded size (for IQ4_NL alignment)
    Q38VulkanBuffer buffer;       // GPU buffer
    
    // Matrix dimensions (for validation)
    uint32_t rows;                // M dimension
    uint32_t cols;                // N dimension
    
    // Eviction tracking
    uint64_t last_used;           // Timestamp for LRU
    uint32_t freq;                // Access count for LFU
    
    // Metadata
    int tensor_type;              // Quantization type
    uint32_t layer_id;            // Which layer this belongs to (-1 if unknown)
    Q38CacheTier tier;            // Current location
    
    // State flags
    bool valid;                   // Entry is populated
    bool on_gpu;                  // Currently in VRAM
    bool late;                    // Promoted during this fetch
    bool transfer_in_flight;      // Async transfer pending
    
    // Async transfer tracking
    VkFence transfer_fence;       // Fence for async operations
} Q38WeightCacheEntry;

// Statistics
typedef struct {
    uint64_t lookups;
    uint64_t hits;
    uint64_t misses;
    uint64_t evictions;
    uint64_t promotions;          // RAM -> VRAM copies
    uint64_t prefetch_issued;     // Speculative prefetches issued
    uint64_t prefetch_used;       // Prefetches that were actually used
    uint64_t prefetch_wasted;     // Prefetches evicted before use
    uint64_t bytes_transferred;
    double total_transfer_time_us;
    double total_wait_time_us;
} Q38WeightCacheStats;

// Configuration
typedef struct {
    uint32_t max_entries;         // Maximum cache entries (default: 256)
    uint32_t max_vram_bytes;      // VRAM budget (0 = auto-detect)
    uint32_t max_promotions_per_layer;  // Limit H2D traffic per layer
    Q38EvictPolicy eviction_policy;
    uint32_t evict_samples;       // Sample size for eviction (default: 16)
    uint64_t age_every;           // Halve counters every N lookups (default: 65536)
    bool enable_prefetch;         // Enable speculative prefetch
    bool enable_async;            // Enable async transfers
    bool debug_stats;             // Print statistics on cleanup
} Q38WeightCacheConfig;

// Speculative prefetch prediction model
#define MAX_PREDICTION_HISTORY 1024
#define MAX_EXPERTS_PER_LAYER 512

typedef struct {
    uint32_t expert_id;
    uint32_t successor_count[MAX_EXPERTS_PER_LAYER];  // What comes after this expert
    uint32_t total_successors;
} Q38ExpertPattern;

typedef struct {
    Q38ExpertPattern patterns[MAX_PREDICTION_HISTORY];
    uint32_t n_patterns;
    uint32_t predictions_correct;
    uint32_t predictions_made;
} Q38PredictionModel;

// Main cache structure
struct Q38WeightCache {
    Q38WeightCacheEntry *entries;
    uint32_t n_entries;
    uint32_t max_entries;
    
    // Timing
    uint64_t timestamp;           // Monotonic counter for LRU
    
    // Configuration
    Q38WeightCacheConfig config;
    
    // Statistics
    Q38WeightCacheStats stats;
    
    // Prediction model for speculative prefetch
    Q38PredictionModel prediction;
    
    // Pending async transfers
    uint32_t n_pending_transfers;
    
    // Context reference (non-owning)
    const Q38VulkanContext *ctx;
};

// Initialize weight cache
int q38_weight_cache_init(
    const Q38VulkanContext *ctx,
    Q38WeightCache *cache,
    const Q38WeightCacheConfig *config);

// Cleanup weight cache
void q38_weight_cache_cleanup(
    const Q38VulkanContext *ctx,
    Q38WeightCache *cache);

// === CORE CACHE OPERATIONS ===

// Get or create cached weight buffer (blocking)
// Returns NULL on failure
Q38VulkanBuffer* q38_weight_cache_get(
    const Q38VulkanContext *ctx,
    Q38WeightCache *cache,
    const void *tensor_data,
    uint64_t tensor_size,
    int tensor_type,
    uint32_t layer_id);

// Extended version with matrix dimensions for validation
Q38VulkanBuffer* q38_weight_cache_get_ext(
    const Q38VulkanContext *ctx,
    Q38WeightCache *cache,
    const void *tensor_data,
    uint64_t tensor_size,
    int tensor_type,
    uint32_t layer_id,
    uint32_t rows,
    uint32_t cols);

// === ASYNC OPERATIONS ===

// Begin non-blocking fetch
// Sets *ready = true if already cached, false if transfer started
// Returns buffer pointer (may be staging buffer until ready)
Q38VulkanBuffer* q38_weight_cache_fetch_begin(
    const Q38VulkanContext *ctx,
    Q38WeightCache *cache,
    const void *tensor_data,
    uint64_t tensor_size,
    int tensor_type,
    uint32_t layer_id,
    bool *ready);

// Wait for pending transfer to complete
int q38_weight_cache_fetch_end(
    const Q38VulkanContext *ctx,
    Q38WeightCache *cache,
    const void *tensor_data);

// Settle all pending async transfers
void q38_weight_cache_settle(
    const Q38VulkanContext *ctx,
    Q38WeightCache *cache);

// === SPECULATIVE PREFETCH ===

// Issue speculative prefetch for predicted weights
void q38_weight_cache_prefetch(
    const Q38VulkanContext *ctx,
    Q38WeightCache *cache,
    const void *tensor_data,
    uint64_t tensor_size,
    int tensor_type,
    uint32_t layer_id);

// Predict next layer's experts based on current routing
void q38_weight_cache_predict_next(
    Q38WeightCache *cache,
    uint32_t current_layer,
    const uint32_t *current_experts,
    uint32_t n_experts,
    uint32_t *predicted_experts,
    uint32_t n_predict);

// Update prediction model with actual routing
void q38_weight_cache_update_prediction(
    Q38WeightCache *cache,
    uint32_t layer,
    const uint32_t *experts,
    uint32_t n_experts);

// === STATISTICS ===

// Get cache statistics
void q38_weight_cache_get_stats(
    const Q38WeightCache *cache,
    Q38WeightCacheStats *stats);

// Print cache statistics
void q38_weight_cache_print_stats(
    const Q38WeightCache *cache);

// Reset statistics
void q38_weight_cache_reset_stats(
    Q38WeightCache *cache);

// === UTILITY ===

// Check if a weight is currently cached
bool q38_weight_cache_contains(
    const Q38WeightCache *cache,
    const void *tensor_data);

// Invalidate a specific cache entry
void q38_weight_cache_invalidate(
    const Q38VulkanContext *ctx,
    Q38WeightCache *cache,
    const void *tensor_data);

// Clear entire cache
void q38_weight_cache_clear(
    const Q38VulkanContext *ctx,
    Q38WeightCache *cache);

// Age all counters (halve frequencies)
void q38_weight_cache_age(
    Q38WeightCache *cache);

#ifdef __cplusplus
}
#endif

#endif // VULKAN_WEIGHT_CACHE_H