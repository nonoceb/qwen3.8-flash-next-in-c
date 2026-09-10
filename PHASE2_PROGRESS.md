# Phase 2 Progress: Three-Tier Expert Cache

**Date**: 2026-09-08
**Status**: 🔨 Infrastructure Complete, Integration In Progress

---

## What Was Accomplished

### 1. Created Expert Cache Header

**File**: `include/vulkan/vulkan_expert_cache.h` (189 lines)

**Key Components**:
- `Q38ExpertHandle`: Handle for accessing expert weights
- `Q38ExpertLayerInfo`: Per-layer slot management
- `Q38ExpertCache`: Main cache structure with three tiers
- `Q38ExpertCacheStats`: Performance tracking

**Features Defined**:
- T0: VRAM tier (GPU memory)
- T1: RAM tier (pinned host memory)
- T2: Disk tier (file-based storage)
- Split fetch API (begin/end for overlap)
- Speculative prefetch support

### 2. Implemented Core Cache Logic

**File**: `src/vulkan/vulkan_expert_cache.c` (465 lines)

**Implemented Functions**:
- ✅ `q38_expert_cache_init()`: Initialize all three tiers
- ✅ `q38_expert_cache_shutdown()`: Cleanup and print stats
- ✅ `q38_expert_cache_fetch()`: Blocking expert fetch
- ✅ `q38_expert_cache_fetch_begin()`: Non-blocking fetch start
- ✅ `q38_expert_cache_fetch_end()`: Wait for completion
- ✅ `q38_expert_cache_prefetch()`: Speculative prefetch stub
- ✅ `q38_expert_cache_print_stats()`: Statistics output

**Memory Management**:
- VRAM buffer allocation via Vulkan
- Host-visible mapping for unified memory (AMD 780M)
- RAM tier allocation with malloc (TODO: use pinned memory)
- Per-layer slot tracking arrays

### 3. Added to Build System

**File**: `GNUmakefile`
- Added `vulkan_expert_cache.o` to build
- Successfully compiles and links

### 4. Verified Compatibility

- ✅ Build succeeds without errors
- ✅ Existing inference still works
- ✅ No regressions in performance

---

## Current Architecture

```
┌─────────────────────────────────────────┐
│         Q38ExpertCache                  │
├─────────────────────────────────────────┤
│ T0: VRAM (~4 GB, ~2,100 experts)       │
│     - VkBuffer + mapped memory          │
│     - Direct GPU access                 │
│     - Target: 57-86% hit rate           │
├─────────────────────────────────────────┤
│ T1: RAM (~16 GB, ~8,700 experts)       │
│     - malloc'd buffer                   │
│     - Fast CPU access                   │
│     - TODO: Use pinned memory           │
├─────────────────────────────────────────┤
│ T2: Disk (remaining experts)            │
│     - File descriptors                  │
│     - TODO: io_uring async reads        │
└─────────────────────────────────────────┘
```

### Slot Allocation Strategy

Each layer gets:
- `vram_slots_total / n_layers` slots in T0
- `ram_slots_total / n_layers` slots in T1
- LRU/Hybrid eviction policy
- Frequency tracking for hot experts

---

## What's NOT Done Yet

### Critical Missing Pieces

1. **GGUF Parsing Integration**
   - Need to read expert offsets from model file
   - Map expert IDs to file positions
   - Determine actual quantization types per part

2. **Disk Loading (T2)**
   - No actual file reading implemented
   - Need io_uring for async I/O
   - O_DIRECT alignment requirements

3. **RAM → VRAM Promotion**
   - Cache miss handling not implemented
   - Need to load from disk on miss
   - Need to promote hot experts to VRAM

4. **Integration with Inference**
   - Not hooked into qwen38_quant.c
   - Need to replace current weight loading
   - Need to route through expert cache

5. **Pinned Memory**
   - Currently using regular malloc
   - Should use CUDA-style pinned memory
   - Enables faster H2D transfers

### Estimated Work Remaining

| Task | Effort | Priority |
|------|--------|----------|
| GGUF parsing for expert locations | 2-3 hours | HIGH |
| Basic disk loading (synchronous) | 2-3 hours | HIGH |
| Integration with inference path | 3-4 hours | HIGH |
| Async disk I/O with io_uring | 4-6 hours | MEDIUM |
| RAM→VRAM promotion logic | 3-4 hours | MEDIUM |
| Pinned memory allocation | 1-2 hours | LOW |

**Total**: 15-22 hours of development work

---

## Next Immediate Steps

### Step 1: Parse GGUF for Expert Metadata

Need to extract from the model file:
```c
// For each layer (0-47):
for (int layer = 0; layer < 48; layer++) {
    // Get tensor metadata for gate_proj, up_proj, down_proj
    // These are MoE expert weights
    
    // Example tensor names:
    // "layers.0.feed_forward.experts.0.gate_proj.weight"
    // "layers.0.feed_forward.experts.0.up_proj.weight"
    // "layers.0.feed_forward.experts.0.down_proj.weight"
    
    // Store:
    // - Offset in file
    // - Size in bytes
    // - Quantization type
}
```

### Step 2: Implement Basic Disk Load

```c
// On cache miss:
if (slot < 0) {
    // Find empty or evict a slot
    int victim = find_victim_slot(cache, layer, prefer_vram);
    
    // Read expert from disk
    lseek(fd, expert_offset, SEEK_SET);
    read(fd, slot_buffer, expert_size);
    
    // Update tracking
    info->expert_slot[expert_id] = victim;
    info->slot_valid[victim] = 1;
}
```

### Step 3: Hook Into Inference

Replace current direct tensor access:
```c
// OLD:
const float *weights = (const float *)tensor->data;

// NEW:
Q38ExpertHandle handle;
q38_expert_cache_fetch(cache, layer, &expert_id, 1, &handle);
const uint8_t *weights = handle.parts[0];
```

---

## Performance Expectations

### Current State (Phase 1)
- Performance: ~1.4 tok/s
- Bottleneck: Every token reloads all weights
- No caching benefit

### After Basic Integration (Phase 2a)
- Expected: 3-5 tok/s
- Benefit: Cache hits avoid disk reads
- Still limited by synchronous I/O

### After Full Optimization (Phase 2b)
- Expected: 6-8 tok/s
- Benefits:
  - Async disk reads
  - VRAM residency for hot experts
  - Reduced latency

### Comparison to Target
- Target: 10-12 tok/s
- Gap: Need Phase 3 (batch matmul) + Phase 4 (prefetch)

---

## Files Created/Modified

```
qwen3.8-flash-next-in-c/
├── include/vulkan/
│   └── vulkan_expert_cache.h      # NEW: Expert cache API
├── src/vulkan/
│   └── vulkan_expert_cache.c      # NEW: Implementation
└── GNUmakefile                    # MODIFIED: Added to build
```

---

## Testing Status

✅ Compiles successfully
✅ Links with existing code
✅ No runtime crashes
❌ Not yet integrated with inference
❌ No cache hits/misses tracked
❌ No performance improvement yet

---

## Key Design Decisions

### 1. Unified Memory Architecture

Chose to use host-visible VRAM because:
- AMD 780M has unified memory
- No explicit H2D copies needed
- Simpler programming model
- Trade-off: Slightly slower than dedicated VRAM

### 2. Per-Layer Slot Pools

Each layer gets its own slots because:
- Experts are layer-specific
- Avoids cross-layer contention
- Simpler eviction logic
- Matches QwFNfer design

### 3. Hybrid Eviction Policy

Using LRU + LFU hybrid because:
- Routing patterns drift over time (need recency)
- Some experts always hot (need frequency)
- Matches QwFNfer measurements
- Better than pure LRU or LFU

### 4. Staged Implementation

Building incrementally because:
- Can test each piece independently
- Easier to debug
- Can measure impact at each stage
- Allows early feedback

---

## References

- `handoff.md`: Original implementation plan
- `QwFNfer/src/qwfn_expert_cache.h`: Reference implementation
- `PHASE1_COMPLETE.md`: Previous phase results

---

**Next Session**: Focus on GGUF parsing and basic disk loading