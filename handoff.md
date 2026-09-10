# Vulkan Q3_K Implementation Handoff Document
## Project: Optimized MoE Inference on AMD 780M

**Status**: ✅ **PHASE 1 COMPLETE + CRITICAL FIX** - Ready for Phase 2 integration and optimization

**Last Updated**: 2026-09-08

---

## Executive Summary

Successfully implemented GPU-accelerated Q3_K quantization support for Qwen3.8-Flash-Next inference on AMD Radeon 780M (RDNA3, Unified Memory Architecture). Fixed critical architectural mismatch with partial block handling. Foundation laid for three-tier expert cache optimization.

### Current Performance

| Metric | Value | Notes |
|--------|-------|-------|
| Baseline (CPU AVX-512) | 2.2-4.0 tok/s | Reference point |
| Vulkan Q3_K (Phase 1) | 1.5-2.0 tok/s | Working but unoptimized |
| Target (Phase 2 complete) | 6-8 tok/s | With expert cache |
| Ultimate target | 10-12 tok/s | Full optimization |

### Key Achievement

✅ **GPU compute working correctly** for all tensor dimensions including non-256-aligned sizes (critical fix)

---

## What's Been Completed

### Phase 1: Q3_K Vulkan Kernel ✅

#### 1.1 Shader Implementation
- **File**: `shaders/gemv_q3_k.comp` (182 lines)
- **Compiled**: `shaders/spv/gemv_q3_k.spv` (12KB)
- **Architecture**: One workgroup per matrix row, 64 threads per workgroup
- **Features**:
  - Complete Q3_K dequantization (110 bytes per 256 elements)
  - Padded to 112 bytes for alignment
  - Parallel reduction within workgroup
  - **CRITICAL**: Partial block handling via bounds checking

#### 1.2 C Integration
- **Files modified**:
  - `include/vulkan/vulkan_gemv.h` - Added Q3_K pipeline enum and function declaration
  - `src/vulkan/vulkan_gemv.c` - Implemented `q38_vulkan_gemv_q3_k()` function (~300 lines)
  - `src/qwen38/qwen38_quant.c` - Enabled Q3_K GPU dispatch path

#### 1.3 Critical Fix: Partial Block Handling
**Problem discovered**: Expert FFN dimension is 640, which is not divisible by 256 (Q3_K block size).
- 640 % 256 = 128 remainder
- Original shader would truncate or fail

**Solution implemented**:
```glsl
// Ceiling division for block count
const uint blocks_per_row = (pc.N + 255u) / 256u;

// Bounds checking in inner loop
for (uint lane = 0u; lane < 32u; lane++) {
    const uint elem_idx = base_elem + h * 128u + field * 32u + lane;
    if (elem_idx >= pc.N) continue;  // Skip out-of-bounds
    // ... process element ...
}
```

**Result**: All dimension sizes now supported correctly.

### Phase 2 Foundation: Three-Tier Expert Cache 🔨

#### 2.1 Infrastructure Created
- **Header**: `include/vulkan/vulkan_expert_cache.h` (210 lines)
  - Defined three-tier architecture (VRAM/RAM/Disk)
  - `Q38ExpertHandle`, `Q38ExpertCache`, `Q38ExpertLayerInfo` structures
  - Split fetch API (begin/end) for compute-I/O overlap
  
- **Implementation**: `src/vulkan/vulkan_expert_cache.c` (450 lines)
  - `q38_expert_cache_init_from_tensors()` - Initialize with pre-loaded model weights
  - `q38_expert_cache_fetch()` - Blocking expert fetch
  - `q38_expert_cache_fetch_begin/end()` - Non-blocking split fetch
  - Statistics tracking built-in

#### 2.2 Design Decisions
1. **Use pre-loaded tensors**: Leverage existing GGUF memory mapping instead of separate disk I/O
2. **Global cache instance**: Simplified testing without modifying complex Q4Model structure
3. **Incremental approach**: Start with tracking only, add actual caching logic progressively

#### 2.3 Current Limitations
- ❌ Not yet integrated into inference path
- ❌ No actual caching/promotion logic implemented
- ❌ Statistics tracked but not meaningful yet
- ❌ VRAM tier not utilized

---

## Quick Start Guide

### Build and Test

```bash
cd qwen3.8-flash-next-in-c

# Compile shaders
cd shaders && make && cd ..

# Build with Vulkan support
make VULKAN_SUPPORT=1

# Run test
./bin/qwen4 --model /path/to/Q3_K.gguf --prompt "Test" --max-tokens 20
```

### Expected Output

```
[weight_cache] Initialized with 256 entries, policy=2
[gemv] Weight caching enabled
Selected GPU: AMD Radeon 780M (RADV PHOENIX)
Unified Memory Architecture: Yes
Vulkan GEMV initialized (pipelines will be created on-demand)
Vulkan GPU acceleration enabled
Hello! I'm here and ready to help.
[prompt=13 tokens, cached=0, TTFT=2.298s, output=10 tokens, TPOT=0.526s]
```

---

## Next Steps: Phase 2 Integration

### Step 1: Initialize Expert Cache (2-3 hours)

**Location**: `src/qwen4/qwen4_model.c` in `q4_model_open_gguf()` after line ~2300

```c
#ifdef VULKAN_SUPPORT
if (vulkan_enabled) {
    const Q38GGUFTensor *gate_tensors[Q4_LAYERS];
    const Q38GGUFTensor *up_tensors[Q4_LAYERS];
    const Q38GGUFTensor *down_tensors[Q4_LAYERS];
    
    for (uint32_t i = 0; i < Q4_LAYERS; i++) {
        gate_tensors[i] = model->layer[i].expert_gate;
        up_tensors[i] = model->layer[i].expert_up;
        down_tensors[i] = model->layer[i].expert_down;
    }
    
    void *ctx = q38_vulkan_get_context();
    q38_init_global_expert_cache(ctx, Q4_LAYERS, Q4_EXPERTS,
        Q4_HIDDEN, Q4_EXPERT_FFN,
        gate_tensors, up_tensors, down_tensors);
}
#endif
```

### Step 2: Hook Into Forward Pass (2-3 hours)

**Location**: `src/qwen4/qwen4_model.c` in `moe()` function around line 1490

Replace direct tensor access with cache lookups. See full details in handoff.md.

### Step 3: Implement Caching Logic (3-4 hours)

Add frequency tracking, hot/cold classification, and LRU eviction.

### Step 4: Add VRAM Tier (4-5 hours)

Allocate GPU buffer for hot experts, implement promotion logic.

---

## Testing Checklist

- [x] Shaders compile without errors
- [x] Project builds with VULKAN_SUPPORT=1
- [x] Basic inference produces coherent output
- [x] Long generation runs without crashes
- [x] Partial blocks handled correctly (N=640 works)
- [ ] Expert cache initialized during model load
- [ ] Cache hooked into forward pass
- [ ] Statistics printed on shutdown
- [ ] Performance improvement measured

---

## File Structure

```
qwen3.8-flash-next-in-c/
├── shaders/gemv_q3_k.comp          # Q3_K compute shader ✓
├── include/vulkan/
│   ├── vulkan_gemv.h               # GEMV API ✓
│   └── vulkan_expert_cache.h       # Cache API ✓
├── src/vulkan/
│   ├── vulkan_gemv.c               # Q3_K implementation ✓
│   └── vulkan_expert_cache.c       # Cache foundation ✓
├── src/qwen4/qwen4_model.c         # MODIFY HERE for integration
└── docs/
    ├── handoff.md                  # This document
    ├── PHASE1_COMPLETE.md
    ├── PHASE2_FOUNDATION_COMPLETE.md
    └── Q3K_PARTIAL_BLOCK_FIX.md
```

---

## Performance Roadmap

| Phase | Status | Speed | Key Feature |
|-------|--------|-------|-------------|
| 1 | ✅ Done | 1.5-2 tok/s | Q3_K GPU kernel |
| 2a | 🔨 Next | 1.5-2 tok/s | Cache integration |
| 2b | ❌ TODO | 3-5 tok/s | VRAM tier |
| 2c | ❌ TODO | 6-8 tok/s | Full cache |
| 3 | ❌ TODO | 8-10 tok/s | Batch matmul |
| 4 | ❌ TODO | 10-12 tok/s | Speculative prefetch |

---

## Known Issues

1. **Slower than CPU**: Expected - no caching benefit yet. Will improve with Phase 2.
2. **No disk I/O**: Using memory-mapped data. Async I/O is future enhancement.
3. **Weight padding**: 110→112 bytes per block. Acceptable 1.8% overhead for alignment.

---

## References

- **QwFNfer CUDA**: `../QwFNfer/src/qwfn_expert_cache.*` (reference implementation)
- **llama.cpp Vulkan**: https://github.com/ggerganov/llama.cpp
- **Vulkan Guide**: https://github.com/KhronosGroup/Vulkan-Guide

---

**Ready to continue**: Foundation complete, compiles cleanly, tests pass. Focus on Steps 1-2 above.

**Estimated time to Phase 2a**: 4-6 hours
**Estimated time to Phase 2b**: 8-12 hours additional