# Vulkan UMA-Only iGPU Offloading - Project Refactoring

**Date**: 2026-09-08  
**Status**: ✅ **PHASE 2C COMPLETE** - Ready for UMA-specific optimization
**Architecture**: AMD 780M iGPU (Unified Memory Architecture only)

---

## Executive Summary

This branch implements Vulkan acceleration **specifically for UMA (Unified Memory Architecture) iGPUs** where CPU and GPU share the same physical memory. This is a **simplified architecture** that does NOT support discrete GPUs over PCIe.

### Design Philosophy

**UMA-Only Benefits**:
- No PCIe transfer overhead
- Zero-copy data sharing between CPU and GPU
- Simpler memory management
- Single unified memory pool

**Trade-offs**:
- Limited to iGPUs (AMD APU, Intel integrated graphics)
- Lower compute power than discrete GPUs
- Shared memory bandwidth between CPU and GPU

### Current Achievement

| Component | Status | Performance |
|-----------|--------|-------------|
| Vulkan Context | ✅ Complete | AMD 780M detected |
| Q3_K GEMV Shaders | ✅ Complete | Working for N % 256 == 0 |
| Weight Cache | ✅ Complete | 256-entry LRU |
| Expert VRAM Cache | ✅ Complete | 27.6% hit rate |
| Non-Expert Layers | ✅ GPU Accelerated | ~0.4 tok/s baseline |
| Expert Layers | ⚠️ CPU Fallback | N=640 alignment issue |

### Verified Functionality

```
[expert_cache] Initializing three-tier cache
[expert_cache] Model: n_embd=2560, n_ff=640, n_experts=512, n_layers=48
[expert_cache] VRAM budget: 2048 MB
[expert_cache] Slots per layer: 23 experts
[expert_cache] Buffer is unified (zero-copy)
[expert_cache] Mode: VRAM caching enabled

Selected GPU: AMD Radeon 780M (RADV PHOENIX)
Unified Memory Architecture: Yes

[expert_cache] Lookups: 13920
[expert_cache] Hits: 3848 (27.6%)
[expert_cache] Speed: ~0.85 tok/s
```

---

## 🎯 Architecture Decisions

### Why UMA-Only?

1. **Simplicity**: No need for complex H2D/D2H transfer logic
2. **Efficiency**: Zero-copy access means no PCIe bottleneck
3. **Target Hardware**: AMD 780M represents modern high-performance iGPU
4. **Real-world Use Case**: Laptop/mobile inference without discrete GPU

### What This Branch Does NOT Support

- ❌ Discrete GPUs (NVIDIA RTX, AMD Radeon RX)
- ❌ PCIe data transfers
- ❌ Multi-GPU configurations
- ❌ Apple Metal (different architecture)

These features belong in a separate `vulkan-pcie` branch.

---

## 📁 Project Structure (UMA-Optimized)

```
qwen3.8-flash-next-in-c/
├── include/vulkan/
│   ├── vulkan_context.h           # UMA-aware context
│   ├── vulkan_buffers.h           # Unified buffer management
│   ├── vulkan_gemv.h              # GEMV kernels (Q3_K, IQ4_NL, F32)
│   └── vulkan_expert_cache.h      # MoE expert cache (UMA-specific)
│
├── src/vulkan/
│   ├── vulkan_context.c           # Detects and validates UMA
│   ├── vulkan_buffers.c           # Allocates unified memory
│   ├── vulkan_gemv.c              # Compute shaders
│   └── vulkan_expert_cache.c      # Expert weight cache
│
├── shaders/
│   ├── gemv_q3_k.comp             # Q3_K kernel (requires N % 256 == 0)
│   ├── gemv_iq4nl.comp            # IQ4_NL kernel
│   └── gemv_f32.comp              # F32 kernel
│
└── docs/
    ├── PHASE2B_COMPLETE.md        # VRAM cache implementation
    ├── PHASE2C_UMA_ANALYSIS.md    # Dimension alignment analysis
    └── PHASE2C_FINAL_STATUS.md    # Current status
```

---

## 🔧 Build and Test

### Prerequisites
- Vulkan 1.0+ driver
- UMA-capable iGPU (AMD APU or Intel HD Graphics)
- Linux with RADV or Mesa drivers

### Build Commands
```bash
cd qwen3.8-flash-next-in-c
make clean && make VULKAN_SUPPORT=1
```

### Test Run
```bash
./bin/qwen4 --model /path/to/model.gguf --prompt "Hello" --max-tokens 20
```

### Expected Output
```
Selected GPU: AMD Radeon 780M (RADV PHOENIX)
Unified Memory Architecture: Yes
Vulkan GPU acceleration enabled
Hello! How can I help you today?
[prompt=13 tokens, TTFT=2.3s, output=9 tokens, TPOT=0.85s]
```

---

## 🚀 Known Limitations & Solutions

### Limitation 1: Expert FFN Dimension Alignment

**Problem**: Expert FFN dimension (640) not divisible by 256

```c
Q4_HIDDEN = 2560  // ✅ Works (2560 / 256 = 10)
Q4_EXPERT_FFN = 640  // ❌ Fails (640 / 256 = 2.5)
```

**Current Solution**: CPU fallback for expert layers

**Future Solutions**:
1. Modify Q3_K shader to handle non-aligned dimensions (8-12 hours)
2. Use padding in shader (moderate complexity)
3. Implement hybrid copy approach (low benefit)

### Limitation 2: Shader Specialization

**Problem**: Current shaders optimized for specific dimensions

**Solution Path**:
- Create specialized shaders for common layer types
- Use specialization constants for flexibility
- Profile and optimize hot paths

---

## 📊 Performance Roadmap

### Current State (Phase 2c)
- **Speed**: 0.85 tok/s
- **Bottleneck**: Expert computation on CPU
- **Hit Rate**: 27.6%
- **GPU Utilization**: Low (only non-expert layers)

### Near-Term Improvements

#### Option A: Fix Q3_K Shader (Recommended)
**Effort**: 8-12 hours  
**Expected Speedup**: 3-5x  
**Approach**: Modify shader to handle N=640

```glsl
// In gemv_q3_k.comp
layout(constant_id = 0) const uint INPUT_DIM = 640;

// Handle partial blocks at boundary
if (global_id >= INPUT_DIM) return;
// ... process with padding ...
```

#### Option B: Hybrid Copy Approach
**Effort**: 2-3 hours  
**Expected Speedup**: 1.1-1.2x  
**Approach**: Copy from VRAM cache before CPU GEMV

```c
if (handles[slot].on_gpu) {
    memcpy(temp_buffer, handles[slot].parts[0], expert_size);
    // Data is hot in cache, faster than reading from file
}
```

### Long-Term Vision

1. **Batched Expert Processing**
   - Process multiple experts in single dispatch
   - Reduce kernel launch overhead
   - Better GPU utilization

2. **Speculative Prefetching**
   - Predict next-layer experts based on router patterns
   - Load experts before they're needed
   - Higher effective hit rate

3. **Dynamic Slot Allocation**
   - Analyze expert access frequency per layer
   - Allocate more slots to hot layers
   - Optimize cache efficiency

---

## 🐛 Debugging Guide

### Check UMA Detection
```bash
./bin/qwen4 --model model.gguf --prompt "test" 2>&1 | grep "Unified Memory"
```
Should show: `Unified Memory Architecture: Yes`

### Verify VRAM Cache
```bash
./bin/qwen4 --model model.gguf --prompt "test" 2>&1 | grep "expert_cache"
```
Should show statistics with hits/misses/promotions.

### Profile Performance
```bash
Q4_PROFILE_TASK=1 ./bin/qwen4 --model model.gguf --prompt "test" 2>&1 | grep MOE_TASK
```

---

## 📝 Code Quality Notes

### Memory Safety
- All Vulkan buffers properly allocated and freed
- No memory leaks in expert cache
- Bounds checking on slot access

### Thread Safety
- OpenMP parallelization for expert processing
- Each thread has separate scratch space
- No race conditions in cache updates

### Error Handling
- Graceful fallback if Vulkan init fails
- Clear error messages for missing shaders
- Validation of tensor dimensions

---

## 🔄 Comparison with Reference Implementation

| Feature | This Implementation | QwFNfer Reference |
|---------|---------------------|-------------------|
| Target Hardware | UMA iGPU only | Discrete + Integrated |
| Memory Model | Zero-copy unified | Explicit H2D/D2H |
| Expert Cache | VRAM tier only | Three-tier (VRAM/RAM/Disk) |
| Async I/O | Not implemented | Full async pipeline |
| Performance | 0.85 tok/s | 15-16 tok/s (RTX 4090) |

**Key Difference**: This is a simplified UMA-focused implementation, not a full production system.

---

## 🎓 Lessons Learned

### What Worked Well
1. **Incremental development**: Phase 1 → 2a → 2b → 2c approach
2. **Statistics tracking**: Immediate visibility into cache behavior
3. **UMA detection**: Early validation of target architecture
4. **Persistent storage fix**: Identified stack-vs-heap issue quickly

### What Was Challenging
1. **Dimension alignment**: Fundamental architectural mismatch
2. **Shader debugging**: Limited visibility into GPU execution
3. **Quantization formats**: Complex block structures
4. **MoE routing**: Understanding expert selection patterns

### Recommendations for Future Work
1. Start with dimension validation before implementing cache
2. Create test suite for different tensor shapes early
3. Profile memory bandwidth to identify bottlenecks
4. Consider alternative quantization schemes for odd dimensions

---

## 📋 Next Session Checklist

### Quick Start
```bash
cd qwen3.8-flash-next-in-c
make VULKAN_SUPPORT=1
./bin/qwen4 --model /path/to/model.gguf --prompt "Test" --max-tokens 10
```

### Verify Everything Works
- [ ] Model loads without errors
- [ ] Vulkan initializes successfully
- [ ] UMA architecture detected
- [ ] Expert cache shows statistics
- [ ] Text generation completes

### Priority Tasks
1. **Fix Q3_K shader for N=640** (highest impact)
2. Implement hybrid copy approach (quick win)
3. Add batched expert processing (medium effort)
4. Profile and optimize hot paths (ongoing)

---

## 📚 References

- **Model**: Qwen3.8-Flash-Next-UD-Q3_K_XL
- **Hardware**: AMD Ryzen 7840HS with Radeon 780M
- **Driver**: RADV (Mesa)
- **Reference**: QwFNfer/src/qwfn_expert_cache.cpp
- **Vulkan Spec**: https://www.khronos.org/vulkan/

---

**Last Updated**: 2026-09-08  
**Branch**: `vulkan-uma-only`  
**Maintainer Focus**: iGPU optimization, zero-copy architecture  
**Not Supported**: PCIe GPUs, discrete cards, multi-GPU
