# Phase 1 Complete: Q3_K Vulkan Integration

**Date**: 2026-09-08
**Status**: ✅ SUCCESS - GPU acceleration working for Q3_K quantized models

---

## What Was Accomplished

### 1. Fixed Q3_K Vulkan Implementation

**Problem**: The Q3_K shader existed but the C implementation was written for a different data structure.

**Solution**: Rewrote `src/vulkan/vulkan_gemv.c` lines 1156-1420 to:
- Use fixed pipeline arrays indexed by `Q38_VK_PIPELINE_GEMV_Q3_K`
- Match the F32/IQ4_NL pattern for buffer creation and command submission
- Properly handle Q3_K block padding (110 bytes → 112 bytes)
- Add correct descriptor set management

### 2. Added Header Declarations

**File**: `include/vulkan/vulkan_gemv.h`
- Added `Q38_VK_PIPELINE_GEMV_Q3_K` to pipeline enum
- Added function declaration for `q38_vulkan_gemv_q3_k()`

### 3. Enabled Call Site

**File**: `src/qwen38/qwen38_quant.c`
- Added extern declaration for `q38_vulkan_gemv_q3_k`
- Added Q3_K path in GPU dispatch logic

---

## Performance Results

### Test Configuration
- **Model**: Qwen3.8-Flash-Next-UD-Q3_K_XL (85GB, 3 shards)
- **GPU**: AMD Radeon 780M (RADV PHOENIX)
- **Architecture**: Unified Memory (no explicit H2D/D2H copies)

### Benchmarks

| Test | Prompt | Tokens | TTFT | TPOT | Speed |
|------|--------|--------|------|------|-------|
| Story | "Write a story about a robot" | 30 | 5.8s | 0.704s | ~1.42 tok/s |
| Quantum | "Explain quantum computing" | 50 | 6.9s | 0.718s | ~1.39 tok/s |
| Hello | "Hello" | 9 | 3.3s | 0.745s | ~1.34 tok/s |

**Average**: ~1.4 tokens/second

### Comparison

| Mode | Performance | Notes |
|------|-------------|-------|
| CPU-only (AVX-512) | 2.2-4.0 tok/s | Baseline from SESSION_SUMMARY.md |
| Vulkan Q3_K (Phase 1) | 1.3-1.4 tok/s | Working but slower |
| Target (Phase 4) | 10-12 tok/s | After three-tier cache + prefetch |

---

## Why Slower Than CPU?

The GPU is currently **slower** than CPU because:

1. **No Expert Caching**: Every token loads ~480 expert weight blocks
   - Model has 512 experts × 48 layers = 55.8 GB total
   - Each token touches only ~1.05 GB
   - Without caching, same weights uploaded repeatedly

2. **Sequential Processing**: Processing 10 MoE experts one at a time
   - 10 separate kernel dispatches per layer
   - High overhead on iGPU

3. **No Overlap**: Synchronous execution
   - No compute-I/O overlap
   - No speculative prefetch

4. **iGPU Limitations**: AMD 780M is integrated graphics
   - Shared memory bandwidth with CPU
   - Lower compute throughput than discrete GPU

---

## Next Steps: Phase 2 - Three-Tier Cache

To achieve target performance (10-12 tok/s), implement:

### T0: VRAM Tier (~4.7 GB)
- Store hottest ~2,100 experts
- Direct GPU access via batch matmul
- Target: 57-86% hit rate

### T1: RAM Tier (~19 GB pinned)
- Warm experts in host memory
- Async DMA transfers (94µs vs 135µs pageable)
- Target: 91-97% overall hit rate

### T2: NVMe Tier (remaining)
- Cold experts on disk
- io_uring O_DIRECT reads
- 23× bandwidth vs demand paging

### Expected Impact
- **After Phase 2**: 6-8 tok/s (cache reduces I/O)
- **After Phase 3**: 8-10 tok/s (batch matmul)
- **After Phase 4**: 10-12 tok/s (speculative prefetch)

---

## Files Modified

```
qwen3.8-flash-next-in-c/
├── include/vulkan/vulkan_gemv.h      # Added Q3_K pipeline + declaration
├── src/vulkan/vulkan_gemv.c          # Rewrote q38_vulkan_gemv_q3_k()
└── src/qwen38/qwen38_quant.c         # Enabled Q3_K GPU path
```

## Build Command

```bash
cd qwen3.8-flash-next-in-c
make clean && make VULKAN_SUPPORT=1
```

## Run Command

```bash
./bin/qwen4 --model /path/to/Q3_K.gguf --prompt "Test" --max-tokens 50
```

---

## Technical Details

### Q3_K Quantization Format
- Block size: 256 elements
- Storage: 110 bytes per block (3.44 bits per element)
- Structure: 32 high bits + 64 low bits + 12 scales + 2-byte scale
- Padding: 112 bytes (28 uints) for GPU alignment

### Shader Architecture
- File: `shaders/gemv_q3_k.comp` (182 lines)
- Compiled: `shaders/spv/gemv_q3_k.spv` (12KB)
- Workgroups: One per matrix row (M workgroups)
- Local size: 64 threads
- Push constants: M, N dimensions

### Buffer Layout
- Binding 0: Padded weights (M × blocks_per_row × 112 bytes)
- Binding 1: Input vector (N floats)
- Binding 2: Output vector (M floats)

---

**Ready for Phase 2**: Three-tier expert cache implementation