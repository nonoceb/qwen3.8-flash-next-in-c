# CPU Inference Benchmark Results

## Status: ✅ ALL TESTS PASSING

### Issue Fixed
**Root Cause**: Broken AVX-512 optimization for IQ4_NL quantization (lines 3105-3200 in `qwen38_quant.c`)

**Bug Details**: The `_mm512_shuffle_epi8()` operation was incorrectly using 16-bit indices with a 128-bit value table, causing corrupted output.

**Solution**: Disabled the broken AVX-512 IQ4_NL path. The code now falls back to the working AVX2 implementation.

---

## Performance Results (CPU-Only, AVX-512 Enabled)

### Test Configuration
- **Model**: Qwen3.8-Flash-Next-UD-Q3_K_XL (85GB total, split across 3 shards)
- **Quantization**: Mix of Q8_0, IQ3_XXS, IQ4_NL, Q3_K_XL
- **Platform**: AMD with AVX-512F, AVX-512BW, AVX-512VL support
- **Vulkan**: Disabled for CPU-only testing

### Benchmark Results (10 iterations)

| Metric | Average | Range |
|--------|---------|-------|
| TTFT | 3.26s | 2.63s - 3.46s |
| TPOT | 0.29s | 0.26s - 0.35s |
| **Speed** | **~3.4 tokens/s** | 2.9 - 3.8 tokens/s |

### Output Quality

All tests produce coherent, grammatically correct English:

```
Test 1: "Hello! I'm ready for Test 1. Please provide the question..."
Test 2: "Please provide the question, problem, or task..."
Test 3: "Please provide the question, problem, or task..."
```

**Previous broken output** (before fix):
```
"Hello 13,are 13361599999999997999999"
"The \"Winnetted and Refined, Sem"
```

---

## What Was Fixed

### 1. Multi-Shard GGUF Loading ✅
- Successfully loads all 3 shards (11MB + 47GB + 38GB)
- All 1224 tensors correctly mapped
- Proper tensor lookup across shards

### 2. AVX-512 IQ4_NL Bug ✅
- **Disabled broken 16-row AVX-512 path**
- Falls back to working 8-row AVX2 implementation
- Preserves correctness while maintaining performance

### 3. Verified Working Optimizations
- ✅ AVX-512 F32 GEMV (16 rows)
- ✅ AVX-512 Q8_0 GEMV (16 rows)
- ❌ AVX-512 IQ4_NL (disabled, uses AVX2 fallback)
- ✅ AVX2 IQ4_NL (8 rows)

---

## Remaining Work

### Vulkan GPU Offload (Not Yet Fixed)
The Vulkan implementation has a separate bug:
```c
// BUG in q38_vulkan_gemv_iq4nl():
q38_vulkan_buffer_destroy(ctx, &codebook_buffer);  // Destroys while GPU reads
```

This causes GPU page faults and crashes. Needs to be fixed separately by making the codebook buffer persistent.

### Future Optimization Opportunities
1. Fix AVX-512 IQ4_NL path for potential 2x speedup on that kernel
2. Implement weight caching for Vulkan (expected 2-3x speedup)
3. Add async execution for CPU/GPU overlap

---

## Files Modified

- `src/qwen38/qwen38_quant.c`: Disabled broken AVX-512 IQ4_NL path (lines 3105-3200)

## Build Commands

```bash
cd qwen3.8-flash-next-in-c
make clean
make VULKAN_SUPPORT=1

# Run CPU inference
Q38_DISABLE_VULKAN=1 ./bin/qwen4 --model <model.gguf> --prompt "Your prompt" --max-tokens 50
```

---

**Date**: 2026-09-08
**Status**: CPU inference fully functional with AVX-512 optimizations
