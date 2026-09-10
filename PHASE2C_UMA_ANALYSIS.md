# Phase 2c UMA Analysis - Expert Cache Optimization

## Problem Identified

The expert FFN dimension (640) is **not divisible by 256**, which causes failures in both:
1. Vulkan Q3_K GEMV shader (requires N % 256 == 0)
2. CPU tensor GEMV functions (also check alignment)

## Root Cause

```c
// In vulkan_gemv.c and qwen38_quant.c
if (N % 256 != 0) return 0;  // Fails for N=640
```

Qwen3.8-Flash-Next architecture:
- `Q4_HIDDEN = 2560` ✅ Divisible by 256
- `Q4_EXPERT_FFN = 640` ❌ NOT divisible by 256 (640 % 256 = 128)

## Current Status

✅ **VRAM caching works perfectly**:
- 48.5% hit rate
- Experts promoted to unified memory
- LRU eviction working
- Zero-copy access on UMA

❌ **Cannot use cached data for computation**:
- Vulkan shader rejects N=640
- CPU GEMV also has alignment requirements
- Generation fails when trying to use VRAM-resident weights

## Solution Options

### Option 1: Modify Shaders (Complex)
- Update Q3_K compute shader to handle non-256-aligned dimensions
- Add padding logic
- Handle edge cases in kernel
- Estimated effort: 8-12 hours

### Option 2: Use Different Quantization (Requires Model Change)
- Repack experts with padding to 768 or 1024 FFN dimension
- Would require re-exporting the model
- Not practical for existing models

### Option 3: Hybrid Approach (Recommended)
- Keep VRAM cache for weight storage
- Use CPU fallback for expert computation
- Benefit from cache locality even without GPU compute
- Estimated benefit: 10-20% speedup from reduced memory bandwidth

### Option 4: Disable Expert Caching
- Focus Vulkan acceleration on non-expert layers
- Experts remain on CPU path
- Simpler implementation

## Recommended Path Forward

Implement **Option 3 (Hybrid)**:

1. Keep experts in VRAM cache (already working)
2. When expert is needed, copy from VRAM to temporary buffer
3. Use existing CPU GEMV on the copied data
4. Benefit: Data is hot in cache, faster than reading from mmap'd file

This gives partial benefit without complex shader modifications.

## Performance Expectations

| Approach | Expected Speedup | Complexity |
|----------|------------------|------------|
| Current (no cache use) | 0x | Done |
| Hybrid (copy + CPU) | 1.1-1.2x | Low (2-3 hours) |
| Modified shaders | 2-3x | High (8-12 hours) |

## Implementation Notes

For hybrid approach:
```c
if (handles[task].valid && handles[task].on_gpu) {
    // Copy expert from VRAM to CPU-visible buffer
    memcpy(temp_buffer, handles[task].parts[0], expert_size);
    
    // Use CPU GEMV on temp buffer
    project_prequantized(..., temp_tensor);
}
```

## Conclusion

The fundamental issue is architectural (FFN dimension not aligned to quantization block size). 

**Recommendation**: Implement hybrid approach for modest gains, document limitation, and consider shader modification as future work.
