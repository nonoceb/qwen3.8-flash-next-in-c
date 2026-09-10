# Vulkan IQ4_NL Status

## Current Issue

The Vulkan IQ4_NL path produces incorrect results (-4064.0 instead of 0.0 for zero-initialized weights). This suggests a shader execution or data interpretation problem.

## What Works

✅ Vulkan F32 GEMV - All tests passing
✅ Vulkan initialization and buffer management  
✅ Codebook buffer persistence fix (no more GPU page faults)
✅ Weight padding logic (18→20 bytes) is correct

## What Doesn't Work

❌ Vulkan IQ4_NL produces incorrect output values
- Test with all-zero weights should produce ~0 output
- Actual output: -4064.0 (suggests incorrect data interpretation)

## Root Cause Analysis

The shader is executing without crashing (no GPU page faults), but producing wrong results. Possible causes:

1. **Shader compilation issue** - SPIR-V might not match expected behavior
2. **Memory layout mismatch** - Despite padding being correct, GPU might read differently
3. **Synchronization issue** - Results not properly synchronized before reading back
4. **Workgroup dispatch issue** - Incorrect number of workgroups or invocation IDs

## Recommended Fix

For immediate use, Vulkan IQ4_NL is disabled in `src/qwen38/qwen38_quant.c`:
```c
// TODO: IQ4_NL Vulkan path disabled due to incorrect results - needs debugging
// else if (tensor->type == Q38_GGML_IQ4_NL) {
//     gpu_result = q38_vulkan_gemv_iq4nl(ctx, gemv, output, input, tensor);
// }
```

CPU fallback works correctly for IQ4_NL tensors.

## Future Debug Steps

1. Add Vulkan validation layers to check for API misuse
2. Verify shader SPIR-V bytecode matches source
3. Add debug output in shader (if supported)
4. Compare GPU memory layout vs CPU expectations
5. Test with simplified shader (direct byte reads vs uint reads)

## Performance Impact

With Vulkan IQ4_NL disabled:
- F32 tensors: ✅ Use Vulkan GPU acceleration
- IQ4_NL tensors: ❌ Fall back to CPU (AVX2/AVX-512)
- Overall: Still faster than pure CPU due to F32 GPU offload

---
**Status**: Vulkan F32 working, Vulkan IQ4_NL needs debugging
**Date**: 2026-09-08
