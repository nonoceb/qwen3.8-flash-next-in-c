
# Qwen3.8-Flash-Next Real Inference Benchmark Results

## Test Configuration
- **Model**: Qwen3.8-Flash-Next-UD-Q3_K_XL (85GB total, 3 shards)
- **Hardware**: AMD system with Radeon 780M iGPU (RDNA3, UMA)
- **Build**: AVX-512 + Vulkan support enabled

## Benchmark Results

### Test 1: Creative Writing (100 tokens)

| Mode | TTFT | TPOT | Total Time | Tokens/sec |
|------|------|------|------------|------------|
| **Vulkan GPU** | 7.022s | 0.506s | 58.3s | **1.98 tok/s** |
| **CPU Only** | 7.917s | 0.448s | 53.4s | **2.23 tok/s** |

### Test 2: Explanation Task (50 tokens)

| Mode | TTFT | TPOT | Total Time | Tokens/sec |
|------|------|------|------------|------------|
| **Vulkan GPU** | 5.964s | 0.542s | 33.6s | **1.85 tok/s** |
| **CPU Only** | 3.119s | 0.248s | 16.4s | **4.03 tok/s** |

## Analysis

### 🔴 Issue Identified: Vulkan Slower Than CPU

The Vulkan GPU implementation is currently **slower** than CPU-only inference:

1. **TPOT (Time Per Output Token)**:
   - CPU: 0.248-0.448s per token
   - Vulkan: 0.506-0.542s per token
   - **Vulkan is 21-118% slower**

2. **Root Cause**:
   - The model uses **Q3_K quantization** (not IQ4_NL or F32)
   - Vulkan kernels only support F32 and IQ4_NL formats
   - Q3_K operations fall back to CPU (AVX-512 optimized)
   - Vulkan overhead (context switching, buffer management) adds latency

3. **TTFT (Time To First Token)**:
   - Vulkan shows higher TTFT due to pipeline initialization overhead
   - CPU benefits from immediate execution without GPU dispatch

### ✅ What's Working

1. **Vulkan Infrastructure**: Successfully initialized and running
2. **Weight Caching**: Operational with 256 entries
3. **F32 Kernels**: Working correctly on GPU
4. **UMA Detection**: Properly identified shared memory architecture

### ❌ Performance Bottlenecks

1. **Missing Q3_K Vulkan Kernel**: Model uses Q3_K but only F32/IQ4_NL have GPU support
2. **Kernel Dispatch Overhead**: Small matrices fall back to CPU anyway
3. **No Async Execution**: Synchronous GPU waits add latency
4. **IQ4_NL Disabled**: Commented out due to "GPU page faults"

## Recommendations

### Immediate Actions (High Impact)

1. **Implement Q3_K Vulkan Kernel**
   - This is the actual quantization format used by the model
   - Would enable true GPU acceleration for all layers
   
2. **Fix IQ4_NL Page Faults**
   - Debug the commented-out IQ4_NL path
   - Could help if model has any IQ4_NL layers

3. **Add Batch Processing**
   - Process multiple tokens during prefill phase
   - Use GEMM instead of GEMV for batch operations

### Medium-Term Optimizations

1. **Async Execution**: Overlap CPU/GPU work
2. **Pipeline Caching**: Pre-create pipelines at startup
3. **Memory Pinning**: Optimize H2D transfers in UMA

### Expected Performance After Fixes

With Q3_K Vulkan kernel implemented:
- Target: **10-15 tok/s** (based on handoff.md projections)
- Current CPU baseline: **2.2-4.0 tok/s**
- Potential speedup: **3-5x**

## Conclusion

The Vulkan infrastructure is functional but not yet beneficial for this specific model because:
1. The model uses Q3_K quantization (no Vulkan support)
2. Operations fall back to highly-optimized AVX-512 CPU code
3. Vulkan overhead outweighs benefits without proper kernel support

**Priority**: Implement Q3_K Vulkan compute shader to unlock GPU acceleration.
