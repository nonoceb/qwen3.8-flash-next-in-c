# Next Steps for Testing Phase 2c

## Quick Test Commands

### 1. Build
```bash
cd qwen3.8-flash-next-in-c
make clean && make VULKAN_SUPPORT=1
```

### 2. Basic Functionality Test
```bash
./bin/qwen4 --model /mnt/DATA2T/_IA_Models/Qwen3.8Next/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf \
    --prompt "Hello, how are you?" \
    --max-tokens 20
```

**What to check**:
- Does it compile and run without crashes?
- Do you see `[expert_cache] Mode: VRAM caching enabled (Phase 2b)`?
- What's the hit rate in the statistics at the end?
- What's the tokens/second (TPOT) value?

### 3. Performance Benchmark
```bash
./bin/qwen4 --model /mnt/DATA2T/_IA_Models/Qwen3.8Next/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf \
    --prompt "Write a detailed explanation of how neural networks learn" \
    --max-tokens 100
```

**Expected results**:
- First token: ~2-4 seconds TTFT (cold cache)
- Tokens 1-10: Gradual speedup as cache warms
- Tokens 10+: Should stabilize around 3-8 tok/s
- Hit rate should climb to 40-60%

### 4. Check Vulkan GEMV is Actually Being Used

Add debug output temporarily:

```bash
# In src/qwen4/qwen4_model.c, around line 1620, add:
if (use_vulkan[task]) {
    static int vulkan_count = 0;
    if (vulkan_count < 5) {
        fprintf(stderr, "[DEBUG] Using Vulkan GEMV for expert %d\\n", task);
        vulkan_count++;
    }
}
```

Then rebuild and test to verify the Vulkan path is being taken.

## Potential Issues to Watch For

### Issue 1: Crash or Segfault
- **Cause**: Likely tensor structure not set up correctly
- **Debug**: Add print statements to see which expert causes crash
- **Fix**: Check tensor dimensions and data pointers

### Issue 2: No Performance Improvement
- **Cause**: Vulkan GEMV might be failing silently and falling back to CPU
- **Debug**: Check if `q38_vulkan_gemv_q3_k` returns success
- **Fix**: Verify shader is compiled and GPU can access VRAM buffer

### Issue 3: Incorrect Output
- **Cause**: Data corruption during promotion or wrong tensor shape
- **Debug**: Compare output with CPU-only version
- **Fix**: Verify memcpy offsets and tensor shapes match

### Issue 4: Low Hit Rate (<20%)
- **Cause**: Too much eviction thrashing
- **Solution**: Increase VRAM budget to 3GB or reduce slots per layer

## Success Criteria

✅ **Minimum viable**: Code runs without crashes, produces coherent output
✅ **Good**: Hit rate >30%, TPOT <500ms
✅ **Target**: Hit rate >50%, TPOT <200ms (5+ tok/s)
✅ **Stretch**: Hit rate >70%, TPOT <100ms (10+ tok/s)

## After Testing

Report back with:
1. Compilation status (success/errors)
2. Runtime behavior (crashes/output quality)
3. Performance metrics (TTFT, TPOT, hit rate)
4. Any error messages or warnings

This will help determine if we need debugging or can proceed to optimization.
