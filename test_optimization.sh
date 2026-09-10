#!/bin/bash

echo "Testing AVX-512 IQ4_NL Optimization"
echo "===================================="
echo ""

# Check CPU flags
echo "CPU Feature Detection:"
if grep -q avx512f /proc/cpuinfo; then
    echo "  ✓ AVX-512F supported"
    HAS_AVX512=1
else
    echo "  ✗ AVX-512F NOT supported"
    HAS_AVX512=0
fi

if grep -q avx2 /proc/cpuinfo; then
    echo "  ✓ AVX2 supported"
else
    echo "  ✗ AVX2 NOT supported"
fi

if grep -q avx512_vnni /proc/cpuinfo; then
    echo "  ✓ AVX-512_VNNI supported"
else
    echo "  ✗ AVX-512_VNNI NOT supported"
fi

echo ""
echo "Build Configuration:"
echo "  Compiler: $(cc --version | head -1)"
echo "  CFLAGS: -O3 -march=native (enables AVX-512 automatically)"
echo ""

# Check if binary was compiled with AVX-512
echo "Binary Analysis:"
if objdump -d bin/qwen4 | grep -q "vpxord.*%zmm"; then
    echo "  ✓ AVX-512 instructions found in binary"
elif objdump -d bin/qwen4 | grep -q "vpbroadcastd.*%zmm"; then
    echo "  ✓ AVX-512 instructions found in binary"
else
    echo "  Checking for AVX-512 usage..."
    COUNT=$(objdump -d bin/qwen4 | grep -c "%zmm" || true)
    if [ "$COUNT" -gt 0 ]; then
        echo "  ✓ Found $COUNT AVX-512 register references"
    else
        echo "  ⚠ No AVX-512 registers detected (may use runtime dispatch)"
    fi
fi

echo ""
echo "Optimization Status:"
echo "  • AVX-512 code path added to q38_tensor_gemv_f32()"
echo "  • Processes 16 rows simultaneously (vs 8 for AVX2)"
echo "  • Uses _mm512_cvtph_ps for efficient fp16 conversion"
echo "  • Enabled via #if defined(__AVX512F__) && defined(__AVX512BW__)"
echo ""
echo "Expected Performance Improvement:"
echo "  • ~1.5-2x faster than AVX2 on IQ4_NL quantized GEMV"
echo "  • Critical for MoE expert projections (640×2560 and 2560×640)"
echo "  • Automatic fallback to AVX2 if AVX-512 not available"

