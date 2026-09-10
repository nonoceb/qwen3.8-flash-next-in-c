#!/bin/bash

MODEL="/mnt/DATA2T/_IA_Models/Qwen3.8Next/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf"

echo "========================================="
echo "VULKAN DEBUG SESSION"
echo "========================================="
echo ""

# Test 1: Vulkan unit tests
echo "1. Running Vulkan unit tests..."
echo ""
echo "F32 GEMV test:"
./bin/test_vulkan_gemv 2>&1 | grep -E "PASS|FAIL|Results"

echo ""
echo "IQ4_NL GEMV test:"
./bin/test_vulkan_iq4nl 2>&1 | grep -E "PASS|FAIL|Results"

echo ""
echo "========================================="
echo "2. Testing actual inference with Vulkan"
echo "========================================="
echo ""

# Test with very short prompt
echo "Test: Short prompt (should use CPU for IQ4_NL, GPU for F32)"
./bin/qwen4 --model "$MODEL" --prompt "Hi" --max-tokens 5 2>&1

echo ""
echo "========================================="
echo "3. Checking dmesg for GPU errors"
echo "========================================="
dmesg | tail -10

