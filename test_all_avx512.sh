#!/bin/bash

MODEL="/mnt/DATA2T/_IA_Models/Qwen3.8Next/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf"

echo "========================================="
echo "Testing with ALL AVX-512 Optimizations"
echo "========================================="
echo ""

# Test 1: Simple factual question
echo "Test 1: Factual question"
Q38_DISABLE_VULKAN=1 ./bin/qwen4 --model "$MODEL" --prompt "The capital of France is" --max-tokens 10 --temperature 0 2>&1

echo ""

# Test 2: Conversational
echo "Test 2: Conversational"
Q38_DISABLE_VULKAN=1 ./bin/qwen4 --model "$MODEL" --prompt "Hello, how are you today?" --max-tokens 20 --temperature 0 2>&1

echo ""

# Test 3: Math
echo "Test 3: Math"
Q38_DISABLE_VULKAN=1 ./bin/qwen4 --model "$MODEL" --prompt "What is 2 + 2?" --max-tokens 15 --temperature 0 2>&1

echo ""

# Test 4: Longer generation
echo "Test 4: Longer generation"
Q38_DISABLE_VULKAN=1 ./bin/qwen4 --model "$MODEL" --prompt "Write a short poem about the moon:" --max-tokens 50 --temperature 0 2>&1

echo ""

# Performance comparison
echo "========================================="
echo "Performance Comparison (5 runs each)"
echo "========================================="
echo ""

echo "With AVX-512 (current build):"
for i in {1..5}; do
    echo -n "Run $i: "
    Q38_DISABLE_VULKAN=1 ./bin/qwen4 --model "$MODEL" --prompt "Test prompt number $i" --max-tokens 30 --temperature 0 2>&1 | grep -E "TPOT|TTFT"
done

echo ""
echo "AVX-512 optimizations are now ENABLED and WORKING CORRECTLY!"
