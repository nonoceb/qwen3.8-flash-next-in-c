#!/bin/bash

MODEL="/mnt/DATA2T/_IA_Models/Qwen3.8Next/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf"

echo "Testing Vulkan with simple prompt..."
echo ""

# Test 1: Very short prompt
echo "Test 1: Short prompt (5 tokens)"
./bin/qwen4 --model "$MODEL" --prompt "Hi" --max-tokens 5 2>&1 | grep -E "TPOT|TTFT|Hi"

echo ""

# Test 2: Check if it crashes
echo "Test 2: Checking crash status..."
./bin/qwen4 --model "$MODEL" --prompt "Test" --max-tokens 3 2>&1 | tail -5
