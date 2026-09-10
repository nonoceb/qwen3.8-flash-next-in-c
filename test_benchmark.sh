#!/bin/bash

MODEL="/mnt/DATA2T/_IA_Models/Qwen3.8Next/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf"

echo "========================================="
echo "CPU-Only Baseline (Vulkan Disabled)"
echo "========================================="
for i in {1..5}; do
    echo "Run $i:"
    Q38_DISABLE_VULKAN=1 ./bin/qwen4 --model "$MODEL" --prompt "Hello, how are you today?" --max-tokens 30 2>&1 | grep -E "TPOT|TTFT"
done

echo ""
echo "========================================="
echo "Vulkan GPU Offload Test"
echo "========================================="
for i in {1..5}; do
    echo "Run $i:"
    ./bin/qwen4 --model "$MODEL" --prompt "Hello, how are you today?" --max-tokens 30 2>&1 | grep -E "TPOT|TTFT"
done
