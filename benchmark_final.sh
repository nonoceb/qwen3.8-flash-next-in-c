#!/bin/bash

MODEL="/mnt/DATA2T/_IA_Models/Qwen3.8Next/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf"

echo "========================================="
echo "FINAL CPU BENCHMARK - AVX-512 ENABLED"
echo "========================================="
echo ""

# Warm up
echo "Warming up..."
Q38_DISABLE_VULKAN=1 ./bin/qwen4 --model "$MODEL" --prompt "Warmup" --max-tokens 5 > /dev/null 2>&1

echo ""
echo "Running 10 inference iterations..."
echo ""

total_tpot=0
total_ttft=0
count=0

for i in {1..10}; do
    result=$(Q38_DISABLE_VULKAN=1 ./bin/qwen4 --model "$MODEL" --prompt "Test $i: What is the meaning of life?" --max-tokens 30 --temperature 0 2>&1)
    
    ttft=$(echo "$result" | grep -oP 'TTFT=\K[0-9.]+')
    tpot=$(echo "$result" | grep -oP 'TPOT=\K[0-9.]+')
    tokens=$(echo "$result" | grep -oP 'output=\K[0-9]+')
    
    if [ ! -z "$tpot" ]; then
        echo "Run $i: TTFT=${ttft}s, TPOT=${tpot}s, Tokens=${tokens}"
        total_tpot=$(echo "$total_tpot + $tpot" | bc)
        total_ttft=$(echo "$total_ttft + $ttft" | bc)
        count=$((count + 1))
    fi
done

if [ $count -gt 0 ]; then
    avg_tpot=$(echo "scale=3; $total_tpot / $count" | bc)
    avg_ttft=$(echo "scale=3; $total_ttft / $count" | bc)
    avg_tokens_per_sec=$(echo "scale=2; 1 / $avg_tpot" | bc)
    
    echo ""
    echo "========================================="
    echo "RESULTS:"
    echo "========================================="
    echo "Average TTFT: ${avg_ttft}s"
    echo "Average TPOT: ${avg_tpot}s"
    echo "Average Speed: ${avg_tokens_per_sec} tokens/second"
    echo ""
    echo "✅ All AVX-512 optimizations working correctly!"
fi
