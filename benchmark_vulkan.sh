#!/bin/bash

MODEL="/mnt/DATA2T/_IA_Models/Qwen3.8Next/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf"

echo "========================================="
echo "VULKAN GPU vs CPU BENCHMARK"
echo "========================================="
echo ""

# Warmup runs
echo "Warming up..."
Q38_DISABLE_VULKAN=1 ./bin/qwen4 --model "$MODEL" --prompt "Warmup" --max-tokens 5 > /dev/null 2>&1
./bin/qwen4 --model "$MODEL" --prompt "Warmup" --max-tokens 5 > /dev/null 2>&1

echo ""
echo "========================================="
echo "CPU-ONLY BASELINE (Vulkan Disabled)"
echo "========================================="
echo ""

cpu_times=()
for i in {1..5}; do
    result=$(Q38_DISABLE_VULKAN=1 ./bin/qwen4 --model "$MODEL" --prompt "Test $i: What is artificial intelligence?" --max-tokens 30 --temperature 0 2>&1)
    ttft=$(echo "$result" | grep -oP 'TTFT=\K[0-9.]+')
    tpot=$(echo "$result" | grep -oP 'TPOT=\K[0-9.]+')
    tokens=$(echo "$result" | grep -oP 'output=\K[0-9]+')
    
    if [ ! -z "$tpot" ]; then
        speed=$(awk "BEGIN {printf \"%.2f\", 1/$tpot}")
        echo "Run $i: TTFT=${ttft}s, TPOT=${tpot}s, Tokens=${tokens}, Speed=${speed} tok/s"
        cpu_times+=("$tpot")
    fi
done

echo ""
echo "========================================="
echo "VULKAN GPU OFFLOAD"
echo "========================================="
echo ""

gpu_times=()
for i in {1..5}; do
    result=$(./bin/qwen4 --model "$MODEL" --prompt "Test $i: What is artificial intelligence?" --max-tokens 30 --temperature 0 2>&1)
    ttft=$(echo "$result" | grep -oP 'TTFT=\K[0-9.]+')
    tpot=$(echo "$result" | grep -oP 'TPOT=\K[0-9.]+')
    tokens=$(echo "$result" | grep -oP 'output=\K[0-9]+')
    
    if [ ! -z "$tpot" ]; then
        speed=$(awk "BEGIN {printf \"%.2f\", 1/$tpot}")
        echo "Run $i: TTFT=${ttft}s, TPOT=${tpot}s, Tokens=${tokens}, Speed=${speed} tok/s"
        gpu_times+=("$tpot")
    fi
done

echo ""
echo "========================================="
echo "RESULTS SUMMARY"
echo "========================================="
echo ""

if [ ${#cpu_times[@]} -gt 0 ] && [ ${#gpu_times[@]} -gt 0 ]; then
    # Calculate averages using awk
    cpu_sum=$(IFS=+; echo "${cpu_times[*]}" | bc)
    cpu_avg=$(awk "BEGIN {printf \"%.3f\", $cpu_sum / ${#cpu_times[@]}}")
    cpu_speed=$(awk "BEGIN {printf \"%.2f\", 1 / $cpu_avg}")
    
    gpu_sum=$(IFS=+; echo "${gpu_times[*]}" | bc)
    gpu_avg=$(awk "BEGIN {printf \"%.3f\", $gpu_sum / ${#gpu_times[@]}}")
    gpu_speed=$(awk "BEGIN {printf \"%.2f\", 1 / $gpu_avg}")
    
    speedup=$(awk "BEGIN {printf \"%.2f\", $cpu_avg / $gpu_avg}")
    
    echo "CPU-Only Average:"
    echo "  TPOT: ${cpu_avg}s"
    echo "  Speed: ${cpu_speed} tokens/s"
    echo ""
    echo "Vulkan GPU Average:"
    echo "  TPOT: ${gpu_avg}s"
    echo "  Speed: ${gpu_speed} tokens/s"
    echo ""
    echo "Speedup: ${speedup}x"
else
    echo "Could not calculate averages"
fi
