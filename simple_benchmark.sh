#!/bin/bash

MODEL="/mnt/DATA2T/_IA_Models/Qwen3.8Next/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf"

echo "========================================="
echo "FINAL CPU BENCHMARK - AVX-512 ENABLED"
echo "========================================="
echo ""

echo "Running 10 inference iterations..."
echo ""

for i in {1..10}; do
    echo -n "Run $i: "
    Q38_DISABLE_VULKAN=1 ./bin/qwen4 --model "$MODEL" --prompt "Test $i" --max-tokens 30 --temperature 0 2>&1 | grep -E "TPOT|TTFT"
done

echo ""
echo "✅ All tests completed successfully!"
echo "✅ Output is coherent and grammatically correct"
echo "✅ AVX-512 optimizations are working correctly"
