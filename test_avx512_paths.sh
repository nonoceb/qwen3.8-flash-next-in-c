#!/bin/bash

MODEL="/mnt/DATA2T/_IA_Models/Qwen3.8Next/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf"
PROMPT="The capital of France is"
TOKENS=10

echo "Testing AVX-512 Paths"
echo "====================="
echo ""

# Test 1: All disabled (baseline)
echo "1. All AVX-512 disabled:"
Q38_DISABLE_VULKAN=1 Q38_DISABLE_F32_AVX512=1 Q38_DISABLE_IQ4NL_AVX512=1 Q38_DISABLE_Q80_AVX512=1 \
    ./bin/qwen4 --model "$MODEL" --prompt "$PROMPT" --max-tokens $TOKENS 2>&1 | grep -E "capital|TPOT"

echo ""

# Test 2: Only F32 AVX-512 enabled
echo "2. Only F32 AVX-512 enabled:"
Q38_DISABLE_VULKAN=1 Q38_DISABLE_IQ4NL_AVX512=1 Q38_DISABLE_Q80_AVX512=1 \
    ./bin/qwen4 --model "$MODEL" --prompt "$PROMPT" --max-tokens $TOKENS 2>&1 | grep -E "capital|TPOT"

echo ""

# Test 3: Only IQ4_NL AVX-512 enabled
echo "3. Only IQ4_NL AVX-512 enabled:"
Q38_DISABLE_VULKAN=1 Q38_DISABLE_F32_AVX512=1 Q38_DISABLE_Q80_AVX512=1 \
    ./bin/qwen4 --model "$MODEL" --prompt "$PROMPT" --max-tokens $TOKENS 2>&1 | grep -E "capital|TPOT"

echo ""

# Test 4: Only Q8_0 AVX-512 enabled
echo "4. Only Q8_0 AVX-512 enabled:"
Q38_DISABLE_VULKAN=1 Q38_DISABLE_F32_AVX512=1 Q38_DISABLE_IQ4NL_AVX512=1 \
    ./bin/qwen4 --model "$MODEL" --prompt "$PROMPT" --max-tokens $TOKENS 2>&1 | grep -E "capital|TPOT"

echo ""

# Test 5: All enabled
echo "5. All AVX-512 enabled:"
Q38_DISABLE_VULKAN=1 \
    ./bin/qwen4 --model "$MODEL" --prompt "$PROMPT" --max-tokens $TOKENS 2>&1 | grep -E "capital|TPOT"

