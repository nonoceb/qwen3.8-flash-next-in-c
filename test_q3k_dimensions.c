#include <stdio.h>
#include <string.h>
#include "qwen38/qwen38_gguf.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/vulkan_gemv.h"

int main() {
    printf("Testing Q3_K dimension requirements:\n");
    printf("Q4_HIDDEN = 2560 (divisible by 256: %d blocks)\n", 2560/256);
    printf("Q4_EXPERT_FFN = 640 (NOT divisible by 256: %.2f blocks)\n\n", 640.0/256);
    
    printf("Expert tensor shapes:\n");
    printf("  gate_proj: [2560, 640] -> GEMV input_dim=2560✓ output_dim=640✗\n");
    printf("  up_proj:   [2560, 640] -> GEMV input_dim=2560✓ output_dim=640✗\n");
    printf("  down_proj: [640, 2560] -> GEMV input_dim=640✗ output_dim=2560✓\n\n");
    
    printf("Problem: Q3_K requires N %% 256 == 0, but 640 %% 256 = %d\n\n", 640 % 256);
    
    printf("Solution needed:\n");
    printf("1. Shader must handle partial blocks (last block may have < 256 elements)\n");
    printf("2. Or quantization must pad dimensions to 256 multiples\n");
    printf("3. Or use different quantization for these tensors\n");
    
    return 0;
}
