#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "qwen4/qwen4_model.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model-00001-of-N.gguf>\n", argv[0]);
        return 1;
    }
    
    // Disable Vulkan
    setenv("Q38_DISABLE_VULKAN", "1", 1);
    
    printf("Loading model from: %s\n\n", argv[1]);
    
    Q4Model *model = q4_model_open_gguf(argv[1], 512);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    
    printf("Model loaded successfully!\n");
    printf("Context length: %u\n", model->context_length);
    printf("Layers: %u\n", model->layers);
    printf("Hidden dim: %u\n", model->hidden);
    printf("Experts: %u\n", model->experts);
    printf("Active experts: %u\n", model->active_experts);
    
    // Check PLE parameters
    printf("\nPLE configuration:\n");
    for (int i = 0; i < 3; i++) {
        printf("  Multiplier[%d] = %llu\n", i, (unsigned long long)model->ple_multipliers[i]);
    }
    for (int i = 0; i < 16; i++) {
        printf("  Offset[%d] = %llu, Size[%d] = %llu\n", 
               i, (unsigned long long)model->ple_offsets[i],
               i, (unsigned long long)model->ple_sizes[i]);
    }
    
    // Test token embedding lookup
    uint32_t test_token = 100;
    float embedding[2560];
    if (q38_tensor_row_f32(embedding, model->embedding, test_token)) {
        printf("\nToken %u embedding: [%.4f, %.4f, %.4f, ...]\n", 
               test_token, embedding[0], embedding[1], embedding[2]);
    } else {
        printf("\nFailed to get embedding for token %u\n", test_token);
    }
    
    q4_model_close(model);
    printf("\nModel closed.\n");
    
    return 0;
}
