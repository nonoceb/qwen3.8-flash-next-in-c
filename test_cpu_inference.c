#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "qwen4/qwen4_model.h"
#include "qwen38/qwen38_tokenizer.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }
    
    const char *model_path = argv[1];
    
    printf("Loading model from: %s\n", model_path);
    
    // Disable Vulkan
    setenv("Q38_DISABLE_VULKAN", "1", 1);
    
    // Open model with small context
    Q4Model *model = q4_model_open_gguf(model_path, 512);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    
    printf("Model loaded successfully!\n");
    printf("Architecture: %s\n", model->architecture == Q4_ARCH_QWEN4EXP ? "qwen4exp" : "unknown");
    printf("Block count: %u\n", model->block_count);
    printf("Embedding length: %u\n", model->embedding_length);
    printf("Context length: %u\n", model->context_length);
    printf("Expert count: %u\n", model->expert_count);
    printf("Experts used: %u\n", model->expert_used_count);
    
    // Test tokenizer
    const char *test_text = "Hello, how are you?";
    printf("\nTokenizing: '%s'\n", test_text);
    
    uint32_t tokens[64];
    size_t token_count = 0;
    
    int result = q38_tokenize(&model->tokenizer, test_text, strlen(test_text),
                              tokens, &token_count, 64, true, true);
    
    if (result != 0) {
        fprintf(stderr, "Tokenization failed with code %d\n", result);
    } else {
        printf("Token count: %zu\n", token_count);
        printf("Tokens: ");
        for (size_t i = 0; i < token_count && i < 10; i++) {
            printf("%u ", tokens[i]);
        }
        printf("\n");
        
        // Decode back
        printf("Decoded: ");
        for (size_t i = 0; i < token_count && i < 10; i++) {
            const char *decoded = q38_decode(&model->tokenizer, tokens[i]);
            if (decoded) {
                printf("%s", decoded);
            } else {
                printf("[UNK:%u]", tokens[i]);
            }
        }
        printf("\n");
    }
    
    q4_model_close(model);
    printf("\nModel closed.\n");
    
    return 0;
}
