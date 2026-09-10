#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "qwen38/qwen38_gguf.h"
#include "qwen4/qwen4_gguf.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model-00001-of-N.gguf>\n", argv[0]);
        return 1;
    }
    
    const char *model_path = argv[1];
    
    printf("Loading tokenizer from: %s\n\n", model_path);
    
    Q4GGUFSet set;
    memset(&set, 0, sizeof(set));
    
    // Open header only for faster loading
    if (!q4_gguf_set_open(&set, model_path, 1)) {
        fprintf(stderr, "FAILED to open GGUF set\n");
        return 1;
    }
    
    // Get metadata
    const Q38GGUF *meta = q4_gguf_metadata(&set);
    if (!meta) {
        fprintf(stderr, "Failed to get metadata\n");
        q4_gguf_set_close(&set);
        return 1;
    }
    
    // Check EOS token
    uint64_t eos_token_id = 0;
    if (q38_gguf_meta_u64(meta, "tokenizer.ggml.eos_token_id", &eos_token_id)) {
        printf("EOS token ID: %llu\n", (unsigned long long)eos_token_id);
    }
    
    // Find token embeddings tensor
    const Q38GGUFTensor *token_embd = q4_gguf_find_tensor(&set, "token_embd.weight");
    if (token_embd) {
        printf("\nToken embedding tensor found:\n");
        printf("  Shape: [%llu, %llu]\n", 
               (unsigned long long)token_embd->shape[0],
               (unsigned long long)token_embd->shape[1]);
        printf("  Type: %u\n", token_embd->type);
        printf("  Vocab size: %llu\n", (unsigned long long)token_embd->shape[1]);
    } else {
        printf("\nWARNING: token_embd.weight not found!\n");
    }
    
    // Check per_layer_token_embd
    const Q38GGUFTensor *per_layer = q4_gguf_find_tensor(&set, "per_layer_token_embd.weight");
    if (per_layer) {
        printf("\nPer-layer token embedding found:\n");
        printf("  Shape: [%llu, %llu]\n", 
               (unsigned long long)per_layer->shape[0],
               (unsigned long long)per_layer->shape[1]);
        printf("  Type: %u\n", per_layer->type);
    }
    
    q4_gguf_set_close(&set);
    printf("\nDone.\n");
    
    return 0;
}
