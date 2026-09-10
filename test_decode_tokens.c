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
    
    printf("Testing token decoding from: %s\n\n", model_path);
    
    Q4GGUFSet set;
    memset(&set, 0, sizeof(set));
    
    // Open header only
    if (!q4_gguf_set_open(&set, model_path, 1)) {
        fprintf(stderr, "FAILED to open GGUF set\n");
        return 1;
    }
    
    // Get metadata
    const Q38GGUF *meta = q4_gguf_metadata(&set);
    
    // Find tokens array
    const Q38GGUFMeta *tokens_meta = q38_gguf_find_meta(meta, "tokenizer.ggml.tokens");
    if (!tokens_meta || tokens_meta->type != Q38_GGUF_META_ARRAY || 
        tokens_meta->array_type != Q38_GGUF_META_STRING) {
        fprintf(stderr, "Tokens metadata not found or wrong type\n");
        q4_gguf_set_close(&set);
        return 1;
    }
    
    printf("Total tokens in vocabulary: %llu\n\n", (unsigned long long)tokens_meta->count);
    
    // Decode some specific tokens
    uint32_t test_tokens[] = {0, 1, 2, 100, 1000, 248046};
    int num_tests = sizeof(test_tokens) / sizeof(test_tokens[0]);
    
    for (int i = 0; i < num_tests; i++) {
        uint32_t token_id = test_tokens[i];
        
        if (token_id >= tokens_meta->count) {
            printf("Token %u: OUT OF RANGE\n", token_id);
            continue;
        }
        
        Q38GGUFString token_str;
        if (q38_gguf_meta_array_string(tokens_meta, token_id, &token_str)) {
            printf("Token %u: '%.*s'\n", token_id, (int)token_str.length, token_str.data);
        } else {
            printf("Token %u: FAILED to decode\n", token_id);
        }
    }
    
    q4_gguf_set_close(&set);
    printf("\nDone.\n");
    
    return 0;
}
