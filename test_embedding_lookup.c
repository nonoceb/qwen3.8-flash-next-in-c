#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "qwen38/qwen38_gguf.h"
#include "qwen4/qwen4_gguf.h"
#include "qwen38/qwen38_quant.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model-00001-of-N.gguf>\n", argv[0]);
        return 1;
    }
    
    const char *model_path = argv[1];
    
    printf("Testing embedding lookup from: %s\n\n", model_path);
    
    Q4GGUFSet set;
    memset(&set, 0, sizeof(set));
    
    // Open with full data
    if (!q4_gguf_set_open(&set, model_path, 0)) {
        fprintf(stderr, "FAILED to open GGUF set\n");
        return 1;
    }
    
    // Find token embedding tensor
    const Q38GGUFTensor *token_embd = q4_gguf_find_tensor(&set, "token_embd.weight");
    if (!token_embd) {
        fprintf(stderr, "token_embd.weight not found\n");
        q4_gguf_set_close(&set);
        return 1;
    }
    
    printf("Token embedding shape: [%llu, %llu]\n",
           (unsigned long long)token_embd->shape[0],
           (unsigned long long)token_embd->shape[1]);
    printf("Type: %u\n\n", token_embd->type);
    
    // Test token IDs
    uint32_t test_tokens[] = {0, 1, 2, 100, 1000, 248046};
    int num_tests = sizeof(test_tokens) / sizeof(test_tokens[0]);
    
    float *embedding = malloc(token_embd->shape[0] * sizeof(float));
    if (!embedding) {
        fprintf(stderr, "Failed to allocate embedding buffer\n");
        q4_gguf_set_close(&set);
        return 1;
    }
    
    for (int i = 0; i < num_tests; i++) {
        uint32_t token_id = test_tokens[i];
        
        if (token_id >= token_embd->shape[1]) {
            printf("Token %u: OUT OF RANGE (vocab size = %llu)\n",
                   token_id, (unsigned long long)token_embd->shape[1]);
            continue;
        }
        
        int result = q38_tensor_row_f32(embedding, token_embd, token_id);
        
        if (result) {
            // Check for NaN/Inf
            int has_nan = 0;
            int has_inf = 0;
            float sum = 0.0f;
            for (uint64_t j = 0; j < token_embd->shape[0]; j++) {
                if (!isfinite(embedding[j])) {
                    has_nan = isnan(embedding[j]);
                    has_inf = isinf(embedding[j]);
                    break;
                }
                sum += embedding[j];
            }
            
            printf("Token %u: ", token_id);
            if (has_nan) printf("HAS NaN! ");
            else if (has_inf) printf("HAS Inf! ");
            else printf("OK (sum=%.2f) ", sum);
            
            // Print first few values
            printf("[%.4f, %.4f, %.4f, ...]\n", 
                   embedding[0], embedding[1], embedding[2]);
        } else {
            printf("Token %u: FAILED to extract\n", token_id);
        }
    }
    
    free(embedding);
    q4_gguf_set_close(&set);
    printf("\nDone.\n");
    
    return 0;
}
