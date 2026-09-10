#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "qwen4/qwen4_gguf.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model-00001-of-N.gguf>\n", argv[0]);
        return 1;
    }
    
    const char *model_path = argv[1];
    
    printf("Opening GGUF set from: %s\n\n", model_path);
    
    Q4GGUFSet set;
    memset(&set, 0, sizeof(set));
    
    // Open with full data (not header_only)
    if (!q4_gguf_set_open(&set, model_path, 0)) {
        fprintf(stderr, "FAILED to open GGUF set\n");
        return 1;
    }
    
    printf("Successfully opened GGUF set!\n");
    printf("Shard count: %zu\n\n", set.shard_count);
    
    // Count total tensors
    size_t total_tensors = 0;
    for (size_t i = 0; i < set.shard_count; i++) {
        printf("Shard %zu: %llu tensors, data_offset=%llu\n", 
               i, 
               (unsigned long long)set.shards[i].tensor_count,
               (unsigned long long)set.shards[i].data_offset);
        total_tensors += set.shards[i].tensor_count;
    }
    
    printf("\nTotal tensors across all shards: %zu\n", total_tensors);
    
    // Try to find some key tensors
    const char *test_tensors[] = {
        "token_embd.weight",
        "output.weight",
        "blk.0.attn_qkv.weight",
        "blk.0.ffn_gate_exps.weight",
        NULL
    };
    
    printf("\nSearching for key tensors:\n");
    for (int i = 0; test_tensors[i]; i++) {
        const Q38GGUFTensor *tensor = q4_gguf_find_tensor(&set, test_tensors[i]);
        if (tensor) {
            printf("  ✓ Found '%s': shape=[", test_tensors[i]);
            for (uint32_t d = 0; d < tensor->n_dims; d++) {
                printf("%llu", (unsigned long long)tensor->shape[d]);
                if (d < tensor->n_dims - 1) printf(", ");
            }
            printf("], type=%u, data=%p\n", tensor->type, (void*)tensor->data);
        } else {
            printf("  ✗ NOT FOUND: '%s'\n", test_tensors[i]);
        }
    }
    
    q4_gguf_set_close(&set);
    printf("\nGGUF set closed.\n");
    
    return 0;
}
