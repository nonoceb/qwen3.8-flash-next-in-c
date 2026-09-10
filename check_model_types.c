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
    
    printf("Checking tensor types in: %s\n\n", model_path);
    
    Q4GGUFSet set;
    memset(&set, 0, sizeof(set));
    
    // Open with full data
    if (!q4_gguf_set_open(&set, model_path, 0)) {
        fprintf(stderr, "FAILED to open GGUF set\n");
        return 1;
    }
    
    printf("Total shards: %zu\n\n", set.shard_count);
    
    // Count tensors by type
    int type_counts[30] = {0};
    size_t total_tensors = 0;
    
    for (size_t shard = 0; shard < set.shard_count; shard++) {
        for (uint64_t i = 0; i < set.shards[shard].tensor_count; i++) {
            const Q38GGUFTensor *t = &set.shards[shard].tensors[i];
            if (t->type < 30) {
                type_counts[t->type]++;
            }
            total_tensors++;
        }
    }
    
    printf("Tensor type distribution:\n");
    printf("Type 8 (Q8_0):     %d tensors\n", type_counts[8]);
    printf("Type 14 (Q3_K_XL): %d tensors\n", type_counts[14]);
    printf("Type 18 (IQ4_NL?): %d tensors\n", type_counts[18]);
    printf("Type 20 (IQ4_NL):  %d tensors\n", type_counts[20]);
    printf("\nOther types:\n");
    for (int i = 0; i < 30; i++) {
        if (type_counts[i] > 0 && i != 8 && i != 14 && i != 18 && i != 20) {
            printf("Type %d: %d tensors\n", i, type_counts[i]);
        }
    }
    
    // Check specific important tensors
    const char *check_tensors[] = {
        "token_embd.weight",
        "output.weight",
        "blk.0.ffn_gate_exps.weight",
        "blk.0.ffn_up_exps.weight",
        "blk.0.ffn_down_exps.weight",
        NULL
    };
    
    printf("\nKey tensor types:\n");
    for (int i = 0; check_tensors[i]; i++) {
        const Q38GGUFTensor *t = q4_gguf_find_tensor(&set, check_tensors[i]);
        if (t) {
            printf("  %s: type=%d\n", check_tensors[i], t->type);
        }
    }
    
    q4_gguf_set_close(&set);
    printf("\nTotal tensors: %zu\n", total_tensors);
    
    return 0;
}
