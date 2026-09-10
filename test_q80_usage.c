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
    
    Q4GGUFSet set;
    memset(&set, 0, sizeof(set));
    
    if (!q4_gguf_set_open(&set, model_path, 0)) {
        fprintf(stderr, "FAILED to open GGUF set\n");
        return 1;
    }
    
    // Check which layers use Q8_0
    printf("Checking for Q8_0 tensors (type 8):\n\n");
    
    for (size_t shard = 0; shard < set.shard_count; shard++) {
        for (uint64_t i = 0; i < set.shards[shard].tensor_count; i++) {
            const Q38GGUFTensor *t = &set.shards[shard].tensors[i];
            if (t->type == 8) {  // Q8_0
                printf("  %.*s: type=%d shape=[%llu, %llu]\n",
                       (int)t->name.length, t->name.data, t->type,
                       (unsigned long long)t->shape[0],
                       (unsigned long long)t->shape[1]);
            }
        }
    }
    
    q4_gguf_set_close(&set);
    return 0;
}
