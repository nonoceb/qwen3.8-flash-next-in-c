#include <stdio.h>
#include <string.h>
#include "qwen38/qwen38_gguf.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }
    
    Q38GGUF gguf;
    if (!q38_gguf_open_header(&gguf, argv[1])) {
        fprintf(stderr, "Failed to open GGUF header\n");
        return 1;
    }
    
    printf("Total tensors: %lu\n", gguf.tensor_count);
    printf("Data offset: %lu\n", gguf.data_offset);
    printf("\nExpert-related tensors:\n");
    
    int expert_count = 0;
    for (uint64_t i = 0; i < gguf.tensor_count && expert_count < 30; i++) {
        Q38GGUFTensor *t = &gguf.tensors[i];
        
        // Look for expert tensors
        if (strstr(t->name.data, "experts")) {
            expert_count++;
            
            printf("  [%3lu] %-80s shape=[", i, t->name.data);
            for (uint32_t d = 0; d < t->n_dims; d++) {
                printf("%lu%s", t->shape[d], d < t->n_dims-1 ? ", " : "");
            }
            printf("] type=%u offset=%lu size=%lu\n", t->type, t->offset, t->nbytes);
        }
    }
    
    q38_gguf_close(&gguf);
    return 0;
}
