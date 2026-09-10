#include <stdio.h>
#include <string.h>
#include "qwen38/qwen38_gguf.h"

int main(int argc, char **argv) {
    if (argc < 2) return 1;
    
    Q38GGUF gguf;
    if (!q38_gguf_open(&gguf, argv[1])) {
        fprintf(stderr, "Failed to open GGUF\n");
        return 1;
    }
    
    // Find expert tensors
    for (uint64_t i = 0; i < gguf.tensor_count; i++) {
        Q38GGUFTensor *t = &gguf.tensors[i];
        
        if (strstr(t->name.data, "ffn_gate_exps.weight") ||
            strstr(t->name.data, "ffn_up_exps.weight") ||
            strstr(t->name.data, "ffn_down_exps.weight")) {
            
            printf("%s: shape=[", t->name.data);
            for (uint32_t d = 0; d < t->n_dims; d++) {
                printf("%lu%s", t->shape[d], d < t->n_dims-1 ? ", " : "");
            }
            printf("] type=%u\n", t->type);
            
            // Only show first layer
            if (strstr(t->name.data, ".0.")) break;
        }
    }
    
    q38_gguf_close(&gguf);
    return 0;
}
