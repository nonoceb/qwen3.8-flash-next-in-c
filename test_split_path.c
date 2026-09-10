#include <stdio.h>
#include <string.h>

static int split_path(char *output, size_t capacity, const char *first,
                      size_t shard, size_t count)
{
    static const char marker[] = "-00001-of-";
    const char *at = strstr(first, marker);
    if (!at || strstr(at + 1, marker)) return 0;
    const char *total = at + sizeof(marker) - 1u;
    char expected[16];
    const int expected_length = snprintf(expected, sizeof(expected),
                                         "%05zu.gguf", count);
    if (expected_length <= 0 || strcmp(total, expected) != 0) return 0;
    const int prefix = (int)(at - first);
    const int length = snprintf(output, capacity, "%.*s-%05zu-of-%05zu.gguf",
                                prefix, first, shard, count);
    return length > 0 && (size_t)length < capacity;
}

int main() {
    const char *input = "/mnt/DATA2T/_IA_Models/Qwen3.8Next/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf";
    char output[4096];
    
    printf("Input: %s\n", input);
    
    // Test shard 2
    if (split_path(output, sizeof(output), input, 2, 3)) {
        printf("Shard 2: %s\n", output);
    } else {
        printf("FAILED to generate path for shard 2\n");
    }
    
    // Test shard 3
    if (split_path(output, sizeof(output), input, 3, 3)) {
        printf("Shard 3: %s\n", output);
    } else {
        printf("FAILED to generate path for shard 3\n");
    }
    
    return 0;
}
