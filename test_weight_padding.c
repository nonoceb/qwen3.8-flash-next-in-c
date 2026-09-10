#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

int main() {
    // Simulate weight padding for one block
    uint8_t original[18] = {0};
    
    // Set scale to 1.0 (fp16 = 0x3C00)
    original[0] = 0x00;
    original[1] = 0x3C;
    // All other bytes are 0
    
    // Pad to 20 bytes
    uint8_t padded[20];
    memcpy(padded, original, 18);
    memset(padded + 18, 0, 2);
    
    printf("Original 18 bytes:\n");
    for (int i = 0; i < 18; i++) {
        printf("%02X ", original[i]);
    }
    printf("\n\nPadded 20 bytes:\n");
    for (int i = 0; i < 20; i++) {
        printf("%02X ", padded[i]);
    }
    printf("\n\nAs uints (5 total):\n");
    
    uint32_t *as_uints = (uint32_t*)padded;
    for (int i = 0; i < 5; i++) {
        printf("u%d = 0x%08X\n", i, as_uints[i]);
    }
    
    printf("\nExpected shader interpretation:\n");
    printf("  u0 & 0xFF = 0x%02X (scale low byte)\n", as_uints[0] & 0xFF);
    printf("  (u0 >> 8) & 0xFF = 0x%02X (scale high byte)\n", (as_uints[0] >> 8) & 0xFF);
    printf("  Expected scale = decode_fp16(0x00, 0x3C) ≈ 1.0\n");
    printf("  u0 >> 16 = 0x%08X (weight byte 0)\n", (as_uints[0] >> 16) & 0xFF);
    printf("  u0 >> 24 = 0x%08X (weight byte 1)\n", (as_uints[0] >> 24) & 0xFF);
    
    return 0;
}
