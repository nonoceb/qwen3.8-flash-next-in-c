#include "vulkan/vulkan_context.h"
#include "vulkan/vulkan_buffers.h"
#include <stdio.h>

int main() {
    printf("Testing Vulkan Compute Setup for AMD 780M\n");
    printf("==========================================\n\n");
    
    // Initialize Vulkan context
    Q38VulkanContext ctx;
    if (!q38_vulkan_init(&ctx)) {
        fprintf(stderr, "Failed to initialize Vulkan\n");
        return 1;
    }
    
    printf("\nVulkan initialized successfully!\n");
    printf("  - UMA support: %s\n", ctx.is_uma ? "Yes" : "No");
    printf("  - Compute queue family: %u\n", ctx.compute_queue_family_index);
    
    // Test buffer creation
    printf("\nTesting buffer creation...\n");
    Q38VulkanBuffer test_buf;
    const uint64_t buf_size = 1024 * 1024;  // 1 MB
    
    if (q38_vulkan_buffer_create(&ctx, &test_buf, buf_size,
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                  VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                  true)) {
        printf("  - Buffer created: %llu bytes\n", (unsigned long long)test_buf.size);
        printf("  - Unified memory: %s\n", test_buf.is_unified ? "Yes" : "No");
        
        // Test write/read
        float test_data[256];
        for (int i = 0; i < 256; i++) test_data[i] = (float)i;
        
        if (q38_vulkan_buffer_write(&ctx, &test_buf, test_data, 0, sizeof(test_data))) {
            printf("  - Write successful\n");
            
            float read_data[256];
            if (q38_vulkan_buffer_read(&ctx, &test_buf, read_data, 0, sizeof(read_data))) {
                printf("  - Read successful\n");
                
                // Verify data
                bool match = true;
                for (int i = 0; i < 256; i++) {
                    if (read_data[i] != test_data[i]) {
                        match = false;
                        break;
                    }
                }
                printf("  - Data verification: %s\n", match ? "PASS" : "FAIL");
            }
        }
        
        q38_vulkan_buffer_destroy(&ctx, &test_buf);
        printf("  - Buffer destroyed\n");
    } else {
        fprintf(stderr, "  - Failed to create buffer\n");
    }
    
    // Cleanup
    printf("\nCleaning up...\n");
    q38_vulkan_cleanup(&ctx);
    printf("Done!\n");
    
    return 0;
}
