#include "vulkan/vulkan_wrapper.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/vulkan_gemv.h"
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>

// Global Vulkan state
static struct {
    bool enabled;
    bool initialized;
    Q38VulkanContext context;
    Q38VulkanGEMV gemv;
} g_vulkan = {.enabled = false, .initialized = false};

static pthread_mutex_t g_vulkan_mutex = PTHREAD_MUTEX_INITIALIZER;

bool q38_vulkan_is_enabled(void) {
    return g_vulkan.enabled && g_vulkan.initialized;
}

void q38_vulkan_set_enabled(bool enabled) {
    pthread_mutex_lock(&g_vulkan_mutex);
    
    // Check for environment variable to disable Vulkan
    if (enabled && getenv("Q38_DISABLE_VULKAN")) {
        g_vulkan.enabled = false;
        pthread_mutex_unlock(&g_vulkan_mutex);
        return;
    }
    
    if (enabled && !g_vulkan.initialized) {
        // Try to initialize Vulkan
        if (q38_vulkan_init(&g_vulkan.context)) {
            if (q38_vulkan_gemv_init(&g_vulkan.context, &g_vulkan.gemv)) {
                g_vulkan.enabled = true;
                g_vulkan.initialized = true;
                printf("Vulkan GPU acceleration enabled\n");
            } else {
                fprintf(stderr, "Failed to initialize Vulkan GEMV kernels\n");
                q38_vulkan_cleanup(&g_vulkan.context);
            }
        } else {
            fprintf(stderr, "Vulkan not available, using CPU fallback\n");
            g_vulkan.enabled = false;
        }
    } else if (!enabled && g_vulkan.initialized) {
        g_vulkan.enabled = false;
    }
    
    pthread_mutex_unlock(&g_vulkan_mutex);
}

Q38VulkanContext* q38_vulkan_get_context(void) {
    if (!g_vulkan.initialized) {
        q38_vulkan_set_enabled(true);
    }
    return g_vulkan.initialized ? &g_vulkan.context : NULL;
}

Q38VulkanGEMV* q38_vulkan_get_gemv(void) {
    if (!g_vulkan.initialized) {
        q38_vulkan_set_enabled(true);
    }
    return g_vulkan.initialized ? &g_vulkan.gemv : NULL;
}

void q38_vulkan_shutdown(void) {
    if (g_vulkan.initialized) {
        q38_vulkan_gemv_cleanup(&g_vulkan.context, &g_vulkan.gemv);
        q38_vulkan_cleanup(&g_vulkan.context);
        g_vulkan.initialized = false;
        g_vulkan.enabled = false;
    }
}
