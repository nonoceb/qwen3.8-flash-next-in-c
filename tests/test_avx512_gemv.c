#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// IQ4_NL codebook
static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1, 13, 25, 38, 53, 69, 89, 113
};

// Simple scalar reference implementation for IQ4_NL
void gemv_iq4nl_scalar(float *output, const uint8_t *weights, const float *input, 
                        uint32_t M, uint32_t N) {
    const uint32_t blocks_per_row = N / 32;
    
    for (uint32_t row = 0; row < M; row++) {
        float sum = 0.0f;
        const uint8_t *row_ptr = weights + row * blocks_per_row * 18;
        
        for (uint32_t block = 0; block < blocks_per_row; block++) {
            const uint8_t *block_ptr = row_ptr + block * 18;
            
            // Read scale (fp16)
            uint16_t scale_u16 = block_ptr[0] | (block_ptr[1] << 8);
            uint32_t sign = (scale_u16 >> 15) & 1;
            uint32_t exp = (scale_u16 >> 10) & 31;
            uint32_t mant = scale_u16 & 1023;
            float scale = (exp == 0) ? 0.0f : 
                (sign ? -1.0f : 1.0f) * (1.0f + mant / 1024.0f) * powf(2.0f, exp - 15.0f);
            
            // Process 16 packed bytes (32 values)
            for (uint32_t i = 0; i < 16; i++) {
                uint8_t packed = block_ptr[2 + i];
                
                // Lower nibble
                uint32_t col_lo = block * 32 + i * 2;
                if (col_lo < N) {
                    float weight = kvalues_iq4nl[packed & 0xF] * scale;
                    sum += weight * input[col_lo];
                }
                
                // Upper nibble
                uint32_t col_hi = block * 32 + i * 2 + 1;
                if (col_hi < N) {
                    float weight = kvalues_iq4nl[(packed >> 4) & 0xF] * scale;
                    sum += weight * input[col_hi];
                }
            }
        }
        output[row] = sum;
    }
}

// External AVX2 and AVX-512 implementations
extern void gemv_iq4nl_avx2(float *output, const uint8_t *weights, const float *input,
                             uint32_t M, uint32_t N);
extern void gemv_iq4nl_avx512(float *output, const uint8_t *weights, const float *input,
                               uint32_t M, uint32_t N);

int main() {
    printf("Testing AVX-512 IQ4_NL GEMV Optimization\n");
    printf("========================================\n\n");
    
    // Check CPU support
    FILE *cpuinfo = fopen("/proc/cpuinfo", "r");
    char line[256];
    bool has_avx512 = false;
    bool has_avx2 = false;
    
    while (fgets(line, sizeof(line), cpuinfo)) {
        if (strstr(line, "flags") == line) {
            if (strstr(line, "avx512f")) has_avx512 = true;
            if (strstr(line, "avx2")) has_avx2 = true;
            break;
        }
    }
    fclose(cpuinfo);
    
    printf("CPU Support:\n");
    printf("  AVX2:    %s\n", has_avx2 ? "Yes" : "No");
    printf("  AVX-512: %s\n\n", has_avx512 ? "Yes" : "No");
    
    if (!has_avx512) {
        printf("AVX-512 not supported on this CPU\n");
        return 1;
    }
    
    // Test sizes relevant to Qwen3.8
    struct {
        uint32_t M;
        uint32_t N;
        const char *desc;
    } test_sizes[] = {
        {640, 2560, "Expert gate/up projection"},
        {2560, 640, "Expert down projection"},
        {2560, 2560, "Hidden dimension"},
        {512, 2560, "Router projection"},
    };
    
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    for (int t = 0; t < num_tests; t++) {
        uint32_t M = test_sizes[t].M;
        uint32_t N = test_sizes[t].N;
        
        printf("Test %d: %s (%u x %u)\n", t + 1, test_sizes[t].desc, M, N);
        
        const uint32_t blocks_per_row = N / 32;
        const uint64_t weight_size = M * blocks_per_row * 18;
        
        // Allocate aligned memory
        uint8_t *weights = aligned_alloc(64, weight_size);
        float *input = aligned_alloc(64, N * sizeof(float));
        float *output_scalar = aligned_alloc(64, M * sizeof(float));
        float *output_avx2 = aligned_alloc(64, M * sizeof(float));
        float *output_avx512 = aligned_alloc(64, M * sizeof(float));
        
        // Initialize with random data
        srand(42 + t);
        for (uint64_t i = 0; i < weight_size; i++) {
            weights[i] = rand() % 256;
        }
        for (uint32_t i = 0; i < N; i++) {
            input[i] = (float)(rand() % 100) / 100.0f - 0.5f;
        }
        
        // Benchmark scalar
        clock_t start = clock();
        for (int iter = 0; iter < 100; iter++) {
            gemv_iq4nl_scalar(output_scalar, weights, input, M, N);
        }
        clock_t end = clock();
        double time_scalar = ((double)(end - start)) / CLOCKS_PER_SEC * 10.0; // ms per iteration
        
        // Benchmark AVX2 (if available)
        double time_avx2 = 0;
        if (has_avx2) {
            start = clock();
            for (int iter = 0; iter < 100; iter++) {
                gemv_iq4nl_avx2(output_avx2, weights, input, M, N);
            }
            end = clock();
            time_avx2 = ((double)(end - start)) / CLOCKS_PER_SEC * 10.0;
        }
        
        // Benchmark AVX-512
        start = clock();
        for (int iter = 0; iter < 100; iter++) {
            gemv_iq4nl_avx512(output_avx512, weights, input, M, N);
        }
        end = clock();
        double time_avx512 = ((double)(end - start)) / CLOCKS_PER_SEC * 10.0;
        
        // Verify correctness
        float max_error = 0.0f;
        for (uint32_t i = 0; i < M; i++) {
            float error = fabsf(output_avx512[i] - output_scalar[i]);
            if (error > max_error) max_error = error;
        }
        
        printf("  Scalar:  %.3f ms\n", time_scalar);
        if (has_avx2) {
            printf("  AVX2:    %.3f ms (%.2fx speedup)\n", time_avx2, time_scalar / time_avx2);
        }
        printf("  AVX-512: %.3f ms (%.2fx speedup vs scalar, %.2fx vs AVX2)\n", 
               time_avx512, time_scalar / time_avx512, time_avx2 / time_avx512);
        printf("  Max error: %.6f %s\n\n", max_error, max_error < 0.01 ? "✓" : "✗");
        
        free(weights);
        free(input);
        free(output_scalar);
        free(output_avx2);
        free(output_avx512);
    }
    
    return 0;
}
