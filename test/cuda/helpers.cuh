#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstdint>

// Read n float32 values from a binary file into newly allocated host memory.
// Caller is responsible for free().
inline float* read_host_vec(const char* path, int n) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    float* h = (float*)malloc(n * sizeof(float));
    if (!h) { fprintf(stderr, "malloc failed\n"); exit(1); }
    if (fread(h, sizeof(float), n, f) != (size_t)n) {
        fprintf(stderr, "Short read from %s\n", path); exit(1);
    }
    fclose(f);
    return h;
}

// Read n float32 values from a binary file directly into device memory.
inline float* read_device_vec(const char* path, int n) {
    float* h = read_host_vec(path, n);
    float* d;
    cudaMalloc(&d, n * sizeof(float));
    cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice);
    free(h);
    return d;
}

// Print n host float32 values to stdout.
inline void print_host_vec(float* h, int n) {
    for (int i = 0; i < n; i++) {
        // Preserve NaNs in a spelling NumPy can read on every host CRT.
        if (h[i] != h[i]) printf("nan");
        else printf("%.8g", h[i]);
        if (i < n - 1) printf(" ");
    }
    printf("\n");
    fflush(stdout);  // Keep this row ahead of any subsequent device-printf row.
}

// Copy before printing: CUDA's bounded printf FIFO truncates large matrices.
// Host output also keeps CHECK flags and following vector rows in order.
inline void print_device_vec(float* d, int n) {
    if (n == 0) { printf("\n"); return; }
    float* h = (float*)malloc(n * sizeof(float));
    if (!h) { fprintf(stderr, "malloc failed\n"); exit(1); }
    cudaError_t error = cudaMemcpy(h, d, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        fprintf(stderr, "Cannot copy test result: %s\n", cudaGetErrorString(error));
        free(h);
        exit(1);
    }
    print_host_vec(h, n);
    free(h);
}

// Convenience: alloc n floats on device, zero-initialized.
inline float* alloc_device_vec(int n) {
    float* d;
    cudaMalloc(&d, n * sizeof(float));
    cudaMemset(d, 0, n * sizeof(float));
    return d;
}

// Copy device vector to newly allocated host memory.
inline float* device_to_host(float* d, int n) {
    float* h = (float*)malloc(n * sizeof(float));
    cudaMemcpy(h, d, n * sizeof(float), cudaMemcpyDeviceToHost);
    return h;
}
