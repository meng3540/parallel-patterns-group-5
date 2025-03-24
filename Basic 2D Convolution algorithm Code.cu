//Global Memory basic Code

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define FILTER_RADIUS 3
#define FILTER_SIZE (2 * FILTER_RADIUS + 1)

__global__ void convolution_2D_kernel(float* input, float* output, float* filter, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        float sum = 0.0f;
        for (int fRow = 0; fRow < FILTER_SIZE; fRow++) {
            for (int fCol = 0; fCol < FILTER_SIZE; fCol++) {
                int iRow = row - FILTER_RADIUS + fRow;
                int iCol = col - FILTER_RADIUS + fCol;
                if (iRow >= 0 && iRow < height && iCol >= 0 && iCol < width) {
                    sum += filter[fRow * FILTER_SIZE + fCol] * input[iRow * width + iCol];
                }
            }
        }
        output[row * width + col] = sum;
    }
}

void print_matrix(float* matrix, int width, int height, const char* name) {
    printf("%s Matrix:\n", name);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%.2f ", matrix[i * width + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    // Initialize CUDA events for timing
    cudaEvent_t start, stop, kernelStart, kernelStop, h2dStart, h2dStop, d2hStart, d2hStop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&kernelStart);
    cudaEventCreate(&kernelStop);
    cudaEventCreate(&h2dStart);
    cudaEventCreate(&h2dStop);
    cudaEventCreate(&d2hStart);
    cudaEventCreate(&d2hStop);

    // Record start of total execution
    cudaEventRecord(start, 0);

    // Load image from specified path
    const char* input_path = "IMG2.jpg";
    int width, height, channels;
    unsigned char* image = stbi_load(input_path, &width, &height, &channels, 0);
    if (!image) {
        printf("Error loading image from %s\n", input_path);
        return 1;
    }

    // Convert to float array (using first channel for grayscale)
    float* h_input = (float*)malloc(width * height * sizeof(float));
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            h_input[i * width + j] = image[i * width * channels + j * channels] / 255.0f;
        }
    }

    // Print original image matrix
    print_matrix(h_input, width, height, "Original Image");

    // Allocate memory
    float* h_output = (float*)malloc(width * height * sizeof(float));
    float* h_filter = (float*)malloc(FILTER_SIZE * FILTER_SIZE * sizeof(float));

    // Define Gaussian blur filter
    float gaussian_filter[FILTER_SIZE * FILTER_SIZE] = {
        1.0f / 16, 2.0f / 16, 1.0f / 16,
        2.0f / 16, 4.0f / 16, 2.0f / 16,
        1.0f / 16, 2.0f / 16, 1.0f / 16
    };
    for (int i = 0; i < FILTER_SIZE * FILTER_SIZE; i++) {
        h_filter[i] = gaussian_filter[i];
    }

    // Device memory allocation
    float* d_input, * d_output, * d_filter;
    cudaMalloc(&d_input, width * height * sizeof(float));
    cudaMalloc(&d_output, width * height * sizeof(float));
    cudaMalloc(&d_filter, FILTER_SIZE * FILTER_SIZE * sizeof(float));

    // Copy data to device (Host to Device)
    cudaEventRecord(h2dStart, 0);
    cudaMemcpy(d_input, h_input, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, FILTER_SIZE * FILTER_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(h2dStop, 0);
    cudaEventSynchronize(h2dStop);

    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    cudaEventRecord(kernelStart, 0);
    convolution_2D_kernel << <gridDim, blockDim >> > (d_input, d_output, d_filter, width, height);
    cudaEventRecord(kernelStop, 0);
    cudaEventSynchronize(kernelStop);

    // Copy result back to host (Device to Host)
    cudaEventRecord(d2hStart, 0);
    cudaMemcpy(h_output, d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(d2hStop, 0);
    cudaEventSynchronize(d2hStop);

    // Print blurred image matrix
    print_matrix(h_output, width, height, "Blurred Image");

    // Convert back to unsigned char and save
    unsigned char* output_image = (unsigned char*)malloc(width * height);
    for (int i = 0; i < width * height; i++) {
        float val = h_output[i];
        if (val < 0.0f) val = 0.0f;
        if (val > 1.0f) val = 1.0f;
        output_image[i] = (unsigned char)(val * 255.0f);
    }

    const char* output_path = "C:\\Users\\N01538486\\Desktop\\blurred_output.png";
    if (!stbi_write_png(output_path, width, height, 1, output_image, width)) {
        printf("Error saving image to %s\n", output_path);
    }
    else {
        printf("Blurred image saved to %s\n", output_path);
    }

    // Record end of total execution
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate timings
    float kernelTime, totalTime, h2dTime, d2hTime;
    cudaEventElapsedTime(&kernelTime, kernelStart, kernelStop);
    cudaEventElapsedTime(&totalTime, start, stop);
    cudaEventElapsedTime(&h2dTime, h2dStart, h2dStop);
    cudaEventElapsedTime(&d2hTime, d2hStart, d2hStop);

    // Calculate total bytes transferred
    size_t h2dBytes = width * height * sizeof(float) + FILTER_SIZE * FILTER_SIZE * sizeof(float); // Input + Filter
    size_t d2hBytes = width * height * sizeof(float); // Output
    size_t totalBytes = h2dBytes + d2hBytes;

    // Calculate bandwidths
    float h2dBandwidth = (h2dBytes / (h2dTime / 1000.0f)) / 1e9; // GB/s
    float d2hBandwidth = (d2hBytes / (d2hTime / 1000.0f)) / 1e9; // GB/s
    float totalEffectiveBandwidth = (totalBytes / (totalTime / 1000.0f)) / 1e9; // GB/s

    // Estimate kernel-to-memory bandwidth (approximation based on reads and writes in the kernel)
    size_t kernelReads = width * height * FILTER_SIZE * FILTER_SIZE * sizeof(float); // Each pixel reads FILTER_SIZE * FILTER_SIZE elements
    size_t kernelWrites = width * height * sizeof(float); // Each pixel writes one element
    size_t kernelMemoryBytes = kernelReads + kernelWrites;
    float kernelMemoryBandwidth = (kernelMemoryBytes / (kernelTime / 1000.0f)) / 1e9; // GB/s

    // Print metrics
    printf("\nPerformance Metrics:\n");
    printf("Kernel Execution Time: %.3f ms\n", kernelTime);
    printf("Total Execution Time: %.3f ms\n", totalTime);
    printf("Host to Device Bandwidth: %.3f GB/s\n", h2dBandwidth);
    printf("Device to Host Bandwidth: %.3f GB/s\n", d2hBandwidth);
    printf("Kernel to Memory Bandwidth: %.3f GB/s\n", kernelMemoryBandwidth);
    printf("Total Effective Bandwidth: %.3f GB/s\n", totalEffectiveBandwidth);
    printf("Total Bytes Transferred: %.3f MB\n", totalBytes / (1024.0f * 1024.0f));

    // Clean up
    stbi_image_free(image);
    free(h_input);
    free(h_output);
    free(h_filter);
    free(output_image);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(kernelStart);
    cudaEventDestroy(kernelStop);
    cudaEventDestroy(h2dStart);
    cudaEventDestroy(h2dStop);
    cudaEventDestroy(d2hStart);
    cudaEventDestroy(d2hStop);

    return 0;
}
