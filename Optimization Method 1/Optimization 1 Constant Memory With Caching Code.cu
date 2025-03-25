#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

// STB image libraries for loading and saving images
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Define filter properties
#define FILTER_RADIUS 3                                 // Radius of filter (e.g., 3 → 7x7)
#define FILTER_SIZE (2 * FILTER_RADIUS + 1)             // Full filter size

// Store filter in fast, read-only constant memory (shared across all threads)
__constant__ float F[FILTER_SIZE * FILTER_SIZE];

// CUDA kernel: performs 2D convolution using constant memory for filter
__global__ void convolution_2D_kernel_opt(float* input, float* output, int width, int height) {
    // Get current thread’s (pixel) position
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Make sure we’re inside image bounds
    if (col < width && row < height) {
        float sum = 0.0f;

        // Loop over the filter window
        for (int fRow = 0; fRow < FILTER_SIZE; fRow++) {
            for (int fCol = 0; fCol < FILTER_SIZE; fCol++) {
                int iRow = row - FILTER_RADIUS + fRow;
                int iCol = col - FILTER_RADIUS + fCol;

                // Check boundary (only read valid pixels)
                if (iRow >= 0 && iRow < height && iCol >= 0 && iCol < width) {
                    sum += F[fRow * FILTER_SIZE + fCol] * input[iRow * width + iCol];
                }
            }
        }

        // Write the final result to output image
        output[row * width + col] = sum;
    }
}

// Print matrix (for debugging)
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
    // CUDA events for timing
    cudaEvent_t total_start, total_stop, kernel_start, kernel_stop;
    cudaEventCreate(&total_start);
    cudaEventCreate(&total_stop);
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);
    float total_milliseconds = 0, kernel_milliseconds = 0;

    cudaEventRecord(total_start); // Start full program timing

    // Load input image
    const char* input_path = "IMG2.jpg";
    int width, height, channels;
    unsigned char* image = stbi_load(input_path, &width, &height, &channels, 0);
    if (!image) {
        printf("Error loading image from %s\n", input_path);
        return 1;
    }

    // Convert image to float grayscale (0.0f - 1.0f)
    float* h_input = (float*)malloc(width * height * sizeof(float));
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            h_input[i * width + j] = image[i * width * channels + j * channels] / 255.0f;
        }
    }

    print_matrix(h_input, width, height, "Original Image");

    // Allocate host memory for output image
    float* h_output = (float*)malloc(width * height * sizeof(float));

    // Define a 3x3 Gaussian filter
    float h_filter[FILTER_SIZE * FILTER_SIZE] = {
        1.0f / 16, 2.0f / 16, 1.0f / 16,
        2.0f / 16, 4.0f / 16, 2.0f / 16,
        1.0f / 16, 2.0f / 16, 1.0f / 16
    };

    // Allocate device memory for input/output
    float *d_input, *d_output;
    cudaMalloc(&d_input, width * height * sizeof(float));
    cudaMalloc(&d_output, width * height * sizeof(float));

    // Measure memory usage for performance metrics
    size_t input_bytes = width * height * sizeof(float);
    size_t output_bytes = width * height * sizeof(float);
    size_t filter_bytes = FILTER_SIZE * FILTER_SIZE * sizeof(float);
    size_t host_to_device_bytes = input_bytes + filter_bytes;
    size_t device_to_host_bytes = output_bytes;
    size_t kernel_reads = width * height * FILTER_SIZE * FILTER_SIZE * sizeof(float);
    size_t kernel_writes = width * height * sizeof(float);
    size_t kernel_total_bytes = kernel_reads + kernel_writes;

    // Copy input image and filter to device
    cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(F, h_filter, filter_bytes); // Copy filter to constant memory

    // Configure kernel launch: 16x16 threads per block
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    // Run the kernel
    cudaEventRecord(kernel_start);
    convolution_2D_kernel_opt<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    cudaEventRecord(kernel_stop);
    cudaEventSynchronize(kernel_stop);
    cudaEventElapsedTime(&kernel_milliseconds, kernel_start, kernel_stop);

    // Copy result back to CPU
    cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);
    print_matrix(h_output, width, height, "Blurred Image");

    // Convert float output to unsigned char for saving
    unsigned char* output_image = (unsigned char*)malloc(width * height);
    for (int i = 0; i < width * height; i++) {
        float val = h_output[i];
        if (val < 0.0f) val = 0.0f;
        if (val > 1.0f) val = 1.0f;
        output_image[i] = (unsigned char)(val * 255.0f);
    }

    // Save output image to file
    const char* output_path = "C:\\Users\\N01538486\\Desktop\\blurred_output.png";
    if (!stbi_write_png(output_path, width, height, 1, output_image, width)) {
        printf("Error saving image to %s\n", output_path);
    } else {
        printf("Blurred image saved to %s\n", output_path);
    }

    // Measure total execution time
    cudaEventRecord(total_stop);
    cudaEventSynchronize(total_stop);
    cudaEventElapsedTime(&total_milliseconds, total_start, total_stop);

    // Calculate bandwidth metrics
    float total_seconds = total_milliseconds / 1000.0f;
    float kernel_seconds = kernel_milliseconds / 1000.0f;

    float host_to_device_bw = (host_to_device_bytes / (1024.0f * 1024.0f * 1024.0f)) / total_seconds;
    float device_to_host_bw = (device_to_host_bytes / (1024.0f * 1024.0f * 1024.0f)) / total_seconds;
    float kernel_bw = (kernel_total_bytes / (1024.0f * 1024.0f * 1024.0f)) / kernel_seconds;
    float total_bw = ((host_to_device_bytes + device_to_host_bytes + kernel_total_bytes)
                      / (1024.0f * 1024.0f * 1024.0f)) / total_seconds;

    // Print performance results
    printf("\nMemory Bandwidth Measurements:\n");
    printf("Kernel Execution Time: %.3f ms\n", kernel_milliseconds);
    printf("Total Execution Time: %.3f ms\n", total_milliseconds);
    printf("Host to Device Bandwidth: %.3f GB/s\n", host_to_device_bw);
    printf("Device to Host Bandwidth: %.3f GB/s\n", device_to_host_bw);
    printf("Kernel Memory Bandwidth: %.3f GB/s\n", kernel_bw);
    printf("Total Effective Bandwidth: %.3f GB/s\n", total_bw);
    printf("Total Bytes Transferred: %.3f MB\n",
           (host_to_device_bytes + device_to_host_bytes + kernel_total_bytes) / (1024.0f * 1024.0f));

    // Free memory
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    cudaEventDestroy(total_start);
    cudaEventDestroy(total_stop);
    stbi_image_free(image);
    free(h_input);
    free(h_output);
    free(output_image);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
