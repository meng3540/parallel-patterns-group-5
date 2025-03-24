#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define FILTER_RADIUS 3
#define FILTER_SIZE (2 * FILTER_RADIUS + 1)

__constant__ float F[FILTER_SIZE * FILTER_SIZE];

__global__ void convolution_2D_kernel_opt(float* input, float* output, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        float sum = 0.0f;
        for (int fRow = 0; fRow < FILTER_SIZE; fRow++) {
            for (int fCol = 0; fCol < FILTER_SIZE; fCol++) {
                int iRow = row - FILTER_RADIUS + fRow;
                int iCol = col - FILTER_RADIUS + fCol;
                if (iRow >= 0 && iRow < height && iCol >= 0 && iCol < width) {
                    sum += F[fRow * FILTER_SIZE + fCol] * input[iRow * width + iCol];
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
    cudaEvent_t total_start, total_stop, kernel_start, kernel_stop;
    cudaEventCreate(&total_start);
    cudaEventCreate(&total_stop);
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);

    float total_milliseconds = 0, kernel_milliseconds = 0;
    cudaEventRecord(total_start);

    const char* input_path = "IMG2.jpg";
    int width, height, channels;
    unsigned char* image = stbi_load(input_path, &width, &height, &channels, 0);
    if (!image) {
        printf("Error loading image from %s\n", input_path);
        return 1;
    }

    float* h_input = (float*)malloc(width * height * sizeof(float));
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            h_input[i * width + j] = image[i * width * channels + j * channels] / 255.0f;
        }
    }

    print_matrix(h_input, width, height, "Original Image");

    float* h_output = (float*)malloc(width * height * sizeof(float));

    float h_filter[FILTER_SIZE * FILTER_SIZE] = {
        1.0f / 16, 2.0f / 16, 1.0f / 16,
        2.0f / 16, 4.0f / 16, 2.0f / 16,
        1.0f / 16, 2.0f / 16, 1.0f / 16
    };

    float *d_input, *d_output;
    cudaMalloc(&d_input, width * height * sizeof(float));
    cudaMalloc(&d_output, width * height * sizeof(float));

    size_t input_bytes = width * height * sizeof(float);
    size_t output_bytes = width * height * sizeof(float);
    size_t filter_bytes = FILTER_SIZE * FILTER_SIZE * sizeof(float);
    size_t host_to_device_bytes = input_bytes + filter_bytes;
    size_t device_to_host_bytes = output_bytes;

    size_t kernel_reads = width * height * FILTER_SIZE * FILTER_SIZE * sizeof(float);
    size_t kernel_writes = width * height * sizeof(float);
    size_t kernel_total_bytes = kernel_reads + kernel_writes;

    cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(F, h_filter, filter_bytes);

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    cudaEventRecord(kernel_start);
    convolution_2D_kernel_opt<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    cudaEventRecord(kernel_stop);
    cudaEventSynchronize(kernel_stop);

    cudaEventElapsedTime(&kernel_milliseconds, kernel_start, kernel_stop);
    cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);

    print_matrix(h_output, width, height, "Blurred Image");

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
    } else {
        printf("Blurred image saved to %s\n", output_path);
    }

    cudaEventRecord(total_stop);
    cudaEventSynchronize(total_stop);
    cudaEventElapsedTime(&total_milliseconds, total_start, total_stop);

    float total_seconds = total_milliseconds / 1000.0f;
    float kernel_seconds = kernel_milliseconds / 1000.0f;

    float host_to_device_bw = (host_to_device_bytes / (1024.0f * 1024.0f * 1024.0f)) / total_seconds;
    float device_to_host_bw = (device_to_host_bytes / (1024.0f * 1024.0f * 1024.0f)) / total_seconds;
    float kernel_bw = (kernel_total_bytes / (1024.0f * 1024.0f * 1024.0f)) / kernel_seconds;
    float total_bw = ((host_to_device_bytes + device_to_host_bytes + kernel_total_bytes) /
                      (1024.0f * 1024.0f * 1024.0f)) / total_seconds;

    printf("\nMemory Bandwidth Measurements:\n");
    printf("Kernel Execution Time: %.3f ms\n", kernel_milliseconds);
    printf("Total Execution Time: %.3f ms\n", total_milliseconds);
    printf("Host to Device Bandwidth: %.3f GB/s\n", host_to_device_bw);
    printf("Device to Host Bandwidth: %.3f GB/s\n", device_to_host_bw);
    printf("Kernel Memory Bandwidth: %.3f GB/s\n", kernel_bw);
    printf("Total Effective Bandwidth: %.3f GB/s\n", total_bw);
    printf("Total Bytes Transferred: %.3f MB\n",
           (host_to_device_bytes + device_to_host_bytes + kernel_total_bytes) / (1024.0f * 1024.0f));

    // Clean up
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
