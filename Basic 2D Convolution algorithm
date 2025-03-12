#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>

// Define filter size (e.g., 3x3 filter with radius 1)
#define FILTER_RADIUS 1 // Defines the radius of the filter
#define FILTER_SIZE (2 * FILTER_RADIUS + 1) // Computes the total size of the filter matrix

// Input and output image dimensions
#define WIDTH 128 // Width of the image
#define HEIGHT 128 // Height of the image


// Kernel function for parallel 2D convolution
__global__ void convolution_2D_kernel(float* input, float* output, float* filter, int width, int height) {
    // Calculate the global row and column for this thread
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;


    // Check if the thread is within the valid image bounds
    if (col < width && row < height) {
        float sum = 0.0f; // Initialize sum for convolution computation


        // Loop over the filter matrix
        for (int fRow = 0; fRow < FILTER_SIZE; fRow++) {
            for (int fCol = 0; fCol < FILTER_SIZE; fCol++) {
                // Compute the corresponding input position
                int iRow = row - FILTER_RADIUS + fRow;
                int iCol = col - FILTER_RADIUS + fCol;

                // Check if the input position is within valid bounds
                if (iRow >= 0 && iRow < height && iCol >= 0 && iCol < width) {
                    sum += filter[fRow * FILTER_SIZE + fCol] * input[iRow * width + iCol];
                }
            }
        }

        // Store the computed sum in the output array
        output[row * width + col] = sum;
    }
}


int main() {
    // Allocate memory for input, output, and filter on the host (CPU)
    float* h_input = (float*)malloc(WIDTH * HEIGHT * sizeof(float));
    float* h_output = (float*)malloc(WIDTH * HEIGHT * sizeof(float));
    float* h_filter = (float*)malloc(FILTER_SIZE * FILTER_SIZE * sizeof(float));


    // Allocate memory for input, output, and filter on the device (GPU)
    float* d_input, * d_output, * d_filter;
    cudaMalloc(&d_input, WIDTH * HEIGHT * sizeof(float));
    cudaMalloc(&d_output, WIDTH * HEIGHT * sizeof(float));
    cudaMalloc(&d_filter, FILTER_SIZE * FILTER_SIZE * sizeof(float));


    // Initialize input data (e.g., filling with ones for simple testing)
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        h_input[i] = 1.0f; // Assigning value 1.0 to all input elements
    }


    // Initialize filter as a simple averaging filter
    for (int i = 0; i < FILTER_SIZE * FILTER_SIZE; i++) {
        h_filter[i] = 1.0f / (FILTER_SIZE * FILTER_SIZE); // Assigning normalized values for an averaging filter
    }


    // Copy data from host (CPU) to device (GPU)
    cudaMemcpy(d_input, h_input, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, FILTER_SIZE * FILTER_SIZE * sizeof(float), cudaMemcpyHostToDevice);


    // Define grid and block dimensions
    dim3 blockDim(16, 16); // Each block contains 16x16 threads
    dim3 gridDim((WIDTH + blockDim.x - 1) / blockDim.x, (HEIGHT + blockDim.y - 1) / blockDim.y);
    // Computes number of blocks needed in x and y direction


    // Launch CUDA kernel for 2D convolution
    convolution_2D_kernel<<<gridDim, blockDim>>>(d_input, d_output, d_filter, WIDTH, HEIGHT);


    // Copy computed output back from device (GPU) to host (CPU)
    cudaMemcpy(h_output, d_output, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);

    // Print a small portion of the output for verification (top-left 4x4 corner)
    printf("Output (top-left 4x4 corner):\n");
    for (int i = 0; i < 4 && i < HEIGHT; i++) {
        for (int j = 0; j < 4 && j < WIDTH; j++) {
            printf("%.2f ", h_output[i * WIDTH + j]);
        }
        printf("\n");
    }


    // Free allocated memory on both host and device
    free(h_input);
    free(h_output);
    free(h_filter);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter);


    return 0; // Program execution completed successfully

}
