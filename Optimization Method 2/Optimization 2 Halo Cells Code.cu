#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>

// Image load/write libs
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// --- CONFIGURATIONS ---
#define TILE_DIM 4                               // Total tile size (blockDim.x = blockDim.y = 4)
#define FILTER_RADIUS 3                          // Half size of filter (3 → 7x7 filter)
#define FILTER_SIZE (2 * FILTER_RADIUS + 1)      // Full filter size = 7
#define OUT_TILE_DIM (TILE_DIM - 2 * FILTER_RADIUS) // Effective output area per tile

// --- FILTER STORED IN CONSTANT MEMORY (FASTER READ ACCESS) ---
__constant__ float F[FILTER_SIZE * FILTER_SIZE];

// --- CUDA KERNEL ---
__global__ void convolution_cached_tiled_2D_kernel(float* N, float* P, int width, int height) {
  // Shared memory tile including halo cells
  __shared__ float N_s[TILE_DIM][TILE_DIM];

  // Global image coordinates for the current thread
  int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x;
  int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y;

  // Load input image into shared memory (with zero-padding if out of bounds)
  if (row < height && col < width) {
    N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
  } else {
    N_s[threadIdx.y][threadIdx.x] = 0.0f;
  }

  // Synchronize threads before computation
  __syncthreads();

  // Re-adjust global coordinates for convolution result
  col -= FILTER_RADIUS;
  row -= FILTER_RADIUS;

  // Only compute if the current thread maps to a valid output pixel
  if (col >= 0 && col < width && row >= 0 && row < height) {
    float Pvalue = 0.0f;

    // Apply 2D convolution using the filter (from constant memory)
    for (int fRow = 0; fRow < FILTER_SIZE; fRow++) {
      for (int fCol = 0; fCol < FILTER_SIZE; fCol++) {
        int row_offset = threadIdx.y - FILTER_RADIUS + fRow;
        int col_offset = threadIdx.x - FILTER_RADIUS + fCol;

        // Make sure we're accessing valid shared memory indices
        if (row_offset >= 0 && row_offset < TILE_DIM &&
            col_offset >= 0 && col_offset < TILE_DIM) {
          Pvalue += F[fRow * FILTER_SIZE + fCol] * N_s[row_offset][col_offset];
        }
      }
    }

    // Write final blurred pixel to global output
    P[row * width + col] = Pvalue;
  }
}

// --- Helper function to print matrices (for debug) ---
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
  // --- Performance Timing Setup ---
  cudaEvent_t total_start, total_stop, kernel_start, kernel_stop;
  cudaEventCreate(&total_start);
  cudaEventCreate(&total_stop);
  cudaEventCreate(&kernel_start);
  cudaEventCreate(&kernel_stop);
  float total_milliseconds = 0, kernel_milliseconds = 0;

  cudaEventRecord(total_start);

  // --- Load Image from Disk ---
  const char* input_path = "IMG2.jpg";
  int width, height, channels;
  unsigned char* image = stbi_load(input_path, &width, &height, &channels, 0);
  if (!image) {
    printf("Error loading image from %s\n", input_path);
    return 1;
  }

  // --- Convert to Grayscale Float Array ---
  float* h_input = (float*)malloc(width * height * sizeof(float));
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      h_input[i * width + j] = image[i * width * channels + j * channels] / 255.0f;
    }
  }

  print_matrix(h_input, width, height, "Original Image");

  float* h_output = (float*)malloc(width * height * sizeof(float));

  // --- Define a 7x7 Gaussian-like filter ---
  float h_filter[FILTER_SIZE * FILTER_SIZE] = {
    1.0f / 256, 4.0f / 256, 6.0f / 256, 4.0f / 256, 1.0f / 256, 0.0f, 0.0f,
    4.0f / 256, 16.0f / 256, 24.0f / 256, 16.0f / 256, 4.0f / 256, 0.0f, 0.0f,
    6.0f / 256, 24.0f / 256, 36.0f / 256, 24.0f / 256, 6.0f / 256, 0.0f, 0.0f,
    4.0f / 256, 16.0f / 256, 24.0f / 256, 16.0f / 256, 4.0f / 256, 0.0f, 0.0f,
    1.0f / 256, 4.0f / 256, 6.0f / 256, 4.0f / 256, 1.0f / 256, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
  };

  // --- Allocate Device Memory ---
  float *d_input, *d_output;
  cudaMalloc(&d_input, width * height * sizeof(float));
  cudaMalloc(&d_output, width * height * sizeof(float));

  // --- Copy Data to Device ---
  cudaMemcpy(d_input, h_input, width * height * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(F, h_filter, FILTER_SIZE * FILTER_SIZE * sizeof(float)); // Copy filter to constant memory

  // --- Launch Kernel ---
  dim3 blockDim(TILE_DIM, TILE_DIM); // Each block has TILE_DIM × TILE_DIM threads
  dim3 gridDim((width + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

  cudaEventRecord(kernel_start);
  convolution_cached_tiled_2D_kernel<<<gridDim, blockDim>>>(d_input, d_output, width, height);
  cudaEventRecord(kernel_stop);
  cudaEventSynchronize(kernel_stop);
  cudaEventElapsedTime(&kernel_milliseconds, kernel_start, kernel_stop);

  // --- Copy result back to CPU ---
  cudaMemcpy(h_output, d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

  print_matrix(h_output, width, height, "Blurred Image");

  // --- Convert float back to 8-bit and save to image ---
  unsigned char* output_image = (unsigned char*)malloc(width * height);
  for (int i = 0; i < width * height; i++) {
    float val = h_output[i];
    if (val < 0.0f) val = 0.0f;
    if (val > 1.0f) val = 1.0f;
    output_image[i] = (unsigned char)(val * 255.0f);
  }

  const char* output_path = "C:\\Users\\N01538486\\Desktop\\blurred_output_tiled.png";
  if (!stbi_write_png(output_path, width, height, 1, output_image, width)) {
    printf("Error saving image to %s\n", output_path);
  } else {
    printf("Blurred image saved to %s\n", output_path);
  }

  // --- End Timing & Print Bandwidth Info ---
  cudaEventRecord(total_stop);
  cudaEventSynchronize(total_stop);
  cudaEventElapsedTime(&total_milliseconds, total_start, total_stop);

  float total_seconds = total_milliseconds / 1000.0f;
  float kernel_seconds = kernel_milliseconds / 1000.0f;

  size_t input_bytes = width * height * sizeof(float);
  size_t output_bytes = width * height * sizeof(float);
  size_t filter_bytes = FILTER_SIZE * FILTER_SIZE * sizeof(float);
  size_t host_to_device_bytes = input_bytes + filter_bytes;
  size_t device_to_host_bytes = output_bytes;

  size_t tiles_per_row = (width + OUT_TILE_DIM - 1) / OUT_TILE_DIM;
  size_t tiles_per_col = (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM;
  size_t total_tiles = tiles_per_row * tiles_per_col;
  size_t kernel_reads_global = total_tiles * TILE_DIM * TILE_DIM * sizeof(float);
  size_t kernel_writes = width * height * sizeof(float);
  size_t kernel_total_bytes = kernel_reads_global + kernel_writes;

  float host_to_device_bw = (host_to_device_bytes / (1024.0f * 1024.0f * 1024.0f)) / total_seconds;
  float device_to_host_bw = (device_to_host_bytes / (1024.0f * 1024.0f * 1024.0f)) / total_seconds;
  float kernel_bw = (kernel_total_bytes / (1024.0f * 1024.0f * 1024.0f)) / kernel_seconds;
  float total_bw = ((host_to_device_bytes + device_to_host_bytes + kernel_total_bytes) / (1024.0f * 1024.0f * 1024.0f)) / total_seconds;

  printf("\nMemory Bandwidth Measurements:\n");
  printf("Kernel Execution Time: %.3f ms\n", kernel_milliseconds);
  printf("Total Execution Time: %.3f ms\n", total_milliseconds);
  printf("Host to Device Bandwidth: %.3f GB/s\n", host_to_device_bw);
  printf("Device to Host Bandwidth: %.3f GB/s\n", device_to_host_bw);
  printf("Kernel Memory Bandwidth: %.3f GB/s\n", kernel_bw);
  printf("Total Effective Bandwidth: %.3f GB/s\n", total_bw);
  printf("Total Bytes Transferred: %.3f MB\n", (host_to_device_bytes + device_to_host_bytes + kernel_total_bytes) / (1024.0f * 1024.0f));

  // --- Clean Up ---
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
