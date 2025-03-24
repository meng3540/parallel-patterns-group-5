# 🔍 CUDA Gaussian Blur (Global Memory)

This project applies a **Gaussian blur** filter to an image using **CUDA parallel programming** with **global memory**. The main goal is to accelerate image processing using the GPU.

---

## 📸 What it Does

- Loads an input image (JPEG).
- Converts it to **grayscale**.
- Applies a **3x3 Gaussian blur** filter using a CUDA kernel.
- Saves the blurred image as a PNG file.
- Measures performance (timings and memory bandwidth).

---

## 🚀 How It Works

### 🧠 1. Image Load
We use the `stb_image.h` library to load the image. Only the **first color channel** is taken to make it grayscale.

```cpp
unsigned char* image = stbi_load(input_path, &width, &height, &channels, 0);
```

---

### 📦 2. Memory Allocation

- Allocate memory on **host (CPU)** and **device (GPU)**.
- The filter and image data are stored as `float` arrays for precision.

---

### 🧮 3. Gaussian Blur Filter

We use a **3x3** Gaussian kernel for blurring:

```cpp
float gaussian_filter[9] = {
    1/16, 2/16, 1/16,
    2/16, 4/16, 2/16,
    1/16, 2/16, 1/16
};
```

This smooths out the image by giving more weight to the center pixel and blending nearby pixels.

---

### 🔁 4. CUDA Kernel Execution

Each CUDA thread processes **one pixel**. It applies the Gaussian filter over its local neighborhood using global memory.

```cpp
__global__ void convolution_2D_kernel(...) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    ...
}
```

---

### ⏱ 5. Performance Measurement

We use **CUDA events** to measure:

- Kernel execution time
- Host-to-device (H2D) and device-to-host (D2H) memory transfer times
- Effective memory bandwidth

This helps analyze how fast the GPU processes the image and moves data.

---

### 🖼 6. Image Save

After applying the filter and copying the data back to host, the blurred image is saved using `stb_image_write.h`.

```cpp
stbi_write_png(output_path, width, height, 1, output_image, width);
```

---

## 📊 Performance Metrics (Example)

```
Kernel Execution Time: 1.23 ms
Total Execution Time: 5.67 ms
Host to Device Bandwidth: 3.45 GB/s
Device to Host Bandwidth: 4.12 GB/s
Kernel to Memory Bandwidth: 89.10 GB/s
Total Effective Bandwidth: 35.60 GB/s
Total Bytes Transferred: 3.45 MB
```

---

## 🧹 Cleanup

All allocated memory (both host and device) is freed at the end to avoid memory leaks.

---

## 📁 Output

- Input: `IMG2.jpg` (must be in the same folder or provide full path)
- Output: `blurred_output.png` (saved to desktop or specified path)

---

## 🛠 Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- `stb_image.h` and `stb_image_write.h` (included in the repo or project folder)

---

## 🙌 Credits

- CUDA programming by [you 🫡]
- Image handling via [stb_image](https://github.com/nothings/stb)

---

## 🔎 ALGORITHM EXPLANATION

This CUDA code performs **2D convolution** using global memory, a fundamental operation in image processing. It's used for applying effects like **blurring**, **sharpening**, and **edge detection**, as well as in **deep learning**.

### 1. Initialization

- Memory is allocated on the **host (CPU)** for:
  - `h_input`: Grayscale version of the input image
  - `h_output`: Result of the convolution
  - `h_filter`: The convolution kernel (Gaussian blur)

- The input image is loaded using `stb_image.h` and converted to grayscale (float values between 0.0 and 1.0).

- The filter used is a **3x3 Gaussian kernel** (smoothing blur), normalized so all values sum to 1.

---

### 2. Memory Transfer (Host ➡️ Device)

- Memory is allocated on the **GPU** for:
  - `d_input`: Image data
  - `d_output`: Result after convolution
  - `d_filter`: Gaussian kernel

- Host-to-Device transfer is performed using `cudaMemcpy()`.

---

### 3. Kernel Execution

- The CUDA kernel `convolution_2D_kernel` is launched with a grid of thread blocks:
  - Each **thread** processes **one pixel** of the output image.
  - It applies the filter to its local region of the input image.
  - Threads run in **parallel**, significantly speeding up processing.

---

### 4. Memory Transfer Back (Device ➡️ Host)

- The blurred result (`d_output`) is copied back to the host as `h_output`.

---

### 5. Image Saving and Cleanup

- The float output is converted back to an `unsigned char` grayscale image.
- It’s saved using `stb_image_write.h` as a PNG.
- All memory (host and device) is freed.
- CUDA events are destroyed after use.

---

## 🧠 KERNEL BREAKDOWN

The kernel `convolution_2D_kernel` is responsible for performing the convolution operation in parallel.

### 1. Thread Indexing

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

- Calculates which pixel each thread is responsible for based on block and thread indices.

---

### 2. Filter Application

- For each thread (pixel), the kernel:
  - Loops over the **filter window**.
  - For valid neighboring pixels, multiplies the filter value with the corresponding image value.
  - Sums the results to get the final pixel value.

```cpp
sum += filter[fRow * FILTER_SIZE + fCol] * input[iRow * width + iCol];
```

---

### 3. Boundary Handling

- Threads skip computations for filter regions that fall **outside** image boundaries (to avoid illegal memory access):

```cpp
if (iRow >= 0 && iRow < height && iCol >= 0 && iCol < width)
```

---

### 4. Output Write

- The final pixel value is stored in the output image:

```cpp
output[row * width + col] = sum;
```

---

## 🖥️ HOST CODE RESPONSIBILITIES

The host (CPU) code manages the following:

- **Image Handling:**
  - Loads a JPEG image.
  - Converts to grayscale.
  - Converts final blurred result back to PNG.

- **Memory Management:**
  - Allocates and frees memory on both host and device.
  - Transfers data between host and GPU.

- **Kernel Launch:**
  - Defines CUDA `dim3` blocks and grids.
  - Launches the convolution kernel.

- **Performance Monitoring:**
  - Uses **CUDA events** to time:
    - Kernel execution
    - Host-to-device copy
    - Device-to-host copy
    - Total execution time

- **Bandwidth Metrics:**
  - Calculates effective memory bandwidths in GB/s.
  - Provides insight into performance bottlenecks.

---

## ✅ TL;DR

This project demonstrates how to use CUDA and global memory to accelerate **Gaussian blurring** on grayscale images. It includes:

- Clean memory management
- Proper kernel launch configuration
- Safe edge handling
- Performance and bandwidth measurement

---
