# ⚡ Optimized CUDA Gaussian Blur (Using Constant Memory)

This version improves the basic global memory convolution by storing the filter kernel in **`__constant__` memory**, which is much faster for small, read-only data like a Gaussian filter.

---

## 📸 What It Does

- Loads a grayscale image
- Applies a **3x3 Gaussian blur** using an optimized CUDA kernel
- Saves the processed output image
- Displays performance metrics including memory bandwidth

---

## 🔧 Optimization Used

### ✅ **Constant Memory**

Instead of storing the filter in global memory (like in the basic version), it is now stored in fast **read-only constant memory (`__constant__`)**, which:
- Is **cached**, allowing for **faster access** when all threads read the same data.
- Is ideal for **small filters** used in convolution.

```cpp
__constant__ float F[FILTER_SIZE * FILTER_SIZE];
```

---

## 💻 Sample Output

```
Blurred image saved to C:\Users\N01538486\Desktop\blurred_output.png

Memory Bandwidth Measurements:
Kernel Execution Time:       1.055 ms
Total Execution Time:        1757.992 ms
Host to Device Bandwidth:    0.000 GB/s
Device to Host Bandwidth:    0.006 GB/s
Kernel Memory Bandwidth:    28.115 GB/s
Total Effective Bandwidth:   0.016 GB/s
Total Bytes Transferred:     28.211 MB
```

📌 **Note:** The extremely low H2D/D2H bandwidth is due to total execution time including image I/O, which dominates on CPU.

---

## 🚀 How It Works

### 1. Image Preparation

- Image is loaded with `stb_image.h`
- Converted to grayscale and normalized to float values between `0.0` and `1.0`

### 2. CUDA Memory Allocation

- Input (`d_input`) and output (`d_output`) allocated in **global memory**
- Filter kernel copied to `__constant__` memory

```cpp
cudaMemcpyToSymbol(F, h_filter, filter_bytes);
```

### 3. Optimized Kernel

Each thread computes the convolution of a single pixel using the shared kernel:

```cpp
__global__ void convolution_2D_kernel_opt(...) {
    float sum = 0.0f;
    for (...) {
        sum += F[...] * input[...]; // Access filter from constant memory
    }
    output[...] = sum;
}
```

### 4. Performance Metrics

CUDA events measure:
- Kernel execution time
- Bandwidths (H2D, D2H, kernel memory usage)
- Total bytes transferred

---

## 📊 Performance Breakdown

| Metric                      | Value        |
|----------------------------|--------------|
| Kernel Execution Time      | 1.055 ms     |
| Total Execution Time       | 1757.992 ms  |
| Kernel Memory Bandwidth    | 28.115 GB/s  |
| Total Bytes Transferred    | 28.211 MB    |
| Effective Bandwidth        | 0.016 GB/s   |

Compared to the basic version, **kernel memory bandwidth has improved** due to using constant memory for the filter.

---

## 🖥️ Key Differences from Basic Version

| Feature                | Basic Version     | Optimized Version       |
|------------------------|-------------------|--------------------------|
| Filter Storage         | Global Memory     | `__constant__` Memory    |
| Kernel Launch          | Same              | Same                     |
| Performance Gain       | ❌ Low bandwidth   | ✅ Higher kernel bandwidth |
| Memory Access Speed    | Slower (global)   | Faster (cached)          |

---

## 🧹 Cleanup and Notes

- All memory is freed properly at the end
- The output is saved as `blurred_output.png`

---

## ✅ Summary

This optimized CUDA version uses **constant memory** to speed up access to the Gaussian filter. This results in **better memory bandwidth** and performance compared to using only global memory.

Next step? You could explore **shared memory** or **tiling** to reduce global memory accesses even more 🔥

---
