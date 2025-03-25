# 🧱 Tiled & Shared Memory CUDA Gaussian Blur

The CUDA Gaussian blur project now has **Second optimal** version. Along with **constant memory** for the filter, it makes significant kernel memory access pattern and performance improvement using **shared memory** and **tiling**.

---


## 📸 What It Does

- Loads and converts an image to grayscale
- Applies a **7x7 Gaussian blur** filter
- Uses **tiling** and **shared memory** to reduce global memory access
- Saves the result as a PNG
- Measures and prints performance metrics

---

## 🔧 Optimizations Used

### ✅ **Shared Memory (Tiling)**

- Every thread block loads a **tile** of the input picture into **shared memory**.
- Repeating values already in shared memory helps to lower sluggish global memory accesses.

### ✅ **Constant Memory**

- The 7x7 filter is stored in fast **read-only constant memory (`__constant__`)**.

### 🔁 **Overlap Handling**

- Every block loads a tile more than the output size to include the **filter radius**.
- Only valid output threads write results.

---

## 💻 Sample Output

```
Blurred image saved to C:\Users\N01538486\Desktop\blurred_output_tiled.png

Memory Bandwidth Measurements:
Kernel Execution Time:       0.009 ms
Total Execution Time:        1777.756 ms
Host to Device Bandwidth:    0.000 GB/s
Device to Host Bandwidth:    0.002 GB/s
Kernel Memory Bandwidth:    295.975 GB/s
Total Effective Bandwidth:   0.002 GB/s
Total Bytes Transferred:     3.762 MB
```

⚠️ Note: H2D/D2H bandwidth is low due to CPU-side image I/O dominating total time.

---

## 🚀 How It Works

### 1. Image Loading & Conversion

Image loaded with `stb_image.h` is grayscale transformed using only the first channel.

### 2. Tiling and Shared Memory

The kernel computes overlap using a `TILE_DIM`, (say, 4) and filter radius. Every block dumps an image tile into shared memory:

```cpp
__shared__ float N_s[TILE_DIM][TILE_DIM];
```

### 3. Convolution Using Shared Data

Only valid threads perform the convolution using the shared tile:

```cpp
Pvalue += F[fRow * FILTER_SIZE + fCol] * N_s[row_offset][col_offset];
```

### 4. Filter in Constant Memory

The 7x7 Gaussian filter is declared and copied into device constant memory:

```cpp
__constant__ float F[FILTER_SIZE * FILTER_SIZE];
cudaMemcpyToSymbol(F, h_filter, filter_bytes);
```

### 5. Kernel Launch Configuration

Blocks are launched with `TILE_DIM × TILE_DIM` threads and a grid that covers the image with `OUT_TILE_DIM` spacing to account for overlap.

---

## 📊 Performance Summary

| Metric                      | Value         |
|----------------------------|---------------|
| Kernel Execution Time      | 0.009 ms      |
| Total Execution Time       | 1777.756 ms   |
| Kernel Memory Bandwidth    | 295.975 GB/s  |
| Total Bytes Transferred    | 3.762 MB      |
| Effective Bandwidth        | 0.002 GB/s    |

🏎️ **Fastest kernel time** so far thanks to efficient memory access.

---

## 🆚 Compared to Previous Versions

| Feature                 | Global Memory | Constant Memory | Shared + Tiled |
|-------------------------|----------------|------------------|------------------|
| Filter Memory           | Global         | Constant          | Constant          |
| Input Access Pattern    | Global         | Global            | Shared            |
| Output Computation      | Global         | Global            | Shared            |
| Kernel Time             | ~1 ms          | ~1 ms             | **0.009 ms**      |
| Kernel Memory Bandwidth | ~89 GB/s       | ~28 GB/s          | **~296 GB/s**     |

✅ Shared memory and tiling give the biggest gain.

---

## 🛠 Requirements

- CUDA-capable NVIDIA GPU
- CUDA Toolkit installed
- `stb_image.h`, `stb_image_write.h`

---

## ✅ Summary

This version minimizes global memory access by means of **tiling and shared memory**. For best performance, the Gaussian filter is likewise kept in **constant memory**.

Of all versions so far, it offers the **lowest kernel execution time** and **highest memory bandwidth**. Ideal for real-time processing pipelines or massive convolution chores.

---
