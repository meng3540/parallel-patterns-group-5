# 🔍 CUDA Gaussian Blur (Global Memory)

The purpose of this project is to apply a Gaussian blur filter to a picture using CUDA parallel programming and global memory. The major goal is to speed up the processing of images using the graphics processing unit (GPU).

---

## 📸 What it Does

- Loads an input image (`IMG2.jpg`).
- Converts it to **grayscale** using only the first color channel.
- Applies a **3x3 Gaussian blur** filter using a CUDA kernel.
- Saves the blurred image as a PNG file (`blurred_output.png`).
- Displays **timing and performance metrics**.

---

## 💻 Sample Output

Here’s a real example of the console output after running the program:

```
Blurred image saved to C:\Users\N01538486\Desktop\blurred_output.png

Performance Metrics:
Kernel Execution Time:       1.074 ms
Total Execution Time:        1758.171 ms
Host to Device Bandwidth:    4.027 GB/s
Device to Host Bandwidth:    5.651 GB/s
Kernel to Memory Bandwidth: 262.480 GB/s
Total Effective Bandwidth:   0.001 GB/s
Total Bytes Transferred:     1.005 MB
```

This gives insight into how efficiently the GPU and memory are being used.

---

## 🚀 How It Works

### 🧠 1. Image Load

The image is loaded using `stb_image.h` and converted to grayscale using only the first channel:

```cpp
unsigned char* image = stbi_load(input_path, &width, &height, &channels, 0);
h_input[i * width + j] = image[i * width * channels + j * channels] / 255.0f;
```

---

### 📦 2. Memory Allocation

Memory is allocated on the host and device for:

- Input image (`h_input`, `d_input`)
- Output image (`h_output`, `d_output`)
- Filter kernel (`h_filter`, `d_filter`)

---

### 🧮 3. Gaussian Blur Filter

The blur uses this normalized **3x3 Gaussian kernel**:

```cpp
float gaussian_filter[9] = {
    1/16, 2/16, 1/16,
    2/16, 4/16, 2/16,
    1/16, 2/16, 1/16
};
```

This smooths the image by giving more weight to the center pixel and blending nearby pixels.

---

### 🔁 4. CUDA Kernel

Each thread computes **one pixel** of the output using the kernel below:

```cpp
__global__ void convolution_2D_kernel(...) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    float sum = 0.0f;
    for (int fRow = 0; fRow < FILTER_SIZE; fRow++) {
        for (int fCol = 0; fCol < FILTER_SIZE; fCol++) {
            ...
            sum += filter[...] * input[...];
        }
    }
    output[row * width + col] = sum;
}
```

**Boundary Handling:** Threads at the edges skip invalid memory access.

---

### 🧪 5. Performance Metrics

CUDA events are used to measure:
- Kernel time
- Host ↔ Device transfer time
- Memory bandwidth (in GB/s)
- Total bytes transferred

---

### 🖼 6. Image Save

The float output is converted to `unsigned char` and saved using:

```cpp
stbi_write_png(output_path, width, height, 1, output_image, width);
```

---

## 🧠 KERNEL EXPLANATION

- Each thread processes one pixel.
- It applies the 3x3 filter to a local region of the input image.
- Skips out-of-bound accesses to prevent crashes.

---

## 📊 INTERPRETING PERFORMANCE OUTPUT

From the sample output:

- **Kernel Time** is fast: `~1 ms`
- **Total Execution Time** is high due to image loading/saving overhead.
- **Memory Bandwidth** for kernel: `~262 GB/s` indicates efficient GPU memory usage.
- **Total Effective Bandwidth** is low due to slow CPU-side operations.

---

## 🛠 REQUIREMENTS

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- `stb_image.h` and `stb_image_write.h` included

---

## ✅ TL;DR

This project demonstrates a CUDA-accelerated Gaussian blur using global memory. It features:

- Efficient thread-level parallelism
- Clean host-device memory flow
- Performance measurement tools

---
