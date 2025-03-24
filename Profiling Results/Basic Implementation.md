# 🧪 Profiling Results – Basic Implementation (Global Memory)

This section breaks down the performance of the **basic CUDA convolution** version in **simple terms**.

--- 


## ⚙️ Overview

### ✅ What This Version Does:

- Uses **global memory** for both the input image and the filter.
- Each GPU thread is responsible for calculating **one pixel** in the output image.

### 👇 What Each Thread Does:

1. Reads **9 pixels** from the image (a 3×3 block around the target pixel).
2. Reads **9 values** from the filter (Gaussian blur kernel).
3. Multiplies each pixel with its corresponding filter value and sums them up.
4. Writes the final blurred value into the output image.

➡️ That’s a total of **19 memory operations per thread**:  
- 18 reads (9 from image + 9 from filter)  
- 1 write to the output image

---

## 🧨 Why It's Slow (In Simple Words)

Even though the math is fast, the problem is **memory**.

### 1. 🧵 Threads Access Memory Inefficiently

- Threads work on neighboring pixels, but the **memory pattern is scattered**.
- GPUs are fast when threads read memory **together** (coalesced), but this 2D pattern doesn’t always allow that.
- This causes the GPU to **spend more time waiting** for data from memory.

### 2. 🔁 Threads Repeat Work

- Threads often read the **same pixels** as their neighbors (because 3x3 blocks overlap).
- But in global memory, there's **no sharing**—so they read the same thing again and again.
- This wastes memory bandwidth and slows things down.

### 3. 🧂 Filter Is Small, But Still Slow

- All threads use the **same 3x3 filter**, but each one reads it from global memory.
- That’s unnecessary! This data could be kept in **faster memory** (like constant memory), but isn’t in this version.

---

## ⛔ Bottlenecks

| Problem                     | Why It Matters                                  |
|----------------------------|--------------------------------------------------|
| Global Memory is Slow      | GPU memory (global) is slower than shared/constant |
| Reads Are Redundant        | Many threads read the same image pixels          |
| Access Pattern Is Scattered | Makes it harder for the GPU to fetch data quickly |
| Filter Reads Are Repeated  | All threads read the same filter separately      |

---

## 📉 Performance Numbers

- **Execution Time:** 1.074 milliseconds
- **Memory Bandwidth Used:** 26.480 GB/s  
➡️ Much lower than the GPU's potential (e.g., 300+ GB/s)

### Why?  
Because most of the time is spent **waiting for memory**, not doing actual math.

---

## 🧠 Final Thoughts

This version is simple, but not efficient.  
It shows how memory access patterns—**not just math**—can make or break GPU performance.  
That’s why further optimizations (like constant and shared memory) are important 🔧💡

---
