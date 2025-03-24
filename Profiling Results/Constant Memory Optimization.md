# ⚡ Profiling Results – Constant Memory Optimization

This section explains how moving the filter to **constant memory** gives a small performance boost in a simple and clear way.

---


## ⚙️ What Changed

### ✅ Main Idea:

Instead of keeping the filter in **global memory** (like in the basic version), we move it to **constant memory** using:

```cpp
__constant__ float F[...];
```

### Why?  
Because **constant memory** is:
- **Faster** than global memory for small, read-only data
- **Cached** and designed to work well when **many threads read the same value**

And in this case, all threads use the **same filter** — so constant memory is a perfect fit!

---

## 🧵 What Each Thread Does

- Threads still read 9 image pixels from **global memory** (this part is unchanged)
- But now they read 9 filter values from **constant memory**, not global memory

So we reduce **some of the slow memory accesses**.

---

## 🔍 What Makes It Better

### 🚀 Faster Filter Accesses
- In the basic version, filter values came from slow global memory.
- Now, they come from fast constant memory (cached).
- Since all threads use the same filter values, the GPU can **broadcast** the values quickly from the cache.

### 📉 Lower Latency = Faster Execution
- Filter reads are no longer a bottleneck — they're quick and efficient.

---

## 📈 Performance Numbers

| Metric                  | Basic Version | Constant Memory Version |
|-------------------------|----------------|--------------------------|
| Kernel Time             | 1.074 ms       | 1.055 ms                 |
| Memory Bandwidth Used   | 26.480 GB/s    | 25.115 GB/s              |
| Speedup (Step)          | —              | ~1.8% faster (1.018×)    |

### So What’s the Catch?

- The speedup is **small**.
- Why? Because **most memory traffic still comes from the image data**, not the filter.
- The filter is tiny (just 9 values), so even optimizing it doesn’t change much.

---

## ✅ Summary – Why This is a Win (Even if a Small One)

| ✅ Before                        | ✅ After                         |
|----------------------------------|----------------------------------|
| Threads read filter from global memory | Threads read filter from fast constant memory |
| Slower access                    | Much faster broadcast reads     |
| W×H×9 slow reads                 | W×H×9 fast cached reads         |

So this optimization **helps**, but the **real bottleneck** (input image reads) still remains.

---

## 🧠 Final Thoughts

Even though the improvement is only ~2%, this is an important step. It shows how **small changes in memory usage** can lead to more efficient GPU code.

When you’re working with small data used by all threads — **constant memory** is your best friend 💡

---
