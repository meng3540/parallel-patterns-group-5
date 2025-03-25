# 📊 GPU Occupancy Analysis

This section summarizes how each optimization impacts **achieved occupancy**, a key metric in CUDA performance profiling.

---


## 🔎 What is Achieved Occupancy?

**Achieved Occupancy** tells us how well the GPU is able to schedule and utilize its resources (like threads, warps, shared memory) to hide memory latency.

- Higher occupancy → better ability to **keep the GPU busy** while waiting on memory.
- Low occupancy can lead to idle GPU cores and slower performance.

---

## 📈 Achieved Occupancy Results

| Version        | Achieved Occupancy [%] | Active Warps per SM | Theoretical Max [%] |
|----------------|-------------------------|----------------------|----------------------|
| Basic          | 92.90                   | 29.73                | 100                  |
| Constant Mem   | 92.91                   | 29.73                | 100                  |
| Shared + Tiling| 99.10                   | 31.71                | 100                  |

---

## 🧠 Analysis

### ⚪ Constant Memory with Caching (92.90% → 92.91%)
- **Impact:** Minimal improvement
- **Why:** Moving the filter to constant memory **doesn’t use extra resources** (registers or shared memory).
- Block size and thread scheduling stay the same → no change in how the GPU handles threads.
- Result: Only a **very tiny** improvement in occupancy (basically the same).

### 🟢 Shared Memory with Tiling (92.91% → 99.10%)
- **Impact:** **Big increase in occupancy**
- **Why it worked:**
  - Smaller block size (e.g., 8×8 threads) allows **more blocks** to fit on each SM.
  - Minimal shared memory use means no bottlenecks.
  - GPU can schedule **more warps**, better hide memory latency, and keep cores busier.

---

## ✅ Summary

| Optimization Step              | Occupancy Change | Main Benefit                         |
|-------------------------------|------------------|--------------------------------------|
| Global → Constant Memory      | ~0%              | Faster filter reads (not occupancy)  |
| Constant → Shared + Tiling    | ↑ ~6%            | More efficient resource usage        |

📌 **Key Insight**: The biggest occupancy gain comes from **reducing resource usage per block** (not just moving data). This allows better scheduling and helps the GPU hide memory latency effectively.

---
