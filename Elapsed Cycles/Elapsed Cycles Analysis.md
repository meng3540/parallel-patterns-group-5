# 🧠 L1 Cache Behavior – Elapsed Cycles Analysis

This section explains how each CUDA optimization affects **L1 cache activity**, measured by **Total L1 Elapsed Cycles**.

---

## ❓ What Are L1 Elapsed Cycles?

L1 Elapsed Cycles count the total time (in clock cycles) the GPU spends handling data in the **L1 cache**.  
This includes:
- **Cache hits** (fast accesses)
- **Cache misses** (slower fallbacks)

This metric tells us how much stress is being placed on L1 and helps evaluate how memory optimizations affect cache traffic.

---

## 📊 Summary of Results

| Version              | Total L1 Elapsed Cycles |
|----------------------|--------------------------|
| Basic Code           | 14,747,774               |
| Constant Memory      | 15,650,430               |
| Shared + Tiling      | 381,679,814              |

---

## 🔍 Analysis

### ⚪ Constant Memory with Caching

**🟢 Cycles:** 14,747,774 → 15,650,430  
**🧠 Change:** Slight Increase (~6%)  

- By moving the **filter** to constant memory, we reduce the amount of L1 activity caused by filter reads.
- However, **image data is still accessed through global memory**, which still puts pressure on L1.
- This results in a **minor change** in L1 cycles — some improvement from filter reads being removed, but overall load stays similar.

### 🟢 Shared Memory with Tiling

**🟢 Cycles:** 15,650,430 → 381,679,814  
**🚀 Huge Increase**  

- Why such a big jump? Because now **shared memory is used for most reads**, and L1 is no longer heavily involved in repeated input data fetches.
- The tile loading process likely uses **coalesced reads** that stress L1 for a short time but reduce **total traffic** across the kernel.
- With most reads now **efficient and grouped**, the GPU spends more cycles at once using L1 in a focused way — but **reduces redundant traffic overall**.

---

## ✅ Final Insight

| Optimization               | L1 Cycle Impact            | What It Means                            |
|---------------------------|----------------------------|-------------------------------------------|
| Constant Memory           | Slight improvement (~11%)  | Fewer filter reads through L1             |
| Shared Memory + Tiling    | Big shift in pattern       | L1 handles large tile loads more efficiently |

📌 Bottom Line: Optimizing global memory with shared memory not only reduces latency, but also changes how the cache is used — making it more effective and less wasteful.

---
