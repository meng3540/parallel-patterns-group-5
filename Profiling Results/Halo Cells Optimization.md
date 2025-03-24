# 🧱 Profiling Results – Shared Memory with Tiling & Halo Cells

This section explains the third (and fastest) optimization using **shared memory**, **tiling**, and **constant memory** in simple, easy-to-understand terms.

---

## ⚙️ What Changed

### ✅ What This Version Does:

- The **filter stays in constant memory** (same as before).
- The **input image is split into tiles**, and each tile is loaded into **shared memory**.
- Each thread then reads input pixels from shared memory instead of slow global memory.

### 🧊 What Are Halo Cells?
- When applying a filter near the edge of a tile, you also need some surrounding pixels.
- These extra pixels are called **halo cells**.
- So, a tile that covers a 16×16 output region might actually load an **18×18** input region.

---

## 💡 Why It’s Smarter

### 🔁 Shared Memory = Fast Access

- Global memory = 200–400 cycles latency (slow!)
- Shared memory = just a few cycles (fast!)
- Instead of every thread reading 9 pixels from global memory, **the whole tile is loaded once**, and reused by every thread in the block.

### 📉 Huge Drop in Global Memory Accesses

| Version               | Global Reads per Thread | Total |
|-----------------------|--------------------------|-------|
| Constant Memory       | 9 input reads            | W×H×9 |
| Shared Memory (Tiling)| ~1.26 input reads        | W×H×1.26 |

➡️ That's a **7× reduction** in input reads from global memory!

### ✅ Coalesced Access = Better Bandwidth

- When loading tiles into shared memory, threads read rows in **contiguous memory**, which makes reads more efficient.
- Once the data is in shared memory, threads use it **without repeating reads**.

---

## 📈 Performance Numbers

| Metric                  | Value       |
|-------------------------|-------------|
| Kernel Time             | 0.009 ms    |
| Previous Version Time   | 1.055 ms    |
| Step Speedup            | ~117×       |
| Total Speedup           | ~119×       |
| Bandwidth               | 295.975 GB/s|

🎯 This version is over **100× faster** than the basic one and uses GPU memory **way more efficiently**.

---

## ✅ Summary – Why This is the Best

| ✅ Benefit                                | 💥 Why It Matters                                |
|-------------------------------------------|--------------------------------------------------|
| Way fewer global memory reads             | Less waiting, faster processing                  |
| Shared memory used for image tiles        | Low-latency reads inside a block                 |
| Coalesced access when loading tiles       | Boosts global memory bandwidth                  |
| Filter still in fast constant memory      | Combines benefits of both previous versions      |
| Execution time dropped to 0.009 ms        | GPU now focuses on **fast math**, not memory     |

---

## 🧠 Final Thoughts

This version finally solves the **main bottleneck**: slow, redundant global memory reads.

With shared memory + tiling + constant memory, this kernel becomes **blazing fast**, and your GPU can finally do what it does best — **parallel computing at light speed** ⚡

---
