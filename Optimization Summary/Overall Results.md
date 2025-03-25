# 🚀 Optimization Summary

| Optimization | Description                          | Execution Time *(ms)* | Memory Bandwidth *(GB/s)* | Step Speedup | Cumulative Speedup |
|--------------|--------------------------------------|------------------------|----------------------------|--------------|---------------------|
| 1.           | Basic Algorithm: Global Memory       | 1.074                  | 26.480                     | 1            | 1                   |
| 2.           | Constant Memory with Caching         | 1.055                  | 25.115                     | 1.018        | 1.018               |
| 3.           | Halo Cells with Shared Memory + Tiling | 0.009                  | 295.975                    | 117.222      | 119.333             |

 
### 📝 Notes:
- **Step Speedup** = (Previous Execution Time) / (Current Execution Time)
- **Cumulative Speedup** = (Initial Execution Time) / (Current Execution Time)
- Shared memory with tiling shows the most **significant improvement** in both time and bandwidth.

---
