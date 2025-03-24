### **ALGORITHM EXPLANATION:**

The provided CUDA code does 2D convolution, a crucial operation in image processing and neural networks. Convolution employs a small filter (kernel) on an input picture to extract features, execute blurring, identify edges, or facilitate other modifications.

#### **1. Initialization**
- Allocate memory on both the host (CPU) and device (GPU) for:

  - Input image (`h_input`)
  - Output image (`h_output`)
  - Filter kernel (`h_filter`)
    
- The input image is initialized with values (for testing, it's filled with ones).
  
- The filter is initialized as an averaging filter, where each element is equal to 1/(FILTER_SIZE * FILTER_SIZE), distributing equal weight across all filter elements.

#### **2. Memory Transfer**
- Copy the input image and filter from the host (CPU) to the device (GPU) using cudaMemcpy().

- Allocate memory on the GPU for storing the output image.

#### **3. Kernel Execution**
- The CUDA kernel (convolution_2D_kernel) is launched using a grid of blocks, where:
  
  - Each thread processes one pixel in the output image.
  - The kernel applies the filter to its corresponding region in the input image.
  - Threads operate in parallel, speeding up convolution.

#### **4. Memory Transfer Back**
- After kernel execution, the computed output image is copied back from the device (GPU) to the host (CPU).

#### **5. Cleanup**
- All dynamically allocated memory on both the host and device is freed.


### **KERNEL BREAKDOWN:**

The CUDA kernel (convolution_2D_kernel) performs the convolution operation:

#### **1. Thread Index Calculation**
  - Each thread computes its global row and column indices (row, col) in the output image.
  - These indices determine which pixel the thread processes.

#### **2. Filter Application**
  - The thread applies the filter by looping over the filter elements.
  - It extracts the corresponding region from the input image.
  - Each element of the filter is multiplied by the corresponding pixel value in the input image.
  - The results are summed to compute the final pixel value.

#### **3. Boundary Handling**
  - If a thread attempts to access a pixel outside the image boundaries, it skips that computation.
  - This ensures that threads at the image edges do not read invalid memory.

#### **4. Result Storage**
  - The computed pixel value is stored in the corresponding location in the output image.


### **HOST CODE:**
The Host Code manages: 

  - **Memory Allocation:** Allocates space for input, output, and filter arrays on both CPU and GPU.
  - **Memory Copy:** Transfers input data and filter from CPU to GPU.
  - **Kernel Launch:** Defines grid and block dimensions to efficiently distribute threads.
  - **Result Copy:** Retrieves the computed convolution output from GPU to CPU.
  - **Output Verification:** Prints the top-left 4×4 region of the processed image for validation.
  - **Cleanup:** Frees memory on both host and device.



- [x]  ***Basic 2D Convolution Algorithm***
