# Parallel Patterns in Computing

## Introduction 
Parallel patterns in computing are common ways to break down tasks so that multiple things can happen at the same time.
Instead of doing everything one by one, computers can split up the work and finish it faster.
There are different types of parallel patterns, like **task parallelism**, where different tasks run independently, and **data parallelism**, where the same task happens to different pieces of data at the same time. 
These patterns help computers work better, especially with multi-core processors and powerful graphics cards (GPUs).  

These patterns are super useful in things like video editing, gaming, artificial intelligence, and scientific research.
For example, when editing a video, instead of applying effects one by one, parallel processing can apply them to multiple frames at once, making the process much faster. 
Without parallel patterns, tasks like running big simulations or training AI models would take forever.  

Using both CPUs and GPUs together (heterogeneous computing) makes parallel processing even better. 
CPUs are great for handling small, complex tasks, while GPUs are built for crunching a lot of similar calculations quickly.
When they work together, big problems can be solved much faster, like real-time video processing, weather predictions, or training deep learning models.
This teamwork helps computers run heavy tasks more efficiently and saves time.

# We as group number 5 are working on Parallel Computation Pattern: 2D Convolution:

## Overview & Applications  
2D Convolution is a core computational pattern widely used in **image processing, computer vision, deep learning, and signal processing**. In this pattern, a small matrix called a **kernel (or filter)** is applied to a larger 2D data array (such as an image) to extract features, detect edges, blur, sharpen, or perform other transformations. Its applications include:  

- **Image Processing:** Enhancing images through edge detection, blurring, and sharpening.  
- **Computer Vision:** Enabling object recognition, facial detection, and motion tracking.  
- **Deep Learning:** Forming the backbone of Convolutional Neural Networks (CNNs) which learn hierarchical features for tasks like classification and segmentation.  
- **Medical Imaging:** Improving diagnostic images (e.g., MRI, CT scans) through noise reduction and detail enhancement.  

## Basic Algorithm Description  
The 2D convolution process involves a systematic, sliding window approach over the input data:  

- **Kernel Sliding:**  
  The kernel moves across the image, positioning itself over different segments.  
- **Element-wise Multiplication:**  
  For each position, every element of the kernel is multiplied by the corresponding pixel value in the covered area of the image.  
- **Summation:**  
  The multiplied values are summed to produce a single output pixel, creating a new image that highlights specific features.  
- **Stride & Padding:**  
  - **Stride:** Defines the step size of the kernel movement.  
  - **Padding:** Adds extra pixels around the image border to control the output size.  

This algorithm is executed repeatedly for every location in the image, making it inherently parallelizable.  

## Rationale for Using Parallel Processing & Hardware  
2D Convolution is computationally demanding due to the vast number of multiplication and addition operations required, especially for high-resolution images and deep neural networks. Parallel processing and specialized hardware such as GPUs are crucial because:  

- **Massive Parallelism:**  
  Each convolution operation (for each pixel) is independent, allowing thousands of GPU cores to perform these calculations simultaneously.  
- **Speed & Efficiency:**  
  GPUs are optimized for handling large-scale, parallel computations, significantly reducing processing time compared to sequential CPU execution.  
- **Scalability:**  
  Parallel processing enables real-time processing and training in complex applications like CNNs, making it feasible to handle large datasets and deep architectures.  
- **Optimizations:**  
  Techniques such as tiling, shared memory usage, and parallel reduction further enhance the performance of convolution operations on GPUs.  

---

