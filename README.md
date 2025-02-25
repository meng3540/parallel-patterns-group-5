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

## Basic Algorithm Description (Simplified for Beginners)  
2D Convolution is a process used to **modify** or **extract features** from an image by applying a small grid (called a **kernel**) over different sections of the image. This operation is widely used in image processing and deep learning.  

### **Step 1: Understanding the Image and Kernel**  
- An **image** is made up of tiny squares called **pixels** arranged in a grid (matrix).  
- A **kernel (or filter)** is a small matrix (grid) with specific numbers that determine how the image will be changed.  
- The goal is to **apply the kernel to the image** and create a new, transformed version of it.  

Example:  
If the image is a **5x5 pixel grid** and the kernel is a **3x3 grid**, we will slide the kernel over the image and perform calculations at each step.  

### **Step 2: Sliding the Kernel Over the Image**  
- We **place** the kernel at the top-left corner of the image.  
- We then **move it** across the image, one step at a time (left to right, top to bottom), like a scanning window.  
- This process is called **kernel sliding**.  

### **Step 3: Element-wise Multiplication**  
- At each position, we **multiply** the numbers in the kernel with the corresponding numbers in the image.  
- This is called **element-wise multiplication** because each number in the kernel is multiplied by the pixel underneath it.  

Example:  
Let’s say our kernel looks like this:  
![image](https://github.com/user-attachments/assets/8f8e5aeb-8503-44db-892e-ba7c0a0694e4)


If this kernel is placed over a section of the image, each pixel in that section gets **multiplied** by the corresponding number in the kernel.  

### **Step 4: Summation (Adding the Values Together)**  
- After multiplying, we **add up all the values** to get a **single new pixel** in the output image.  
- This summed value replaces the **center pixel** in that section of the image.  

### **Step 5: Moving the Kernel (Stride & Padding)**  
- **Stride:** Determines **how much** the kernel moves after each step.  
  - A stride of **1** means the kernel moves **one pixel at a time**.  
  - A stride of **2** means it moves **two pixels at a time** (faster, but lower detail).  
- **Padding:**  
  - If we want the output image to be the **same size** as the input, we add **extra pixels (usually zeros)** around the image borders.  
  - This allows the kernel to process **edge pixels** that would otherwise be ignored.  

### **Final Step: Repeating the Process**  
- This **sliding, multiplying, summing, and moving** process is repeated **until every pixel in the image is processed**.  
- The result is a **new transformed image** where features like edges, textures, or patterns have been enhanced or modified.  

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

