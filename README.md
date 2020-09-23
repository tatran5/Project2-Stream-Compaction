CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Thy (Tea) Tran 
  * [LinkedIn](https://www.linkedin.com/in/thy-tran-97a30b148/), [personal website](https://tatran5.github.io/), [email](thytran316@outlook.com)
* Tested on: Windows 10, i7-8750H @ 2.20GHz 22GB, GTX 1070

# Scan, Stream Compaction and Radix Sort
![].(img/main.PNG)

## Performance Analysis

The runtime of the naive method on GPU is worst, followed by that of CPU, then GPU work-efficient method, and finally by thrust. 

Both the naive and work-efficient method on the GPU only rely on global memory instead of ultilizing shared memory, which potentially lead to much slower runtime than thrust method (and CPU for the naive method.) Naive and work-efficient methods also do not use up the potential of warp partitioning or memory coalescing, which are features that thrust may have to reduce its runtime. 

![](img/Scan runtime.PNG)

The same explanation above is applicable to the below as well. 

![](img/Stream compaction runtime.PNG)


Commparisons of GPU Scan implementations: