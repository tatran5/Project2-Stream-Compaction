#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <memory>
#include <iostream>

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */


        /* Copy initial data over and pad 0's if out of scope of initial size 
         * aka the input array has a smaller initial size than the final array, 
         * and anything larger than index [size of input array] will be 0 in the output array
         */
        __global__ void formatInitData(int initSize, int finalSize, int* data) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x; 
          if (index >= initSize && index < finalSize) {
            data[index] = 0;
          }
        }

        __global__ void add(int n, int ignoreIndexCount, int* odata, const int* idata) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index < ignoreIndexCount) {
            odata[index] = idata[index];
          } else if (index < n) {
            int x1 = idata[index - ignoreIndexCount];
            int x2 = idata[index];
            odata[index] = x1 + x2;
          }
        }

        // Careful with non-power of 2
        void scan(int n, int *odata, const int *idata) {
            if (n < 1) {
              return;
            }
            // Calculate the number of elements the input can be treated as an array with a power of two elements
            int kernelInvokeCount = ilog2ceil(n);
            int n2 = pow(2, kernelInvokeCount);
            
            // Declare data to be on the gpu
            int* dev_odata;
            int* dev_idata;
            std::unique_ptr<int[]> tdata{ new int[n2] };

            // Allocate data to be on the gpu
            cudaMalloc((void**)&dev_odata, n2 * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");
            
            cudaMalloc((void**)&dev_idata, n2 * sizeof(int));
            checkCUDAError("cudaMalloc dev_tdata failed!");

            // Transfer data from cpu to gpu
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_idata failed!");

            timer().startGpuTimer();

            int blockSize = 128;
            dim3 blockCount((n2 + blockSize - 1) / blockSize);

            // Format input data (pad 0s to the closest power of two elements, inclusively)
            StreamCompaction::Common::formatInitData << <blockCount, blockSize >> > (n, n2, dev_idata);

            std::cout << "kernel invoke count: " << kernelInvokeCount << std::endl;

            for (int i = 1; i <= kernelInvokeCount; i++) {
              int ignoreIndexCount = pow(2, i - 1);
              add << <blockCount, blockSize >> > (n2, ignoreIndexCount, dev_odata, dev_idata);

              int* temp = dev_idata;
              dev_idata = dev_odata;
              dev_odata = temp;
            }

            // Shift things to the right to make the inclusive can into exclusive scan
            StreamCompaction::Common::shiftRight<< <blockCount, blockSize >> > (n, dev_idata, dev_odata);

            // Transfer data from gpu to cpu
            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_odata failed!");

            cudaFree(dev_odata);
            checkCUDAError("cudaFree dev_odata failed!");
            cudaFree(dev_idata);
            checkCUDAError("cudaFree dev_idata failed!");
            
            // Calculate the number of blocks and threads per block
            timer().endGpuTimer();
        }
    }
}
