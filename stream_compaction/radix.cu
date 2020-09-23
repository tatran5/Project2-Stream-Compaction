#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <iostream>

namespace StreamCompaction {
    namespace Radix {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void computeBoolsForBits(int n, int bit, int* data, int* bools0, int* bools1) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index >= n) {
            return;
          }
          int bitVal = (data[index] >> bit) & 1;
          bools0[index] = 1 - bitVal;
          bools1[index] = bitVal;
        }

        __global__ void computeTotalFalses(int n, int* totalFalses, int* bools0, int* falseAddreses) {
          totalFalses[0] = bools0[n - 1] + falseAddreses[n - 1];
        }

        __global__ void computeAddressForTrueKeys(int n, const int* totalFalses, int* trueAddresses, const int* falseAddreses) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index >= n) {
            return;
          }
          trueAddresses[index] = index - falseAddreses[index] + totalFalses[0];
        }

        __global__ void computeAddresses(int n, int* bools1, const int* trueAddresses, const int* falseAddresses, int* allAddresses) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index >= n) {
            return;
          }
          allAddresses[index] = bools1[index] > 0 ? trueAddresses[index] : falseAddresses[index];
        }
        
        __global__ void scatter(int n, int* odata, int* idata, int* addresses) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index >= n) {
            return;
          }
          odata[addresses[index]] = idata[index];
        }

        void sort(int n, int* odata, const int* idata) {
          // TODO
          if (n < 1) {
            return;
          }
          int blockSize = 128;
          dim3 blockCount((n + blockSize - 1) / blockSize);

          int* dev_idata;
          int* dev_odata;
          int* dev_bools0;
          int* dev_bools1;
          int* dev_falseAddresses;
          int* dev_trueAddresses;
          int* dev_addresses;
          int* dev_totalFalses;

          cudaMalloc((void**)&dev_idata, n * sizeof(int));
          checkCUDAError("cudaMalloc dev_idata failed!");

          cudaMalloc((void**)&dev_odata, n * sizeof(int));
          checkCUDAError("cudaMalloc dev_odata failed!");

          cudaMalloc((void**)&dev_bools0, n * sizeof(int));
          checkCUDAError("cudaMalloc dev_bools0 failed!");

          cudaMalloc((void**)&dev_bools1, n * sizeof(int));
          checkCUDAError("cudaMalloc dev_bools1 failed!");
          
          cudaMalloc((void**)&dev_falseAddresses, n * sizeof(int));
          checkCUDAError("cudaMalloc dev_falseAddresses failed!");

          cudaMalloc((void**)&dev_trueAddresses, n * sizeof(int));
          checkCUDAError("cudaMalloc dev_trueAddresses failed!");

          cudaMalloc((void**)&dev_addresses, n * sizeof(int));
          checkCUDAError("cudaMalloc dev_addresses failed!");

          cudaMalloc((void**)&dev_totalFalses, 1 * sizeof(int));
          checkCUDAError("cudaMalloc dev_totalFalses failed!");

          cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
          checkCUDAError("cudaMemcpy dev_idata failed!");

          int largestNumBit = 0;
          for (int i = 0; i < n; i++) {
            int numBit = (int)log2(idata[i]) + 1;
            if (largestNumBit < numBit) {
              largestNumBit = numBit;
            }
          }
          std::cout << "largest number of bits: " << largestNumBit << std::endl;
          // Compute array which is true/false for bit n
          for (int i = 0; i < largestNumBit; i++) {
            computeBoolsForBits << <blockCount, blockSize >> > (n, i, dev_idata, dev_bools0, dev_bools1);
            StreamCompaction::Efficient::scan(n, dev_falseAddresses, dev_bools0);
            computeTotalFalses << <blockCount, blockSize >> > (n, dev_totalFalses, dev_bools0, dev_falseAddresses);
            computeAddressForTrueKeys << <blockCount, blockSize >> > (n, dev_totalFalses, dev_trueAddresses, dev_falseAddresses);
            computeAddresses << <blockCount, blockSize >> > (n, dev_bools1, dev_trueAddresses, dev_falseAddresses, dev_addresses);
            scatter << <blockCount, blockSize >> > (n, dev_odata, dev_idata, dev_addresses);
            
            int* temp = dev_idata;
            dev_idata = dev_odata;
            dev_odata = temp;
          }

          // Transfer data from gpu to cpu
          cudaMemcpy(odata, dev_idata, sizeof(int) * n, cudaMemcpyDeviceToHost);
          checkCUDAError("cudaMemcpy dev_odata failed!");

          // Cleanup
          cudaFree(dev_addresses);
          checkCUDAError("cudaFree dev_addresses failed!");
          cudaFree(dev_bools0);
          checkCUDAError("cudaFree dev_bools0 failed!");
          cudaFree(dev_bools1);
          checkCUDAError("cudaFree dev_bools1 failed!");
          cudaFree(dev_falseAddresses);
          checkCUDAError("cudaFree dev_falseAddresses failed!");
          cudaFree(dev_idata);
          checkCUDAError("cudaFree dev_idata failed!");
          cudaFree(dev_odata);
          checkCUDAError("cudaFree dev_odata failed!");
          cudaFree(dev_trueAddresses);
          checkCUDAError("cudaFree dev_trueAddresses failed!");
          cudaFree(dev_totalFalses);
          checkCUDAError("cudaFree dev_totalFalses failed!");
        }
    }
}
