#include "common.h"

void checkCUDAErrorFn(const char *msg, const char *file, int line) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}


namespace StreamCompaction {
    namespace Common {

        /**
         * Maps an array to an array of 0s and 1s for stream compaction. Elements
         * which map to 0 will be removed, and elements which map to 1 will be kept.
         */
        __global__ void kernMapToBoolean(int n, int *bools, const int *idata) {
          // TODO
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index >= n) {
            return;
          }
          bools[index] = idata[index] != 0? 1 : 0;
        }

        /**
         * Performs scatter on an array. That is, for each element in idata,
         * if bools[idx] == 1, it copies idata[idx] to odata[indices[idx]].
         */
        __global__ void kernScatter(int n, int *odata,
                const int *idata, const int *bools, const int *indices) {
          // TODO
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index >= n) {
            return;
          }
          if (bools[index] == 1) {
            odata[indices[index]] = idata[index];
          }
        }

        __global__ void shiftRight(int n, int* idata, int* odata) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index >= n) {
            return;
          }
          if (index == 0) {
            odata[index] = 0;
          }
          else {
            odata[index] = idata[index - 1];
          }
        }

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
    }
}
