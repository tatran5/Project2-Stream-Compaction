#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
          // Compute exclusive prefix sum
            //timer().startCpuTimer();
            // TODO
            if (n < 0) {
              return;
            }
            odata[0] = 0;
            for (int i = 1; i < n; i++) {
              odata[i] = odata[i - 1] + idata[i - 1];
            }
            //timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int o = 0;
            for (int i = 0; i < n; i++) {
              if (idata[i] != 0) {
                odata[o] = idata[i];
                o++;
              }
            }
            timer().endCpuTimer();
            return o;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int* bdata = new int[n];
            for (int i = 0; i < n; i++) {
              bdata[i] = idata[i] != 0 ? 1 : 0;
            }

            int* sdata = new int[n];
            scan(n, sdata, bdata);
            
            int o = 0;
            for (int i = 0; i < n; i++) {
              if (bdata[i] == 1) {
                odata[o] = idata[i];
                o++;
              }
            }
            
            delete[] bdata;
            delete[] sdata;
            
            timer().endCpuTimer();
            return o;
        }
    }
}
