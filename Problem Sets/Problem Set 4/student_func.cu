//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

#include <stdio.h>
__global__ void histogram_kernel(unsigned int* const d_inputVals,
                                 unsigned int* const d_inputPos,
                                 unsigned int* const d_digits,
                                 unsigned int* const d_histogram,
                                 int digit,
                                 int numElems)
{
  unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= numElems) return;

  d_digits[index] = d_inputVals[index] & digit;
  unsigned int bin = d_digits[index];
  unsigned int step = 1;
  atomicAdd(&(d_histogram[bin]), step);

  if (index == 0)
  {
    printf("numElems = %d\n", numElems);
    printf("d_digits[%d] = %d\n", index, d_digits[index]);
    printf("hist : 0 = %d, 1 = %d\n", d_histogram[0], d_histogram[1]);
  }
}

__global__ void relative_exclusive_scan(unsigned int* const d_inputVals,
                                        unsigned int* const d_inputPos,
                                        unsigned int* const d_outputVals,
                                        unsigned int* const d_outputPos,
                                        unsigned int* d_prefix_sum,
                                        const size_t numElems)
{
  unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= numElems) return;

  //unsigned int bin = d_inputVals[index] & 1;
  

}

__global__ void exclusive_scan_kernel(unsigned int* d_digits,
                                      unsigned int* d_digits_pos,
                                      const size_t numElems)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if( index >= numElems) return;

  // simple scan implementation
  int sum = 0;
  for (int i = 1; i <= index; ++i)
  {
    sum += d_digits[i-1];
  }
  __syncthreads();
  d_digits_pos[index] = sum;
  __syncthreads();

  if (index == 0)
  {
    for (unsigned int i = 0; i < numElems; ++i)
    {
      printf("%u:%u ", i, d_digits_pos[i]);
    }
  }
  
  // blelloch scan
  // for (int i = 2; i <= numBins; i *= 2)
  // {
  //   if ((index+1)%i == 0)
  //   {
  //     int temp = sh_bin[index];
  //     int temp2 = sh_bin[index - i/2];
  //     sh_bin[index] = temp + temp2;
      
  //   }
  //   __syncthreads();
  // }

  // if (index == numBins-1)
  // {
  //  sh_bin[numBins-1] = 0;
  // }
  // __syncthreads();

  // for (int i = numBins; i >= 2; i = i/2)
  // {
  //   if ((index + 1) % i == 0)
  //   {
  //     int temp = sh_bin[index];
  //     int temp2 =sh_bin[index - i/2];
  //     sh_bin[index - i/2] = temp;
  //     sh_bin[index] = temp + temp2;
  //   }
  //   __syncthreads();
  // }
  // bin[index] = sh_bin[index];

  // __syncthreads();
}



void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
  //TODO
  //PUT YOUR SORT HERE
  int threads = 1024;
  int blocks = ceil(1.0 * numElems/ threads);

  unsigned int* d_digits;
  unsigned int* d_histogram;
  unsigned int* d_prefix_sum;
  checkCudaErrors(cudaMalloc(&d_digits, sizeof(unsigned int) * numElems));
  checkCudaErrors(cudaMalloc(&d_histogram, sizeof(unsigned int) * 2));
  checkCudaErrors(cudaMalloc(&d_prefix_sum, sizeof(unsigned int) * 2));
  checkCudaErrors(cudaMemset(d_histogram, 0, sizeof(unsigned int) * 2));
  checkCudaErrors(cudaMemset(d_prefix_sum, 0, sizeof(unsigned int) * 2));
  // 1)
  histogram_kernel<<<blocks, threads>>>(d_inputVals, d_inputPos, d_digits, d_histogram, 0x1, numElems);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  // 2)
  checkCudaErrors(cudaMemcpy(&(d_prefix_sum[1]), &(d_histogram[0]), sizeof(unsigned int), cudaMemcpyDeviceToDevice));

  unsigned int* h_prefix_sum = (unsigned int*)malloc(sizeof(unsigned int) * 2);
  checkCudaErrors(cudaMemcpy(h_prefix_sum, d_prefix_sum, sizeof(unsigned int) * 2, cudaMemcpyDeviceToHost));
  printf("d_prefix_sum[0]= %d, d_prefix_sum[1] = %d\n", h_prefix_sum[0], h_prefix_sum[1]);

  // 3)
  unsigned int* d_digits_pos;
  checkCudaErrors(cudaMalloc(&d_digits_pos, sizeof(unsigned int) * numElems));
  exclusive_scan_kernel<<<blocks, threads>>>(d_digits, d_digits_pos, numElems);

  checkCudaErrors(cudaFree(d_digits));
  checkCudaErrors(cudaFree(d_histogram));


}
