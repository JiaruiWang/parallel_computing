//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/device_vector.h>
#include <cstdio>

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
__global__ void histogram_kernel(unsigned int* const d_inputVals_t,
                                 unsigned int* const d_inputPos_t,
                                 unsigned int* const d_digits,
                                 unsigned int* const d_histogram,
                                 unsigned int digit,
                                 unsigned int i,
                                 int numElems)
{
  unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= numElems) return;

  d_digits[index] = (d_inputVals_t[index] & digit) >> i;
  unsigned int bin = d_digits[index];
  unsigned int step = 1;
  atomicAdd(&(d_histogram[bin]), step);

  // if (index == 0)
  // {
  //   printf("numElems = %d\n", numElems);
  //   printf("d_digits[%d] = %d\n", index, d_digits[index]);
  //   printf("hist : 0 = %d, 1 = %d\n", d_histogram[0], d_histogram[1]);
  // }
}

__global__ void block_exclusive_scan_kernel(unsigned int* d_digits,
                                            unsigned int* d_digits_pos,
                                            unsigned int* d_blelloch_inter,
                                            const unsigned int numElems)
{
  extern __shared__ unsigned int sh_mem[];
  int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  int block_index = threadIdx.x;
  if( global_index >= numElems) sh_mem[block_index] = 0;
  else sh_mem[block_index] = d_digits[global_index];
__syncthreads();
//   if (global_index == 1024)
//   {
//     for (int i = 0; i < 1024; ++i)
//     {
//       printf("sh_mem_0[%d] =%u, \n", i, sh_mem[i]);
//     }
//   }
// __syncthreads();
    // Blelloch scan
  

  for (int i = 2; i <= 1024; i *= 2)
  {
    if ((block_index+1)%i == 0)
    {
      int temp = sh_mem[block_index];
      int temp2 = sh_mem[block_index - i/2];
      sh_mem[block_index] = temp + temp2;
      
    }
    __syncthreads();

  }

  if (block_index == 1023)
  {
   sh_mem[1023] = 0;
  }
  __syncthreads();

  
  for (int i = 1024; i >= 2; i = i/2)
  {
    __syncthreads();
    if ((block_index + 1) % i == 0)
    {
      int temp = sh_mem[block_index];
      int temp2 =sh_mem[block_index - i/2];
      sh_mem[block_index - i/2] = temp;
      sh_mem[block_index] = temp + temp2;
    }
    __syncthreads();

  }
  
  if (block_index == 1023)
  {
   d_blelloch_inter[blockIdx.x] = sh_mem[block_index] + d_digits[global_index];
   // printf("blockidx.x =%d\n", blockIdx.x);
  }
  __syncthreads();
  if (global_index < numElems) d_digits_pos[global_index] = sh_mem[block_index];
  __syncthreads();

//   if (global_index == 220000)
//   {
//     for (int i = 0; i < 1024; ++i)
//     {
//       printf("sh_mem_0[%d] =%u, \n", i, sh_mem[i]);
//     }
//   }
// __syncthreads();
//   if (global_index == 1024)
//   {
//     // int sum = 0;
//     // for (int i = 0; i < 216; ++i)
//     // {
//     //   printf("d_blelloch_inter[%d] = %u, \n", i, d_blelloch_inter[i]);
//     //   sum += d_blelloch_inter[i];
//     // }
//     // printf("sum = %d\n", sum);
//     // for (int i = 0; i < numElems; ++i)
//     // {
//     //   printf("%d:%u\n", i, d_digits_pos[i]);
//     // }
//   }
// __syncthreads();
}


__global__ void self_exclusive_scan_kernel(unsigned int* d_blelloch_inter,
                                           const unsigned int numElems)
{
  extern __shared__ unsigned int sh_mem[];
  int block_index = blockDim.x * blockIdx.x + threadIdx.x;
  if (block_index >= numElems) sh_mem[block_index] = 0;
  else sh_mem[block_index] = d_blelloch_inter[block_index];
  __syncthreads();

  for (int i = 2; i <= 1024; i *= 2)
  {
    if ((block_index+1)%i == 0)
    {
      int temp = sh_mem[block_index];
      int temp2 = sh_mem[block_index - i/2];
      sh_mem[block_index] = temp + temp2;
      
    }
    __syncthreads();

  }

  if (block_index == 1023)
  {
   sh_mem[1023] = 0;
  }
  __syncthreads();

  
  for (int i = 1024; i >= 2; i = i/2)
  {
    __syncthreads();
    if ((block_index + 1) % i == 0)
    {
      int temp = sh_mem[block_index];
      int temp2 =sh_mem[block_index - i/2];
      sh_mem[block_index - i/2] = temp;
      sh_mem[block_index] = temp + temp2;
    }
    __syncthreads();

  }

  d_blelloch_inter[block_index] = sh_mem[block_index];
  __syncthreads();
  // if (block_index == 0)
  // {
  //   for (int i = 0; i < numElems; ++i)
  //   {
  //     printf("d_blelloch_inter[%d] = %u\n", i, d_blelloch_inter[i]);
  //   }
  // }
}

__global__ void add_inter_kernel(unsigned int* d_blelloch_inter,
                                 unsigned int* d_digits_pos,
                                 unsigned int numElems)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= numElems) return;

  d_digits_pos[index] = d_digits_pos[index] + d_blelloch_inter[blockIdx.x];

  // if (index == 0)
  // {
  //   for (int i = 0; i < numElems; ++i)
  //   {
  //     printf("%d:%u\n", i, d_digits_pos[i]);
  //   }
  // }
}

__global__ void exclusive_scan_kernel(unsigned int* d_digits,
                                      unsigned int* d_digits_pos,
                                      const unsigned int numElems)
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


  
  // blelloch scan
  // d_digits_pos[index] = d_digits[index];
  // __syncthreads();

  // for (int i = 2; i <= 262144; i *= 2)
  // {
  //   if ((index+1 + 41664)%i == 0)
  //   {
  //     unsigned int temp = d_digits_pos[index];
  //     unsigned int temp2 = 0;

  //     if (index - i/2 >= 0)
  //       temp2 = d_digits_pos[index - i/2];
  //     d_digits_pos[index] = temp + temp2;
  //   }
  //   __syncthreads();
  // }
  // __syncthreads();
  // if (index == 262144-1-41664)
  // {
  //  d_digits_pos[numElems-1] = 0;
  // }
  // __syncthreads();

  // for (int i = 262144; i >= 2; i = i/2)
  // {
  //   if ((index + 1 + 41664) % i == 0)
  //   {
  //     unsigned int temp = d_digits_pos[index];
  //     unsigned int temp2 = 0;

  //     if (index - i/2 >= 0)
  //       temp2 = d_digits_pos[index - i/2];
      
  //     d_digits_pos[index] = temp + temp2;

  //     if (index - i/2 >= 0)
  //       d_digits_pos[index - i/2] = temp;
  //   }
  //   __syncthreads();
  // }

  // Hillis/Steele
  // d_digits_pos[index] = d_digits[index];
  // __syncthreads();

  // for (int i = 2; i <= 512; i *= 2)
  // {
  //   unsigned int current = d_digits_pos[index];
  //   unsigned int pre = 0;
  //   if (index - i/2 >=0)
  //     pre = d_digits_pos[index - i/2];
  //   __syncthreads();
  //   d_digits_pos[index] = current + pre;
  //   __syncthreads();
  // }
  // d_digits_pos[index] = d_digits_pos[index] - d_digits[index];
  // __syncthreads();

  // if (index == 0)
  // {
  //   // int sum = 0;
  //   for (unsigned int i = 0; i < numElems; ++i)
  //   {
  //     printf("%u:%u ", i, d_digits_pos[i]);
  //     // sum += d_digits_pos[i];
  //   }
  //   // printf("\nsum = %d\n", sum);
  // }

}

__global__ void switch_ones_zeros(unsigned int* d_digits,
                                  unsigned int* d_digits_reverse,
                                  const size_t numElems)
{
  unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= numElems) return;

  if (d_digits[index] == 1)
  {
    d_digits_reverse[index] = 0;
  }
  if (d_digits[index] == 0)
  {
    d_digits_reverse[index] = 1;
  }
  __syncthreads();
  // if (index == 0)
  // {
  //   unsigned int sum = 0;
  //   for (int i = 0; i < numElems; ++i)
  //   {
  //     printf("%u: %u->%u ", i, d_digits[i], d_digits_reverse[i]);
  //     sum += d_digits_reverse[i];
  //   }
  //   printf("sum = %u\n", sum);
  // }
}

__global__ void add_1_0_pos(unsigned int* d_digits_1_0_pos,
                            unsigned int* d_digits,
                            unsigned int* d_digits_1_pos,
                            unsigned int* d_digits_reverse,
                            unsigned int* d_digits_0_pos,
                            const size_t numElems)
{
  unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= numElems) return;

  if (d_digits[index] == 1)
  {
    d_digits_1_0_pos[index] = d_digits_1_pos[index];
  }
  __syncthreads();
  if (d_digits_reverse[index] == 1)
  {
    d_digits_1_0_pos[index] = d_digits_0_pos[index]; 
  }
  __syncthreads();

  // if (index == 1)
  // {
  //   unsigned int sum = 0;
  //   for (int i = 0; i < numElems; ++i)
  //   {
  //     printf("%u:%u ", i, d_digits_1_0_pos[i]);
  //     sum += d_digits_1_0_pos[i];
  //   }
  //   printf("\nsum = %u\n", sum);
  // }
}

__global__ void add_prefix_to_pos(unsigned int* d_digits_ab_pos,
                                  // unsigned int* d_digits_1_0_pos,
                                  unsigned int* d_digits,
                                  // unsigned int* d_digits_1_pos,
                                  // unsigned int* d_digits_reverse,
                                  // unsigned int* d_digits_0_pos,
                                  unsigned int* d_prefix_sum,
                                  const size_t numElems)
{
  unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= numElems) return;

  if (d_digits[index] == 1)
  {
    d_digits_ab_pos[index] = d_digits_ab_pos[index] + d_prefix_sum[1];
  }
  __syncthreads();


  // if (index == 0)
  // {
  //   for (int i = 0; i < numElems; ++i)
  //   {
  //     printf("%u:%u ", i, d_digits_ab_pos[i]);
  //   }
  // }
}

__global__ void move_kernel(unsigned int* d_inputVals_t,
                            unsigned int* d_inputPos_t,
                            unsigned int* d_outputVals_t,
                            unsigned int* d_outputPos_t,
                            unsigned int* d_digits_ab_pos,
                            const size_t numElems)
{
  unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= numElems) return;
  
  d_outputVals_t[d_digits_ab_pos[index]] = d_inputVals_t[index];
  d_outputPos_t[d_digits_ab_pos[index]] = d_inputPos_t[index];

  // if (index == 0)
  // {
  //   for (unsigned int i = 0; i < numElems; ++i)
  //   {
  //     if (i == 220474)
  //     {
  //       printf("d_inputVals_t[%u] = %u\nd_inputPos_t[%u] = %u\nd_digits_ab_pos[%u] = %u\n", 
  //         i, d_inputVals_t[i], i, d_inputPos_t[i], i, d_digits_ab_pos[i]);
  //     }
  //     if (i == 220479)
  //     {
  //       printf("d_outputVals_t[%u] = %u\nd_outputPos_t[%u] = %u\n", i, d_outputVals_t[i], i, d_outputPos_t[i]);
  //     }
  //   }
  // }
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
  
  unsigned int* d_inputVals_t;
  checkCudaErrors(cudaMalloc(&d_inputVals_t, sizeof(unsigned int) * numElems));
  checkCudaErrors(cudaMemcpy(d_inputVals_t, d_inputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
  unsigned int* d_inputPos_t;
  checkCudaErrors(cudaMalloc(&d_inputPos_t, sizeof(unsigned int) * numElems));
  checkCudaErrors(cudaMemcpy(d_inputPos_t, d_inputPos, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
  unsigned int* d_outputVals_t;
  checkCudaErrors(cudaMalloc(&d_outputVals_t, sizeof(unsigned int) * numElems));
  checkCudaErrors(cudaMemcpy(d_outputVals_t, d_outputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
  unsigned int* d_outputPos_t;
  checkCudaErrors(cudaMalloc(&d_outputPos_t, sizeof(unsigned int) * numElems));
  checkCudaErrors(cudaMemcpy(d_outputPos_t, d_outputPos, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));

  unsigned int* d_digits;
  checkCudaErrors(cudaMalloc(&d_digits, sizeof(unsigned int) * numElems));
  unsigned int* d_histogram;
  checkCudaErrors(cudaMalloc(&d_histogram, sizeof(unsigned int) * 2));
  
  unsigned int* d_prefix_sum;
  checkCudaErrors(cudaMalloc(&d_prefix_sum, sizeof(unsigned int) * 2));
  
  unsigned int* d_digits_1_pos;
  checkCudaErrors(cudaMalloc(&d_digits_1_pos, sizeof(unsigned int) * numElems));
  unsigned int* d_digits_0_pos;
  checkCudaErrors(cudaMalloc(&d_digits_0_pos, sizeof(unsigned int) * numElems));
  unsigned int* d_digits_reverse;
  checkCudaErrors(cudaMalloc(&d_digits_reverse, sizeof(unsigned int) * numElems));
  unsigned int* d_digits_1_0_pos;
  checkCudaErrors(cudaMalloc(&d_digits_1_0_pos, sizeof(unsigned int) * numElems));
  unsigned int* d_digits_ab_pos;
  checkCudaErrors(cudaMalloc(&d_digits_ab_pos, sizeof(unsigned int) * numElems));
  unsigned int* d_blelloch_inter;
  checkCudaErrors(cudaMalloc(&d_blelloch_inter, sizeof(unsigned int) * blocks));

  unsigned int one = 1;
  unsigned int thirtyTwo = (unsigned int)sizeof(unsigned int) * 8;
  for (unsigned int i = 0; i < thirtyTwo; ++i)
  {
    // 1)
    // printf("%u, one << i = %u, sizeof(unsigned int) * 8 = %u\n", i, one << i,(unsigned int)sizeof(unsigned int) * 8);
    checkCudaErrors(cudaMemset(d_histogram, 0, sizeof(unsigned int) * 2));
    checkCudaErrors(cudaMemset(d_prefix_sum, 0, sizeof(unsigned int) * 2));
    checkCudaErrors(cudaMemset(d_digits_1_pos, 0, sizeof(unsigned int) * numElems));
    
    histogram_kernel<<<blocks, threads>>>(d_inputVals_t, d_inputPos_t, d_digits, d_histogram, one << i, i, numElems);
    // printf("1\n");
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    // 2)
    // printf("2\n");
    checkCudaErrors(cudaMemcpy(&(d_prefix_sum[1]), &(d_histogram[0]), sizeof(unsigned int), cudaMemcpyDeviceToDevice));

    // unsigned int* h_prefix_sum = (unsigned int*)malloc(sizeof(unsigned int) * 2);
    // checkCudaErrors(cudaMemcpy(h_prefix_sum, d_prefix_sum, sizeof(unsigned int) * 2, cudaMemcpyDeviceToHost));
    // printf("d_prefix_sum[0]= %d, d_prefix_sum[1] = %d\n", h_prefix_sum[0], h_prefix_sum[1]);
    // cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // 3)
    checkCudaErrors(cudaMemcpy(d_digits_1_pos, d_digits, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemset(d_blelloch_inter, 0, sizeof(unsigned int) * blocks));
    block_exclusive_scan_kernel<<<blocks, threads, sizeof(unsigned int) * 1024>>>(d_digits, d_digits_1_pos, d_blelloch_inter, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    self_exclusive_scan_kernel<<<1, threads, sizeof(unsigned int) * 1024>>>(d_blelloch_inter, 216);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    add_inter_kernel<<<blocks, threads>>>(d_blelloch_inter, d_digits_1_pos, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    // unsigned int* h_blelloch_inter = (unsigned int*)malloc(sizeof(unsigned int) * blocks);
    // checkCudaErrors(cudaMemcpy(h_blelloch_inter, d_blelloch_inter, sizeof(unsigned int) * blocks, cudaMemcpyDeviceToHost));
    // for (int i = 0; i < blocks; ++i)
    // {
    //   printf("d_blelloch_inter[%d]= %d,", i, d_blelloch_inter[i]);
    // }
   
    // cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError()); 

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


    switch_ones_zeros<<<blocks, threads>>>(d_digits, d_digits_reverse, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    // exclusive_scan_kernel<<<blocks, threads>>>(d_digits_reverse, d_digits_0_pos, numElems);
    checkCudaErrors(cudaMemset(d_blelloch_inter, 0, sizeof(unsigned int) * blocks));
    block_exclusive_scan_kernel<<<blocks, threads, sizeof(unsigned int) * 1024>>>(d_digits_reverse, d_digits_0_pos, d_blelloch_inter, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    self_exclusive_scan_kernel<<<1, threads, sizeof(unsigned int) * 1024>>>(d_blelloch_inter, 216);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    add_inter_kernel<<<blocks, threads>>>(d_blelloch_inter, d_digits_0_pos, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


    add_1_0_pos<<<blocks, threads>>>(d_digits_1_0_pos, d_digits, d_digits_1_pos, d_digits_reverse, d_digits_0_pos, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


    // 4)

    checkCudaErrors(cudaMemcpy(d_digits_ab_pos, d_digits_1_0_pos, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
    add_prefix_to_pos<<<blocks, threads>>>(d_digits_ab_pos, d_digits, d_prefix_sum, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    move_kernel<<<blocks, threads>>>(d_inputVals_t, d_inputPos_t, d_outputVals_t, d_outputPos_t, d_digits_ab_pos, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    std::swap(d_inputVals_t, d_outputVals_t);
    std::swap(d_inputPos_t, d_outputPos_t);
  }
  
  checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals_t, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos_t, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaFree(d_blelloch_inter));
  checkCudaErrors(cudaFree(d_inputVals_t));
  checkCudaErrors(cudaFree(d_inputPos_t));
  checkCudaErrors(cudaFree(d_outputVals_t));
  checkCudaErrors(cudaFree(d_outputPos_t));
  checkCudaErrors(cudaFree(d_digits));
  checkCudaErrors(cudaFree(d_histogram));
  checkCudaErrors(cudaFree(d_prefix_sum));
  checkCudaErrors(cudaFree(d_digits_1_pos));
  checkCudaErrors(cudaFree(d_digits_0_pos));
  checkCudaErrors(cudaFree(d_digits_reverse));
  checkCudaErrors(cudaFree(d_digits_1_0_pos));
  checkCudaErrors(cudaFree(d_digits_ab_pos));
}
