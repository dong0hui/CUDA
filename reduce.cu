#include <stdio.h>
#include <fstream>
#include <sstream>
#include <string>
#include <cuda.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>
#include <stdlib.h>
#include <vector>
#include <iostream>

//Simply copied from others' scripts
#if __DEVICE_EMULATION__
#define DEBUG_SYNC __syncthreads();
#else
#define DEBUG_SYNC
#endif

//we use MIN to find minimum
#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif

/*
__device__int__mul24(int x, int y)
calculate the least significant 32 bits of the product of the
least significant 24 bits of x and y. The high order 8 bits are
ignored.
*/
#if (__CUDA_ARCH__ < 200)
#define int_mult(x,y)   __mul24(x,y)
#else
#define int_mult(x,y)   x*y
#endif

#define inf 0x7f800000

/*
check in binary whether x is 10, 100, or 1000,....
*/
bool isPowerOf2 (unsigned int x) {
	return ( (x & (x-1)) == 0);
}

/* 
Why we need this? we want blockSize to be power of 2.
Find the next largest number which is power of 2
e.g, x = 1001 (=9 base 10, 2^3 < 9 < 2^4), --x = 1000
x |= x >> 1 -> 1000 | 0100 = 1100
x |= x >> 2 -> 1100 | 0011 = 1111
++x -> 1111 + 1 = 10000 = 16(base 10)
*/
unsigned int nextPowerOf2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

/*
Reference: The CUDA Handbook: A Comprehensive Guide to GPU Programming, P379
It makes extern __shared__ int sdata[]; more general.
*/
template<class T>
struct SharedMemory {
	__device__ inline operator T *() {
		extern __shared__ int __smem[];
		return (T *) __smem;
	}
	
	__device__ inline operator const T *() const {
		extern __shared__ int __smem[];
		return (T *) __smem;
	}
};

///////////////////////////////////////////////////////////////
//                CUDA KERNEL: Find MIN in each block
///////////////////////////////////////////////////////////////
/*
CUDA reduceMin kernel
Reference please search for Mark Harris from NVIDIA engineering
We use template becuase we don't know type T, blockSize, and nIsPowerOf2 during compilation
g_idata: global input data, needed to be copied to shared memory
g_odata: global output data
n: length of input array
*/
template<class T, unsigned int blockSize, bool nIsPowerOf2>
__global__ void findMin(T *g_idata, T *g_odata, unsigned int n) {
	/*
	we could simply use extern __shared__ int sdata[]; because we know input 
	array is int. But I want to make it general. SharedMemory is struct I defined before
	*/
	T *sdata = SharedMemory<T>();
	
	/*
	To avoid half of threads to be idle during 1st loop, we use (blockSize*2), refer to Mark's slides.
	We have gridDim.x blocks within each grid.
	We have threadIdx.x threads within each block.
	Thread is the basic unit.
	*/
	unsigned int tid = threadIdx.x;
	//e.g. blockSize=4 (blockSize must be power of 2), i=1,2,3,4, 9,10,11,12, 17,18,19,20,....
	unsigned int i = blockIdx.x * (blockSize*2) + threadIdx.x;
	//gridSize: total number of threads in one grid
	unsigned int gridSize = (blockSize * 2) * gridDim.x;
	
	/*
	To avoid idle threads during the 1st loop, we define i as above.
	e.g. In certain grid, we have block0, block1, block2, block3.
	We first find min in block0 and block2. Then we find min of block0&block1
	and block2&block3. In this way, we always use (blockSize) ge threads.
	We repeat above min/thread grid by grid by i += gridSize.
	*/
	T myMin = 99999; //actually we know the numbers between 0-999.
	while (i < n) {
		myMin = MIN(g_idata[i], myMin);
		
		//If i+blockSize is out of bound, no need to consider g_idata[i+blockSize]
		//If nIsPowerOf2 is true, don't worry n would exceed n-blockSize
		if(nIsPowerOf2 || i + blockSize < n) {
			//now we deal with i=5,6,7,8, 13,14,15,16, 21,22,23,24,...
			//In this way, we use all GPUs to do 2-step comparison instead of making half of them idle.
			myMin = MIN(g_idata[i + blockSize], myMin);
		}
		i += gridSize;
	}
	
	/*
	Now suppose we have 512 threads, each threads have min from while-loop.
	We need barrier to make sure all the threads are ready.
	*/
	sdata[tid] = myMin;
	__syncthreads();
	
	/*
	///////////////////////////////////////////////////////////////
	Below to end is Tree Reduction
	///////////////////////////////////////////////////////////////
	Do reduction in shared memory. It is called complete unroll
	Search for "#pragma unroll" in Google. Basically it speeds up
	by avoiding unnecessary check of tid within the same warp.
	*/
	if (blockSize >= 512) {if (tid < 256) {sdata[tid] = myMin = MIN(sdata[tid + 256], myMin);}__syncthreads();}
	if (blockSize >= 256) {if (tid < 128) {sdata[tid] = myMin = MIN(sdata[tid + 128], myMin);}__syncthreads();}
	if (blockSize >= 128) {if (tid < 64)  {sdata[tid] = myMin = MIN(sdata[tid + 64],  myMin);}__syncthreads();}
	
	/*
	NEED TO REVIEW THIS SECTION AGAIN!!!!!!!!
	*/
#if (__CUDA_ARCH__ >= 300 )
    if (tid < 32) {
        if (blockSize >= 64){
            myMin = MIN(sdata[tid + 32], myMin);
        }
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            float tempMyMin = __shfl_down(myMin, offset);
            myMin = MIN(tempMyMin, myMin);
        }
    }
#else
	if ( (blockSize >= 64) && (tid < 32) ) {
		sdata[tid] = myMin = MIN(sdata[tid + 32], myMin);
	}
	__syncthreads();
	if ( (blockSize >= 32) && (tid < 16) ) {
		sdata[tid] = myMin = MIN(sdata[tid + 16], myMin);
	}
	__syncthreads();
	if ( (blockSize >= 16) && (tid < 8) ) {
		sdata[tid] = myMin = MIN(sdata[tid + 8], myMin);
	}
	__syncthreads();
	if ( (blockSize >= 8) && (tid < 4) ) {
		sdata[tid] = myMin = MIN(sdata[tid + 4], myMin);
	}
	__syncthreads();
	if ( (blockSize >= 4) && (tid < 2) ) {
		sdata[tid] = myMin = MIN(sdata[tid + 2], myMin);
	}
	__syncthreads();
	if ( (blockSize >= 2) && (tid < 1) ) {
		sdata[tid] = myMin = MIN(sdata[tid + 1], myMin);
	}
	__syncthreads();
#endif
    __syncthreads();
	if (tid == 0) {
		g_odata[blockIdx.x] = myMin;
	}
}

////////////////////////////////////////////////////////////////////
//                 CUDA Kernel: extract last digits
////////////////////////////////////////////////////////////////////
/* 
This kernel doesn't need reduction
*/
template<class T, unsigned int blockSize>
__global__ void lastDigit(T *g_idata, T *g_odata) {
	//i: index of a number in the whole to-be-processed array
	unsigned int i = blockIdx.x * blockSize + threadIdx.x;
	int lastdig = (int) g_idata[i] % 10;
	g_odata[i] = lastdig;
}

////////////////////////////////////////////////////////////////////
//           GET NUMBER OF BLOCKS AND THREADS                     //
////////////////////////////////////////////////////////////////////
/*
Google getNumBlocksAndThreads you'll find a lot of people use this function
Basically, it is section of codes proposed by NVIDIA.
*/
void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks,
        int maxThreads, int &blocks, int &threads) {

    //get device property, to avoid block/grid size exceed the upper bound
    cudaDeviceProp prop;
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaGetDeviceProperties(&prop, deviceId);

    if (whichKernel < 3) {
        threads = (n < maxThreads) ? nextPowerOf2(n) : maxThreads;
        blocks = (n + threads - 1) / threads;
    } else {
        threads = (n < maxThreads * 2) ? nextPowerOf2((n + 1) / 2) : maxThreads;
        blocks = (n + (threads * 2 - 1)) / (threads * 2);
    }

    if ((float) threads * blocks
            > (float) prop.maxGridSize[0] * prop.maxThreadsPerBlock) {
        printf("n is too large, please choose a smaller number!\n");
    }

    if (blocks > prop.maxGridSize[0]) {
        printf(
                "Grid size <%d> exceeds the device capability <%d>, set block size as %d (original %d)\n",
                blocks, prop.maxGridSize[0], threads * 2, threads);

        blocks /= 2;
        threads *= 2;
    }

    if (whichKernel == 6) {
        blocks = MIN(maxBlocks, blocks);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
template<class T>
void reduceMin(int size, int threads, int blocks, int whichKernel, T *d_idata, T *d_odata) {
	//Define CUDA geometry. threads threads in one block, and blocks blocks in one grid, both along x-direction only
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

    if (isPowerOf2(size)) {
        switch (threads) {
        case 512:
            findMin<T, 512, true> <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 256:
            findMin<T, 256, true> <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 128:
            findMin<T, 128, true> <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 64:
            findMin<T, 64, true> <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 32:
            findMin<T, 32, true> <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 16:
            findMin<T, 16, true> <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 8:
            findMin<T, 8, true> <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 4:
            findMin<T, 4, true> <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 2:
            findMin<T, 2, true> <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 1:
            findMin<T, 1, true> <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;
        }
    } else {
        switch (threads) {
        case 512:
            findMin<T, 512, false> <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 256:
            findMin<T, 256, false> <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 128:
            findMin<T, 128, false> <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 64:
            findMin<T, 64, false> <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 32:
            findMin<T, 32, false> <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 16:
            findMin<T, 16, false> <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 8:
            findMin<T, 8, false> <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 4:
            findMin<T, 4, false> <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 2:
            findMin<T, 2, false> <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 1:
            findMin<T, 1, false> <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;
        }
    }

}

///////////////////////////////////////////////////////////////////////////////
//              Wrapper for lastDigit
///////////////////////////////////////////////////////////////////////////////
/*
template<class T, unsigned int blockSize>
__global__ void lastDigit(T *g_idata, T *g_odata)
*/
template<class T>
void lastDigitWrapper(int threads, int blocks, T *d_idata, T *d_odata) {
	//Block has (threads) ge threads along x-direction
	dim3 dimBlock(threads, 1, 1);
	//Grid has (blocks) ge blocks in it
	dim3 dimGrid(blocks, 1, 1);
	int smemSize = threads * sizeof(T);
	
	//we use template because we don't know the data type in test file and the number of threads	
	switch (threads) {
        case 512:
            lastDigit<T, 512><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata);
            break;

        case 256:
            lastDigit<T, 256><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata);
            break;

        case 128:
            lastDigit<T, 128><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata);
            break;

        case 64:
            lastDigit<T, 64><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata);
            break;

        case 32:
            lastDigit<T, 32><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata);
            break;

        case 16:
            lastDigit<T, 16><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata);
            break;

        case 8:
            lastDigit<T, 8><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata);
            break;

        case 4:
            lastDigit<T, 4><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata);
            break;

        case 2:
            lastDigit<T, 2><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata);
            break;

        case 1:
            lastDigit<T, 1><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata);
            break;
        }
}

////////////////////////////////////////////////////////////////////////////////
//! Compute MIN reduction on CPU
//! We use Kahan summation for an accurate sum of large arrays.
//! http://en.wikipedia.org/wiki/Kahan_summation_algorithm
//!
//! @param data       pointer to input data
//! @param size       number of input data elements
////////////////////////////////////////////////////////////////////////////////
template<class T>
void reduceMINCPU(T *data, int size, T *min)
{
    *min = data[0];

	
    for (int i = 1; i < size; i++)
    {
        T y = data[i];
        T t = MIN(*min, y);
        (*min) = t;
    }
    return;
}

// Instantiate the reduction function for 3 types
template void
reduceMin<int>(int size, int threads, int blocks, int whichKernel, int *d_idata, int *d_odata);

template void
reduceMin<float>(int size, int threads, int blocks, int whichKernel, float *d_idata, float *d_odata);

template void
reduceMin<double>(int size, int threads, int blocks, int whichKernel, double *d_idata, double *d_odata);

template void
lastDigitWrapper<int>(int threads, int blocks, int *d_idata, int *d_odata);
		
////////////////////////////////////////////////////////
//               TEST FUNCTION
////////////////////////////////////////////////////////
unsigned long long int my_min_test(const char *filename) {
    // timers
    //unsigned long long int start;
    unsigned long long int delta = 1;

    int maxThreads = 256;  // number of threads per block
    int whichKernel = 6;
    int maxBlocks = 64;

	//Read input file
	std::vector<float> data;
    std::string line_;
    std::ifstream file_(filename);
    if(file_.is_open()) {
        while (getline(file_, line_)) {
            std::stringstream ss(line_);
            float i;
            while(ss>>i) {
                data.push_back(i);
                if (ss.peek() == ',' || ss.peek() == ' ') {
                    ss.ignore();
                }
            }
        }
        file_.close();
    } 
    int num_els = data.size();
	
	//d_in and d_out are in and out array on Device memory
    float* d_in = NULL;
    float* d_out = NULL;

    //printf("%d elements\n", num_els);
    //printf("%d threads (max)\n", maxThreads);

	//Assign proper number of threads and blocks
    int numBlocks = 0;
    int numThreads = 0;
    getNumBlocksAndThreads(whichKernel, num_els, maxBlocks, maxThreads, numBlocks, numThreads);

	//Allocate shared memory space on CUDA devices
    cudaMalloc((void **) &d_in, num_els * sizeof(float));
    cudaMalloc((void **) &d_out, numBlocks * sizeof(float));

    float* in = (float*) malloc(num_els * sizeof(float));
    float* out = (float*) malloc(numBlocks * sizeof(float));
	std::copy(data.begin(), data.end(), in);
	//in = &data[0];

    // copy data directly to device memory
    cudaMemcpy(d_in, in, num_els * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, out, numBlocks * sizeof(float),cudaMemcpyHostToDevice);

	//<<<>>> things
    reduceMin<float>(num_els, numThreads, numBlocks, whichKernel, d_in, d_out);

    cudaMemcpy(out, d_out, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

	//save data to output file
	std::ofstream fout("q1.txt", std::ios::app);
	float outMin = out[0];
    for(int i=0; i< numBlocks; i++) {
		if (i < numBlocks-1 && outMin > out[i+1]) outMin = out[i+1];
        //printf("\n Reduce MIN GPU value: %f",out[i]);
		if(fout.is_open()) {
			fout << "Reduce MIN GPU value:" << out[i] <<"\n";
		}
	}
	//printf("\n Final MIN GPU value: %f",outMin);
	if(fout.is_open()) {
		fout << "Final MIN GPU value:" << outMin <<"\n";
	}
	fout.close();
	fout.clear();
	


    float min;
    reduceMINCPU<float>(in, num_els, &min);

    //printf("\n\n Reduce MIN CPU value: %f", min);

    cudaFree(d_in);
    cudaFree(d_out);

    free(in);
    free(out);

    return delta;
}

void myLastDigitTest(const char *filename) {
    int maxThreads = 512; //must be power of 2
    //int maxBlocks = 20; 
	
	//read test file and decide size of array
	std::vector<int> data;
    std::string line_;
    std::ifstream file_(filename);
    if(file_.is_open()) {
        while (getline(file_, line_)) {
            std::stringstream ss(line_);
            int i;
            while(ss>>i) {
                data.push_back(i);
                if (ss.peek() == ',' || ss.peek() == ' ') {
                    ss.ignore();
                }
            }
        }
        file_.close();
    } 
    int num_els = data.size();
    int maxBlocks = num_els/maxThreads+1;	
	//d_in and d_out are in and out array on Device memory
    int* d_in = NULL;
    int* d_out = NULL;
	
	//printf("%d elements\n", num_els);
    //printf("%d threads (max)\n", maxThreads);
	
	//Allocate shared memory space on CUDA devices
    cudaMalloc((void **) &d_in, num_els * sizeof(int));
    cudaMalloc((void **) &d_out, num_els * sizeof(int));
	
	int* in = (int*) malloc(num_els * sizeof(int));
    int* out = (int*) malloc(num_els * sizeof(int));
	in = &data[0];
	
	// copy data directly to device memory
    cudaMemcpy(d_in, in, num_els * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, out, num_els * sizeof(int),cudaMemcpyHostToDevice);
	
	lastDigitWrapper(maxThreads, maxBlocks, d_in, d_out);
	
	//After kernel finishes, copy back
	cudaMemcpy(out, d_out, num_els * sizeof(int), cudaMemcpyDeviceToHost);
	
	std::ofstream fout1("q1.txt", std::ios::app);
	for(int i = 0; i < num_els; i++) {
		//printf("\n B[%d]: %d", i, out[i]);
		if(fout1.is_open()) {
			fout1 << "\n B[" <<i<<"]: " <<out[i];
		}
	}
	fout1.close();
	fout1.clear();
	
	cudaFree(d_in);
	cudaFree(d_out);
	
	free(in);
	free(out);
}

int main(int argc, char **argv) {
	my_min_test("inp.txt");
	myLastDigitTest("inp.txt");
	return 0;
}
