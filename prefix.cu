#include <math.h>
#include <cuda.h>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <iostream>

#define BLOCK_WIDTH 512
/*
extern __shared__ stands for shared memory on device, which has two "warps" of 32 threads.
Google CUDA shared memory and warps.
To replace extern __shared__ int __smem[]; which requires you to explicitly
know the data type is integer in advance. But input file could be int, float, or double.
Since we don't know the data type of shared meomry __smem[], we use
template<class T> where T stands for all possible data types. We also 
need to instantiate all possible data types later
In return (T *) __smem; it is data type conversion
Suggest to figure out difference between overload, override, redefine
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

/////////////////////////////////////////////////////////////////////////////
//                   CUDA Kernel: Global memory
/////////////////////////////////////////////////////////////////////////////
template<class T, int blockSize>
__global__ void countGlobalMem(T *g_idata, int *g_odata, int N) {
	unsigned int i = blockSize * blockIdx.x + threadIdx.x;
	int gi = 0;
	if (i < N) {
		if (g_idata[i] == 1000) {
			atomicAdd(&g_odata[9], 1);
		} else {
			gi = (int) g_idata[i] / 100;
			atomicAdd(&g_odata[gi], 1);
		}
	}
}

/////////////////////////////////////////////////////////////////////////////
//                   CUDA Kernel: shared memory
/////////////////////////////////////////////////////////////////////////////
template<class T, int blockSize>
__global__ void countSharedMem(T *g_idata, int *g_odata, int N, int maxNum, int barrelSize) {
	/*
	Each block has a sdata
	*/
	extern __shared__ int sdata[];
	unsigned int tid = threadIdx.x;
	int numBarrel = maxNum/barrelSize;
	
	unsigned int i = blockSize * blockIdx.x + threadIdx.x;
	//gi is group/barrel index
	int gi = 0;
	if (i < N) {
		if (g_idata[i] == maxNum) {
			atomicAdd(&sdata[numBarrel-1], 1);
		} else {
			gi = (int) g_idata[i] / barrelSize;
			atomicAdd(&sdata[gi], 1);
		}
	}
	
	//wait until sdata[0~9] in all blocks are ready
	__syncthreads();
	/*
	every block has threadIdx.x from 0 to 511
	size of g_odata is numBarrel * blocks
	sum of all blocks is done in myCountTest(), note there
	is += when output to "q2b.txt"
	*/
	if (tid < numBarrel) {
		g_odata[blockIdx.x * numBarrel + tid] = sdata[tid];
	}	
}

//////////////////////////////////////////////////////////////////////////////
//                     CUDA Kernel: prefix sum (Naiive)
///////////////////////////////////////////////////////////////////////////////
int nextPowerOf2(int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

__global__ void scan(int *d_idata, int *d_odata, int N) {
	extern __shared__ int sdata[];
	
	//cunyi
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		sdata[threadIdx.x] = d_idata[i];
		//printf("\n sdata[%d]: %d", i, sdata[threadIdx.x]);
	}
	
	for (unsigned int stride = 1; stride <= threadIdx.x; stride *= 2) {
		__syncthreads();
		int in1 = sdata[threadIdx.x - stride];
		__syncthreads();
		sdata[threadIdx.x] += in1;
	}
	__syncthreads();
	if(i < N) {
		d_odata[threadIdx.x] = sdata[threadIdx.x];
		//printf("\n sdata[%d]: %d", i, d_odata[threadIdx.x]);
	}
	
}

///////////////////////////////////////////////////////////////////////////////
//                   Wrapper for countGlobalMem
///////////////////////////////////////////////////////////////////////////////
template<class T>
void countGMWrapper(int threads, int blocks, T *g_idata, int *g_odata ,int N) {
	/*
	1D block and 1D grid
	*/
	dim3 dimBlock(threads, 1, 1); 
	dim3 dimGrid(blocks, 1, 1);
	int smemSize = threads * sizeof (T);
	
	countGlobalMem<T, BLOCK_WIDTH><<<dimGrid, dimBlock, smemSize>>>(g_idata, g_odata, N);
}

///////////////////////////////////////////////////////////////////////////////
//                   Wrapper for countSharedMem
///////////////////////////////////////////////////////////////////////////////
template<class T>
void countSWrapper(int threads, int blocks, T *g_idata, int *g_odata ,int N, int maxNum, int barrelSize) {
	/*
	1D block and 1D grid
	*/
	dim3 dimBlock(threads, 1, 1); 
	dim3 dimGrid(blocks, 1, 1);
	int smemSize = threads * sizeof (T);
	
	countSharedMem<T, BLOCK_WIDTH><<<dimGrid, dimBlock, smemSize>>>(g_idata, g_odata, N, maxNum, barrelSize);
}

/////////////////////////////////////////////////////////////////////////////////
//                       Instantiate Template
/////////////////////////////////////////////////////////////////////////////////
template void
countGMWrapper<int>(int threads, int blocks, int *g_idata, int *g_odata, int N);

template void
countGMWrapper<float>(int threads, int blocks, float *g_idata, int *g_odata, int N);

template void
countGMWrapper<double>(int threads, int blocks, double *g_idata, int *g_odata, int N);

template void
countSWrapper<int>(int threads, int blocks, int *g_idata, int *g_odata ,int N, int maxNum, int barrelSize);

//////////////////////////////////////////////////////////////////////////////////
//						Test Function
//////////////////////////////////////////////////////////////////////////////////
void myCountTest(const char* filename) {
	int numBarrel = 10;
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
	int numBlocks = num_els/BLOCK_WIDTH + 1;
	
	//Start to run Kernel_a
	int *d_in = NULL;
	int *d_out = NULL;
	cudaMalloc( (void **) &d_in, num_els * sizeof(int));
	cudaMalloc( (void **) &d_out, numBarrel * sizeof(int));
	int *in = (int *) malloc(num_els * sizeof(int));
	int *out = (int *) malloc(numBarrel * sizeof(int));	
	in = &data[0];
	std::vector<int> v(10);
	std::fill(v.begin(), v.end(), 0);
	std::copy(v.begin(), v.end(), out);
	cudaMemcpy(d_in, in, num_els * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_out, out, numBarrel * sizeof(int), cudaMemcpyHostToDevice);
	countGMWrapper(BLOCK_WIDTH, numBlocks, d_in, d_out, num_els);
	cudaMemcpy(out, d_out, numBarrel * sizeof(int), cudaMemcpyDeviceToHost);
	std::ofstream fout1("q2a.txt", std::ios::app);
	for(int i = 0; i < numBarrel; i++) {
		if(fout1.is_open()) {
			fout1 << "\n Count[" <<i<<"]: " <<out[i];
		}
	}
	fout1.close();
	fout1.clear();
	cudaFree(d_out);
	//free(out);
	//d_in is not cleaned because we are going to run more cuda kernels using d_in
	free(in);
	//cudaFree(d_in);
	
	//Start to run Kernel_b, almost the same as kernel_a
	int *d_out_b = NULL;
	cudaMalloc( (void **) &d_out_b, numBarrel * numBlocks * sizeof(int));
	int *out_b = (int *) malloc(numBarrel * numBlocks * sizeof(int));
	//size of out_b is changed
	v.resize(numBarrel * numBlocks);
	std::fill(v.begin(), v.end(), 0);
	std::copy(v.begin(), v.end(), out_b);
	cudaMemcpy(d_out_b, out_b, numBarrel * numBlocks  * sizeof(int), cudaMemcpyHostToDevice);
	countSWrapper(BLOCK_WIDTH, numBlocks, d_in, d_out ,num_els, 1000, 100);
	cudaMemcpy(out_b, d_out_b, numBarrel * numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
	std::ofstream fout2("q2b.txt", std::ios::app);
	int out_b_all;
	//int B[numBarrel];
	for(int i = 0; i < numBarrel; i++) {
		out_b_all = 0;
		for (int j = 0; j < numBlocks; j++)
			out_b_all += out_b[i + j * numBarrel];
		//B[i] = out_b_all;
		if(fout2.is_open()) {
			fout2 << "\n Count[" <<i<<"]: " <<out_b_all;
		}
	}
	fout2.close();
	fout2.clear();
	cudaFree(d_out_b);
	free(out_b);
	cudaFree(d_in);
	
	//start to run Kernel_c
	int n3 = nextPowerOf2(numBarrel);
	int *d_out_c = NULL;
	int *d_in_c = NULL;
	int *out_c = (int *) malloc(n3 * sizeof(int));
	v.resize(n3);
	std::fill(v.begin(), v.end(), 0);
	std::copy(v.begin(), v.end(), out_c);
	cudaMalloc( (void **) &d_in_c, n3 * sizeof(int));
	cudaMalloc( (void **) &d_out_c, n3 * sizeof(int));
	
	int *in_test = (int *) malloc(n3 * sizeof(int));
	std::vector<int> in_c;
	for (int i = 0; i < n3; i++) {
		if (i < numBarrel) {
			in_c.push_back(out[i]);
		} else {
			in_c.push_back(0);
		}
		//printf("\n c: %d", in_c[i]);
	}
	in_test = &in_c[0];
	cudaMemcpy(d_in_c, in_test, n3 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_out_c, out_c, n3 * sizeof(int), cudaMemcpyHostToDevice);
	scan<<<1, n3, n3*sizeof(int)>>>(d_in_c, d_out_c, n3);
	cudaMemcpy(out_c, d_out_c, n3 * sizeof(int), cudaMemcpyDeviceToHost);
	std::ofstream fout3("q2c.txt", std::ios::app);
	for(int i = 0; i < numBarrel; i++) {
		if(fout3.is_open()) {
			fout3 << "\n prescan[" <<i<<"]: " <<out_c[i];
		}
	}
	fout3.close();
	fout3.clear();
}

int main(int argc, char **argv) {
	myCountTest("inp.txt");
	return 0;
}
