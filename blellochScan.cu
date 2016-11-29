//This file contains Blelloch scan or work-efficiency prefix sum
#include<iostream>
#include<math.h>
#include<cuda.h>

//P8 Hillis-Steele Scan (or naiive prefix sum)
//P12 Blelloch Scan (or work-efficient prefix)
//inputarray == outputarray, this is in-place prefix-sum
//loop = (int)log2f(number); number is length of array
__global__ void blellochscan(int *inputarray, int loop, int *outputarray, int number) {
	unsigned int Idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	int divisor = 2;
	int adder = 1;
	int temp;
	
	for (int i = 0; i < loop; i++) {
		/*
		1st step: index of 0,2,4,6,8,... add to 1,3,5,7,9,... (they mod 2 == 1 i.e. 2-1, divisor==2; adder=1-0,3-2,5-4,==1)
		2nd step: index of 1,5,9,... add to 3,7,11,... (they mod 4 == 3 i.e. 4-1, divisor == 2*old_divisor==2*2=4; adder=3-1,7-5,11-9==2)  
		*/
		if (Idx % divisor == divisor - 1) {
			outputarray[Idx] = outputarray[Idx] + outputarray[Idx-adder];
		}
		__synthreads();
		divisor *= 2;
		adder *= 2;
	}
	
	//number is the length of array (number of elements in the array)
	//Take length of 8 as an example
	divisor = number;
	adder = divisor/2;
	
	outputarray[number-1] = 0;
	for (int i = 0; i < loop; i++) {
		if(Idx % divisor == divisor -1) {
			temp = outputarray[Idx];
			//In the first loop, outputarray[7] = outputarray[7] + outputarray[3];
			outputarray[Idx] = outputarray[Idx] + outputarray[Idx-adder];
			outputarray[Idx-adder] = temp;
		}
		__syncthreads();
		divisor /= 2;
		adder /= 2;
	}
}
