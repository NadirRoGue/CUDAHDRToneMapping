#include <device_functions.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

#define _BLOCK_SIZE 64


#define checkCudaErrorsC(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
	if (err != cudaSuccess) {
		std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
		std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
		exit(1);
	}
}

// =========================================================================================================
// =========================================================================================================

/*
* Extracts the highest value and the lowest value from the max_input and min_input. This kernel is prepared to be
* executed iteratively, as it might process large amount of data that cannot be stored in a single block's shared memory
* Each iteration will use the output of the previous one as input
* @param min_input elements from which to find the lowest value
* @param max_input elements from which to find the highest value
* @param min_output output array of size len / blockSize which will contain the lowest elements from min_input
* @param max_output output array of size len / blockSize which will contain the highest elements from max_output
* @param len number of elments to process from both inputs, starting from the begining (allows iterative processing without resizing the input/output)
*/
__global__ void findMinMax(const float * min_input, const float * max_input, float * min_output, float * max_output, unsigned int len) 
{
	unsigned int ti = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + ti;
	__shared__ float min_cache[_BLOCK_SIZE * 2];
	__shared__ float max_cache[_BLOCK_SIZE * 2];

	// Fill the cache of out of bounds elements with values that will be discarded
	if (i >= len)
	{
		min_cache[ti] = 99999.f;
		min_cache[ti + blockDim.x] = 99999.f;
		max_cache[ti] = -99999.f;
		max_cache[ti + blockDim.x] = -99999.f;
	}
	// Fill in bound ids with real iput values
	else
	{
		min_cache[ti] = min_input[i];
		max_cache[ti] = max_input[i];

		// If possible, load the next _BLOCK_SIZE elements into shared cache 
		// to be able to perform the process using the whole block
		unsigned int secondIndex = blockIdx.x * (blockDim.x * 2) + ti;
		if (secondIndex < len)
		{
			min_cache[ti + blockDim.x] = min_input[secondIndex];
			max_cache[ti + blockDim.x] = max_input[secondIndex];
		}
		else
		{
			min_cache[ti + blockDim.x] = 99999.f;
			max_cache[ti + blockDim.x] = -99999.f;
		}
	}

	// Allow the whole block to have written their values
	__syncthreads();

	// Recursively divide the block piece of input by 2 until everything is reduced to the first element
	for (unsigned int s = _BLOCK_SIZE; s > 0; s /= 2)
	{
		// min
		float current = min_cache[ti];
		float test = min_cache[ti + s];
		min_cache[ti] = current < test ? current : test;

		// max
		current = max_cache[ti];
		test = max_cache[ti + s];
		max_cache[ti] = current > test ? current : test;
		
		__syncthreads();
	}

	// Thread 0 will use the blockid to write the output in the appropiate position
	if (ti == 0)
	{
		min_output[blockIdx.x] = min_cache[0];
		max_output[blockIdx.x] = max_cache[0];
	}
}

// =========================================================================================================
// =========================================================================================================

/*
* Builds an instogram by analyzing the input lighting. Transform each value to the instogram bin to which it belongs
* based on the range from the min value to the max value. Extracted from class slides
* @param buffer input lighting channel
* @param size input lighting channel number of elements
* @param histo output array where the histogram will be stored
* @param minLum minimun value of lighting in the input lighting array
* @param lumRange difference between the higeset value and the lowest value in the input lighting array
* @param numBins number of bins of which the histogram will be composed
*/
__global__ void histo(const float *buffer, size_t size, unsigned int *histo, float minLum, float lumRange, size_t numBins)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	// All threads handle blockDim.x * gridDim.x  consecutive elements
	if (i < size) 
	{
		// Compute the bin to which the being-processed lighting value belongs
		unsigned int bin = ((buffer[i] - minLum) / lumRange) * numBins;
		atomicAdd(&(histo[bin]), 1);  //Varios threads podrían intentar incrementar el mismo valor a  la vez
	}
}

// =========================================================================================================
// =========================================================================================================

/*
* Performs exclusive scan over a set of numbers. Extracted from class slides. Is designed to work with a single
* block thread (it requires to store all the input data into shared memory for later use, so it will not work by
* splitting the kernel execution into different blocks which wont share the cache)
* @param numberArray elements in which to perform the scan & reduction
* @param texSize number of elements in the input array
*/
__global__ void exclusive_scan(unsigned int * numberArray, unsigned int texSize)
{
	// Shared memory size is passed throught kernel template call, based on the number
	// of thread which will perform the exclusive scan
	__shared__  unsigned int tempArray[1024];
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	int threadId = threadIdx.x;
	int offset = 1;
	unsigned int temp;
	int ai = threadId;
	int bi = threadId + texSize / 2;
	int i;
	//assign the shared memory
	tempArray[ai] = numberArray[id];
	tempArray[bi] = numberArray[id + texSize / 2];
	//up tree
	for (i = texSize >> 1; i > 0; i >>= 1)
	{
		__syncthreads();
		if (threadId < i)
		{
			ai = offset * (2 * threadId + 1) - 1;
			bi = offset * (2 * threadId + 2) - 1;
			tempArray[bi] += tempArray[ai];
		}
		offset <<= 1;
	}
	//put the last one 0
	if (threadId == 0)
		tempArray[texSize - 1] = 0;
	//down tree
	for (i = 1; i < texSize; i <<= 1) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (threadId < i)
		{
			ai = offset * (2 * threadId + 1) - 1;
			bi = offset * (2 * threadId + 2) - 1;
			temp = tempArray[ai];
			tempArray[ai] = tempArray[bi];
			tempArray[bi] += temp;
		}
	}
	__syncthreads();
	numberArray[id] = tempArray[threadId];
	numberArray[id + texSize / 2] = tempArray[threadId + texSize / 2];
}

// =========================================================================================================
// =========================================================================================================

/*
 * Returns the needed amount of grids to have 1 thread per element, given the defined block size
 * @param lenToProcess number of elements to process
 * @param blockSizeX number of threads per block which will be used
 * @returns dim3 size of the grid to use
 */
inline dim3 getGridSize(unsigned int lenToProcess, unsigned int blockSizeX)
{
	unsigned int init = lenToProcess / blockSizeX;
	// Make sure to launch more threads than elements, not otherwise
	if (lenToProcess % blockSizeX > 0)
		init++;

	init = init == 0 ? 1 : init;

	return dim3(init, 1, 1);
}

// =========================================================================================================
// =========================================================================================================

void calculate_cdf(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
	// TODO
	// 1) Encontrar el valor máximo y mínimo de luminancia en min_logLum and max_logLum a partir del canal logLuminance 
	// -------------------------------------------------------------------------------------------------------------------
	const unsigned int len = unsigned int(numRows) * unsigned int(numCols);
	
	const dim3 blockSize(_BLOCK_SIZE, 1, 1);
	
	
	float * d_max_output, * d_min_output, * d_min_input, * d_max_input;

	// Initialize buffers over which we will perform the reduction iteratively
	checkCudaErrorsC(cudaMalloc(&d_min_input, len * sizeof(float)));
	checkCudaErrorsC(cudaMemcpy(d_min_input, d_logLuminance, len * sizeof(float), cudaMemcpyDeviceToDevice));

	checkCudaErrorsC(cudaMalloc(&d_max_input, len * sizeof(float)));
	checkCudaErrorsC(cudaMemcpy(d_max_input, d_logLuminance, len * sizeof(float), cudaMemcpyDeviceToDevice));
	
	// Iterator which delimites the range of the data we are processing
	// is updated according to the reduction performed in the kernel
	unsigned int iterator = len;
	// Compute grid size based on half of the the elements, since we are using 1 thread to process 2 elements on initialisation in the kernel
	dim3 gridSize = getGridSize(len / 2, _BLOCK_SIZE);

	// We initialize the output buffers the the size required for the first iteration
	checkCudaErrorsC(cudaMalloc(&d_max_output, (len / _BLOCK_SIZE) * sizeof(float)));
	checkCudaErrorsC(cudaMalloc(&d_min_output, (len / _BLOCK_SIZE) * sizeof(float)));

	do
	{
		// Reduce the current input
		findMinMax << <gridSize, blockSize >> > (d_min_input, d_max_input, d_min_output, d_max_output, iterator);
		cudaDeviceSynchronize(); checkCudaErrorsC(cudaGetLastError());

		// Reduce the iterator according to the reduction performed
		iterator /= _BLOCK_SIZE;
		gridSize = getGridSize(iterator / 2, _BLOCK_SIZE);
		
		// Update next kernel call input with previous kernel call output
		if (iterator > 0)
		{
			// Copy only the necessary data
			checkCudaErrorsC(cudaMemcpy(d_min_input, d_min_output, iterator * sizeof(float), cudaMemcpyDeviceToDevice));
			checkCudaErrorsC(cudaMemcpy(d_max_input, d_max_output, iterator * sizeof(float), cudaMemcpyDeviceToDevice));
		}
	} while (iterator > 0);

	checkCudaErrorsC(cudaMemcpy(&min_logLum, d_min_output, sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrorsC(cudaMemcpy(&max_logLum, d_max_output, sizeof(float), cudaMemcpyDeviceToHost));

	checkCudaErrorsC(cudaFree(d_min_input));
	checkCudaErrorsC(cudaFree(d_max_input));
	checkCudaErrorsC(cudaFree(d_min_output));
	checkCudaErrorsC(cudaFree(d_max_output));
	// -------------------------------------------------------------------------------------------------------------------


	// 2) Obtener el rango a representar
	// -------------------------------------------------------------------------------------------------------------------
	const float range = max_logLum - min_logLum;
	// -------------------------------------------------------------------------------------------------------------------


	// 3) Generar un histograma de todos los valores del canal logLuminance usando la formula: bin = (Lum [i] - lumMin) / lumRange * numBins
	// -------------------------------------------------------------------------------------------------------------------
	
	// Use the same array where the output accumulate distribution will be computed, saving allocating and trasfering data
	// from an auxiliar buffer
	gridSize = getGridSize(len, _BLOCK_SIZE);
	histo << <gridSize, blockSize >> > (d_logLuminance, len, d_cdf, min_logLum, range, numBins);
	cudaDeviceSynchronize(); checkCudaErrorsC(cudaGetLastError());

	unsigned int h_histo[1024];
	memset(h_histo, 0, 1024 * sizeof(unsigned int));
	checkCudaErrorsC(cudaMemcpy(h_histo, d_cdf, 1024 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	unsigned int nonZeroes = 0;
	for (unsigned int i = 0; i < 1024; i++)
	{
		if (h_histo[i] != 0)
			nonZeroes++;
	}
	std::cout << nonZeroes << std::endl;
	// -------------------------------------------------------------------------------------------------------------------


	// 4) Realizar un exclusive scan en el histograma para obtener la distribución acumulada (cdf)
	// de los valores de luminancia. Se debe almacenar en el puntero c_cdf
	// -------------------------------------------------------------------------------------------------------------------
	
	gridSize = dim3(1, 1, 1);
	dim3 bs(unsigned int(numBins) / 2, 1, 1);
	bs.x = bs.x > 512u ? 512u : bs.x; // max threads per block is 512
	exclusive_scan << <gridSize, bs >> > (d_cdf, unsigned int(numBins));
	cudaDeviceSynchronize(); checkCudaErrorsC(cudaGetLastError());

	// -------------------------------------------------------------------------------------------------------------------
}
