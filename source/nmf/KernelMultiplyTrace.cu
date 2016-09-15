/*
nmfgpu - CUDA accelerated computation of Non-negative Matrix Factorizations (NMF)

Copyright (C) 2015-2016  Sven Koitka (sven.koitka@fh-dortmund.de)

This file is part of nmfgpu.

nmfgpu is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

nmfgpu is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with nmfgpu.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "KernelMultiplyTrace.h"
#include "KernelHelper.cuh"

namespace NMF {
	namespace Kernel {
		template<typename NumericType, bool transposeA>
		__global__ void kernelMulTrace(unsigned nDiagElements, unsigned n, const NumericType* A, unsigned ldda, const NumericType* B, unsigned lddb, NumericType* partialSums) {
			//__shared__ NumericType sharedPartialSums[32];

			// Index des Diagonalelementes berechnen
			int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
			int indexDiag = threadIndex / 32;
			int laneIndex = threadIndex % 32;

			if(indexDiag >= nDiagElements)
				return;

			// Indices initialisieren, um nur mit einer Addition eines Offsets rechnen zu müssen
			NumericType sum = 0.f;
			int indexA, indexB;
			if(transposeA) {
				indexA = indexDiag * ldda + laneIndex;
			} else {
				indexA = indexDiag + laneIndex * ldda;
			}
			indexB = indexDiag * lddb + laneIndex;

			// Spatprodukt von Zeilenvektor A und Spaltenvektor B bilden
			for(int i = laneIndex; i < n; i+=32) {
				// Werte für Matrix A laden
				NumericType a1;
				if(transposeA) {
					a1 = A[indexA];
					indexA+=32;
				} else {
					a1 = CUDA_READ_ONLY_CACHE(A[indexA]);
					indexA += 32*ldda;
				}

				// Werte für Matrix B laden
				NumericType b1 = B[indexB];
				indexB += 32;

				// Werte verarbeiten
				sum += a1 * b1;
			}
			
			// Summe für den Warp bilden und somit Diagonalelement bilden
			sum += __shfl_xor(sum, 16);
			sum += __shfl_xor(sum, 8);
			sum += __shfl_xor(sum, 4);
			sum += __shfl_xor(sum, 2);
			sum += __shfl_xor(sum, 1);
			
			if(laneIndex == 0)
				partialSums[blockIdx.x * 16 + threadIdx.x / 32] = sum;

#if 0
			// Summe des Warps in den SharedMemory schreiben
			if(laneIndex == 0)
				sharedPartialSums[threadIdx.x / 32] = sum;

			__syncthreads();

			// Summe für den Threadblock berechnen
			if(threadIdx.x < 32) {
				const int warpCount = (blockDim.x + 31) / 32;
				sum = threadIdx.x < warpCount ? sharedPartialSums[threadIdx.x] : 0;

				sum += __shfl_xor(sum, 16);
				sum += __shfl_xor(sum, 8);
				sum += __shfl_xor(sum, 4);
				sum += __shfl_xor(sum, 2);
				sum += __shfl_xor(sum, 1);

				// Der erste Thread schreibt das Ergebnis des Blocks
				if(threadIdx.x == 0)
					partialSums[blockIdx.x] = sum;
			}
#endif
		}

		void multiplyTrace(bool transposeA, unsigned m, unsigned n, unsigned r, const float* A, unsigned ldda, const float* B, unsigned lddb, float* partialSums, unsigned& psLength) {
			// Check dimensions
			if(m != n)
				return;

			// Compute grid dimensions
			dim3 blockDim(512);
			dim3 gridDim((m + 15) / 16);
	
			// Check if partialSums array is sufficient large	
			if(partialSums == 0) {
				psLength = gridDim.x * 16;
				return;
			} else if(gridDim.x > psLength)
				return;
		
			// Launch kernel
			if(transposeA)
				kernelMulTrace<float, true><<<gridDim, blockDim>>>(n, r, A, ldda, B, lddb, partialSums);	
			else
				kernelMulTrace<float, false><<<gridDim, blockDim>>>(n, r, A, ldda, B, lddb, partialSums);		
		}

		void multiplyTrace(bool transposeA, unsigned m, unsigned n, unsigned r, const double* A, unsigned ldda, const double* B, unsigned lddb, double* partialSums, unsigned& psLength) {
			// Check dimensions
			if(m != n)
				return;

			// Compute grid dimensions
			dim3 blockDim(512);
			dim3 gridDim((m + 15) / 16);
	
			// Check if partialSums array is sufficient large	
			if(partialSums == 0) {
				psLength = gridDim.x * 16;
				return;
			} else if(gridDim.x > psLength)
				return;
		
			// Launch kernel
			if(transposeA)
				kernelMulTrace<double, true><<<gridDim, blockDim>>>(n, r, A, ldda, B, lddb, partialSums);	
			else
				kernelMulTrace<double, false><<<gridDim, blockDim>>>(n, r, A, ldda, B, lddb, partialSums);	
		}


		#include <stdio.h>

		namespace Details {
			template<typename NumericType>
			NumericType nmf_resolve_mul_trace(const NumericType* partialSums, unsigned psLength) {
				NumericType* hostBuffer = new NumericType[psLength];
				NumericType sum = 0.f;
				
				cudaMemcpy(hostBuffer, partialSums, sizeof(NumericType) * psLength, cudaMemcpyDeviceToHost);
				//printf("%d\n", psLength);
				for(unsigned i = 0; i < psLength; ++i) {
					//printf("%f ", hostBuffer[i]);
					sum += hostBuffer[i];
				}
				delete[] hostBuffer;
	
				return sum;
			}
		}

		float resolveMultiplyTrace(const float* partialSums, unsigned psLength) {
			return Details::nmf_resolve_mul_trace<float>(partialSums, psLength);
		}

		double resolveMultiplyTrace(const double* partialSums, unsigned psLength) {
			return Details::nmf_resolve_mul_trace<double>(partialSums, psLength);
		}
	}
}
