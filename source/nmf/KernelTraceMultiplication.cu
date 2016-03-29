/*
nmfgpu - CUDA accelerated computation of Non-negative Matrix Factorizations (NMF)

Copyright (C) 2015-2016  Sven Koitka (svenkoitka@fh-dortmund.de)

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

#include <common/Logging.h>
#include <common/MatrixDescription.h>
#include <nmf/KernelTraceMultiplication.h>
#include <nmf/KernelHelper.cuh>

namespace nmfgpu {
	namespace kernel {
		namespace impl {
			template<typename NumericType, bool transposeA>
			__global__ void traceMultiplication(unsigned nDiagElements, unsigned n, const NumericType* A, unsigned ldda, const NumericType* B, unsigned lddb, NumericType* partialSums) {
				//__shared__ NumericType sharedPartialSums[32];

				// Index des Diagonalelementes berechnen
				int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
				int indexDiag = threadIndex / 32;
				int laneIndex = threadIndex % 32;

				if (indexDiag >= nDiagElements)
					return;

				// Indices initialisieren, um nur mit einer Addition eines Offsets rechnen zu müssen
				NumericType sum = 0.f;
				int indexA, indexB;
				if (transposeA) {
					indexA = indexDiag * ldda + laneIndex;
				} else {
					indexA = indexDiag + laneIndex * ldda;
				}
				indexB = indexDiag * lddb + laneIndex;

				// Spatprodukt von Zeilenvektor A und Spaltenvektor B bilden
				for (int i = laneIndex; i < n; i += 32) {
					// Werte für Matrix A laden
					NumericType a1;
					if (transposeA) {
						a1 = A[indexA];
						indexA += 32;
					} else {
						a1 = CUDA_READ_ONLY_CACHE(A[indexA]);
						indexA += 32 * ldda;
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

				if (laneIndex == 0)
					partialSums[blockIdx.x * 16 + threadIdx.x / 32] = sum;

#if 0
				// Summe des Warps in den SharedMemory schreiben
				if (laneIndex == 0)
					sharedPartialSums[threadIdx.x / 32] = sum;

				__syncthreads();

				// Summe für den Threadblock berechnen
				if (threadIdx.x < 32) {
					const int warpCount = (blockDim.x + 31) / 32;
					sum = threadIdx.x < warpCount ? sharedPartialSums[threadIdx.x] : 0;

					sum += __shfl_xor(sum, 16);
					sum += __shfl_xor(sum, 8);
					sum += __shfl_xor(sum, 4);
					sum += __shfl_xor(sum, 2);
					sum += __shfl_xor(sum, 1);

					// Der erste Thread schreibt das Ergebnis des Blocks
					if (threadIdx.x == 0)
						partialSums[blockIdx.x] = sum;
				}
#endif
			}
		}

		size_t traceMultiplicationGetElementCount(size_t diagonalElements) {
			return (diagonalElements + 15) / 16 * 16;
		}

		void traceMultiplication(bool transposeLeft, const MatrixDescription<float>& left, const MatrixDescription<float>& right, DeviceMemory<float>& partialSums, cudaStream_t stream /* = nullptr */) {
			// Check dimensions
			/*if (left.dimension() != right.dimension()) {
				return;
			}*/

			// Compute grid dimensions
			dim3 blockDim(512);
			dim3 gridDim((right.columns + 15) / 16);

			// Check if partialSums array is of sufficient size
			if (partialSums.elements() < gridDim.x * 16) {
				std::cerr << __FILE__ << "(" << __LINE__ << "): Parameter 'partialSums' does not have sufficient memory!" << std::endl;
				return;
			}/* else if (gridDim.x > psLength) // ???
				return;*/

			// Launch kernel
			if (transposeLeft)
				impl::traceMultiplication<float, true><<<gridDim, blockDim, 0, stream>>>(right.columns, right.rows, left.dense.values, left.dense.leadingDimension, right.dense.values, right.dense.leadingDimension, partialSums.get());
			else
				impl::traceMultiplication<float, false><<<gridDim, blockDim, 0, stream>>>(right.columns, right.rows, left.dense.values, left.dense.leadingDimension, right.dense.values, right.dense.leadingDimension, partialSums.get());
		}

		void traceMultiplication(bool transposeLeft, const MatrixDescription<double>& left, const MatrixDescription<double>& right, DeviceMemory<double>& partialSums, cudaStream_t stream /* = nullptr */) {
			// Check dimensions
			/*if (left.dimension() != right.dimension()) {
				return;
			}*/

			// Compute grid dimensions
			dim3 blockDim(512);
			dim3 gridDim((right.columns + 15) / 16);

			// Check if partialSums array is of sufficient size
			if (partialSums.elements() < gridDim.x * 16) {
				std::cerr << __FILE__ << "(" << __LINE__ << "): Parameter 'partialSums' does not have sufficient memory!" << std::endl;
				return;
			}/* else if (gridDim.x > psLength) // ???
			 return;*/

			// Launch kernel
			if (transposeLeft)
				impl::traceMultiplication<double, true><<<gridDim, blockDim, 0, stream>>>(right.columns, right.rows, left.dense.values, left.dense.leadingDimension, right.dense.values, right.dense.leadingDimension, partialSums.get());
			else
				impl::traceMultiplication<double, false><<<gridDim, blockDim, 0, stream>>>(right.columns, right.rows, left.dense.values, left.dense.leadingDimension, right.dense.values, right.dense.leadingDimension, partialSums.get());
		}
	}
}
