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

#include <common/Matrix.h>
#include <init/EInNMF.h>
#include <nmf/KernelHelper.cuh>

namespace nmfgpu {
	namespace kernels {
		namespace details {
			template<typename NumericType>
			__device__ inline NumericType distanceSq(unsigned m, unsigned idxA, unsigned idxB, const NumericType* A, unsigned lda, const NumericType* B, unsigned ldb) {
				NumericType sum = 0.0;

				for (auto i = threadIdx.x; i < m; i += 32u) {
					auto diff = A[idxA * lda + i] - B[idxB * ldb + i];
					sum += diff * diff;
				}

				return sumWarpReduction(sum);
			}

			// Left: Clusters
			// Right: Data
			template<typename NumericType>
			__global__ void kernelComputeDistanceMatrix(unsigned m, unsigned nW, unsigned nV, const NumericType* W, unsigned ldw, const NumericType* V, unsigned ldv, NumericType* distanceMatrix, unsigned ldDistanceMatrix) {
				// Get ID of corresponding data entry for this warp
				auto indexV = blockDim.y * blockIdx.y + threadIdx.y;
				if (indexV >= nV) {
					return;
				}

				// Find cluster with the minimum distance
				for (auto indexW = 0u; indexW < nW; ++indexW) {
					auto distance = distanceSq(m, indexW, indexV, W, ldw, V, ldv);
					distanceMatrix[indexV * ldDistanceMatrix + indexW] = distance;
				}
			}

			template<typename NumericType>
			__global__ void kernel_EInNMF_prefix_scan_kepler(unsigned m, unsigned n, NumericType* H, unsigned ldh) {
				auto column = blockDim.y * blockIdx.y + threadIdx.y;
				auto row = threadIdx.x;
				auto previousSum = NumericType(0.0);
				
				if (column >= n) {
					return;
				}

				while (row < m) {
					// Fetch value
					auto valueSaved = H[column * ldh + row];
					auto value = 1.f / (valueSaved + 1.e-9);

					// Prefix sum using shuffle intrinsics
					#pragma unroll
					for (auto i = 1u; i < 32u; i *= 2u) {
						auto n = __shfl_up(value, i);
						if (threadIdx.x >= i) {
							value += n;
						}
					}

					value += previousSum;

					// Write final result
					H[column * ldh + row] = 1.f / (valueSaved * value + 1.e-9);

					// Last element of warp will be base sum for next iteration
					previousSum = __shfl(value, 31);
					row += 32;
				}
			}

			template<typename NumericType>
			void computeDistanceMatrix(const DeviceMatrix<NumericType>& matrixW, const DeviceMatrix<NumericType>& matrixV, const DeviceMatrix<NumericType>& matrixDistances) {
				auto blockDim = dim3(32, 8);
				auto gridDim = dim3(1, (matrixV.columns() + blockDim.y - 1) / blockDim.y);
				kernelComputeDistanceMatrix<NumericType><<<gridDim, blockDim>>>(matrixW.rows(), matrixW.columns(), matrixV.columns(), matrixW.get(), matrixW.leadingDimension(),
																				matrixV.get(), matrixV.leadingDimension(), matrixDistances.get(), matrixDistances.leadingDimension());
				CUDA_CALL(cudaGetLastError());
			}

			template<typename NumericType>
			void computePrefixScan(const DeviceMatrix<NumericType>& matrixH) {
				auto blockDim = dim3(32, 8);
				auto gridDim = dim3(1, (matrixH.columns() + blockDim.y - 1) / blockDim.y);
				kernel_EInNMF_prefix_scan_kepler<NumericType><<<gridDim, blockDim>>>(matrixH.rows(), matrixH.columns(), matrixH.get(), matrixH.leadingDimension());
				CUDA_CALL(cudaGetLastError());
			}
		}

		void computeDistanceMatrix(const DeviceMatrix<float>& matrixW, const DeviceMatrix<float>& matrixV, const DeviceMatrix<float>& matrixDistances) {
			details::computeDistanceMatrix(matrixW, matrixV, matrixDistances);
			details::computePrefixScan(matrixDistances);
		}

		void computeDistanceMatrix(const DeviceMatrix<double>& matrixW, const DeviceMatrix<double>& matrixV, const DeviceMatrix<double>& matrixDistances) {
			details::computeDistanceMatrix(matrixW, matrixV, matrixDistances);
			details::computePrefixScan(matrixDistances);
		}
	}
}