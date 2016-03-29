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
#include <nmf/KernelNormalizeColumns.h>

namespace nmfgpu {
	namespace kernel {
		namespace impl {
			template<typename NumericType>
			__global__ void normalizeColumns(unsigned m, unsigned n, NumericType* A, unsigned ldda) {
				unsigned column = blockDim.y * blockIdx.y + threadIdx.y;

				if (column >= n) {
					return;
				}

				// Compute sum of column
				NumericType sum = 0;

				for (unsigned i = threadIdx.x; i < m; i += blockDim.x) {
					sum += A[column * ldda + i] * A[column * ldda + i];
				}

				// Using warp reduction to get column sum
				sum += __shfl_xor(sum, 16);
				sum += __shfl_xor(sum, 8);
				sum += __shfl_xor(sum, 4);
				sum += __shfl_xor(sum, 2);
				sum += __shfl_xor(sum, 1);

				// Normalize column
				if (sum > 0) {
					sum = sqrt(sum);
					for (unsigned i = threadIdx.x; i < m; i += blockDim.x) {
						NumericType value = A[column * ldda + i];
						A[column * ldda + i] = value / sum;
					}
				}
			}
		}

		void normalizeColumns(const MatrixDescription<float>& target, cudaStream_t stream /* = nullptr */) {
			if (target.format != StorageFormat::Dense) {
				Logging::instance().error()
					.print("[ERROR] " NMFGPU_FILE_LINE_PREFIX ": Input matrix must be stored in dense format!").lineFeed();
				return;
			}

			auto m = static_cast<unsigned>(target.rows);
			auto n = static_cast<unsigned>(target.columns);
			dim3 blockDim(32, 8);
			dim3 gridDim(1, (n + blockDim.y - 1) / blockDim.y);
			impl::normalizeColumns<float><<<gridDim, blockDim, 0, stream>>>(m, n, target.dense.values, static_cast<unsigned>(target.dense.leadingDimension));
		}

		void normalizeColumns(const MatrixDescription<double>& target, cudaStream_t stream /* = nullptr */) {
			if (target.format != StorageFormat::Dense) {
				Logging::instance().error()
					.print("[ERROR] " NMFGPU_FILE_LINE_PREFIX ": Input matrix must be stored in dense format!").lineFeed();
				return;
			}

			auto m = static_cast<unsigned>(target.rows);
			auto n = static_cast<unsigned>(target.columns);
			dim3 blockDim(32, 8);
			dim3 gridDim(1, (n + blockDim.y - 1) / blockDim.y);
			impl::normalizeColumns<double><<<gridDim, blockDim, 0, stream>>>(m, n, target.dense.values, static_cast<unsigned>(target.dense.leadingDimension));
		}
	}
}
