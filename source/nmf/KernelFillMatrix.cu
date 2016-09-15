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

#include <common/Logging.h>
#include <common/MatrixDescription.h>
#include <nmf/KernelMultiplyDivide.h>

namespace nmfgpu {
	namespace kernel {
		namespace impl {
			template<typename NumericType, bool ReuseValue>
			__global__ void fillMatrix(unsigned m, unsigned n, NumericType* A, unsigned lda, NumericType diagValue, NumericType offDiagValue) {
				unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
				unsigned column = blockIdx.y * blockDim.y + threadIdx.y;
				
				if(row >= m || column >= n)
					return;
				
				unsigned index = column * lda + row;
				
				const NumericType ReuseValueFactor = ReuseValue ? 1.0 : 0.0;
				if (row == column) {
					A[index] = A[index] * ReuseValueFactor + diagValue;
				} else {
					A[index] = A[index] * ReuseValueFactor + offDiagValue;
				}
			}
		}

		void fillMatrix(const MatrixDescription<float>& target, float diagValue, float offDiagValue, cudaStream_t stream /* = nullptr */) {
			if (target.format != StorageFormat::Dense) {
				Logging::instance().error()
					.print("[ERROR] " NMFGPU_FILE_LINE_PREFIX ": Input matrix must be stored in dense format!").lineFeed();
				return;
			}

			auto m = static_cast<unsigned>(target.rows);
			auto n = static_cast<unsigned>(target.columns);
			dim3 blockDim(32, 32);
			dim3 gridDim((m + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);
			impl::fillMatrix<float, false><<<gridDim, blockDim, 0, stream>>>(m, n, target.dense.values, static_cast<unsigned>(target.dense.leadingDimension), diagValue, offDiagValue);
		}

		void fillMatrix(const MatrixDescription<double>& target, double diagValue, double offDiagValue, cudaStream_t stream /* = nullptr */) {
			if (target.format != StorageFormat::Dense) {
				Logging::instance().error()
					.print("[ERROR] " NMFGPU_FILE_LINE_PREFIX ": Input matrix must be stored in dense format!").lineFeed();
				return;
			}

			auto m = static_cast<unsigned>(target.rows);
			auto n = static_cast<unsigned>(target.columns);
			dim3 blockDim(32, 32);
			dim3 gridDim((m + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);
			impl::fillMatrix<double, false><<<gridDim, blockDim, 0, stream>>>(m, n, target.dense.values, static_cast<unsigned>(target.dense.leadingDimension), diagValue, offDiagValue);
		}

		void addConstantToMatrix(const MatrixDescription<float>& target, float diagValue, float offDiagValue, cudaStream_t stream /* = nullptr */) {
			if (target.format != StorageFormat::Dense) {
				Logging::instance().error()
					.print("[ERROR] " NMFGPU_FILE_LINE_PREFIX ": Input matrix must be stored in dense format!").lineFeed();
				return;
			}

			auto m = static_cast<unsigned>(target.rows);
			auto n = static_cast<unsigned>(target.columns);
			dim3 blockDim(32, 32);
			dim3 gridDim((m + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);
			impl::fillMatrix<float, true><<<gridDim, blockDim, 0, stream >> >(m, n, target.dense.values, static_cast<unsigned>(target.dense.leadingDimension), diagValue, offDiagValue);
		}

		void addConstantToMatrix(const MatrixDescription<double>& target, double diagValue, double offDiagValue, cudaStream_t stream /* = nullptr */) {
			if (target.format != StorageFormat::Dense) {
				Logging::instance().error()
					.print("[ERROR] " NMFGPU_FILE_LINE_PREFIX ": Input matrix must be stored in dense format!").lineFeed();
				return;
			}

			auto m = static_cast<unsigned>(target.rows);
			auto n = static_cast<unsigned>(target.columns);
			dim3 blockDim(32, 32);
			dim3 gridDim((m + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);
			impl::fillMatrix<double, true><<<gridDim, blockDim, 0, stream >> >(m, n, target.dense.values, static_cast<unsigned>(target.dense.leadingDimension), diagValue, offDiagValue);
		}
	}
}
