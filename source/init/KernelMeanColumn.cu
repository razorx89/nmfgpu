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
#include <init/KernelMeanColumn.h>
#include <stdexcept>

namespace nmfgpu {
	namespace kernel {
		namespace impl {
			template<typename NumericType>
			__global__ void kernelMeanColumn(NumericType* matrixA, unsigned ldda, unsigned rows, unsigned columnsA, const NumericType* matrixB, unsigned lddb, unsigned meanColumnCount, const unsigned* randomColumnIndices) {
				// Determine which matrix element has to be processed
				unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
				unsigned column = blockIdx.y * blockDim.y + threadIdx.y;

				if (row >= rows || column >= columnsA)
					return;

				// Fetch index of column to add to result
				NumericType result = 0;
				for (int i = 0; i < meanColumnCount; ++i) {
					unsigned randomColumn = randomColumnIndices[column * meanColumnCount + i];

					// Fetch elements from source column
					result += matrixB[randomColumn * lddb + row];
				}

				// Write result
				matrixA[column * ldda + row] = result / meanColumnCount;
			}

			template<typename NumericType>
			void meanColumn(const MatrixDescription<NumericType>& target, const MatrixDescription<NumericType>& matrixData, const unsigned* indices, unsigned meanCount) {
				if (target.format != StorageFormat::Dense || matrixData.format != StorageFormat::Dense) {
					Logging::instance().error()
						.print("[ERROR] " NMFGPU_FILE_LINE_PREFIX ": Input matrices must be stored in dense format!").lineFeed();
					return;
				}
				
				if (target.rows != matrixData.rows) {
					throw std::invalid_argument("Target and data matrix must have same row count!");
				}

				auto m = static_cast<unsigned>(target.rows);
				auto nA = static_cast<unsigned>(target.columns);
				//auto nB = static_cast<unsigned>(matrixData.columns);

				dim3 blockDim(32, 4);
				dim3 gridDim((m + blockDim.x - 1) / blockDim.x * blockDim.x, (nA + blockDim.y - 1) / blockDim.y * blockDim.y);
				impl::kernelMeanColumn<NumericType><<<gridDim, blockDim>>>(target.dense.values, target.dense.leadingDimension, m, nA, matrixData.dense.values, matrixData.dense.leadingDimension, meanCount, indices);
			}
		}

		void meanColumn(const MatrixDescription<float>& target, const MatrixDescription<float>& matrixData, const unsigned* indices, unsigned meanCount /* = 5 */) {
			impl::meanColumn(target, matrixData, indices, meanCount);
		}

		void meanColumn(const MatrixDescription<double>& target, const MatrixDescription<double>& matrixData, const unsigned* indices, unsigned meanCount /* = 5 */) {
			impl::meanColumn(target, matrixData, indices, meanCount);
		}
	}
}