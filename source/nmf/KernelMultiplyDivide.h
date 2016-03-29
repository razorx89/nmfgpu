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

#pragma once

#include <cuda_runtime_api.h>

namespace nmfgpu {
	template<typename NumericType>
	struct MatrixDescription;

	namespace kernel {
		/** Performs an element-wise multiplication of the single-precision target matrix with the numerator matrix, followed by an element-wise divide with the denominator matrix.
		The denominator will be increased by @p epsilon to prevent division by zero.
		@param target Target matrix which should be multiplied and divided. 
		@param numerator Matrix for element-wise multiplication.
		@param denominator Matrix for element-wise division.
		@param stream CUDA stream for asynchronous kernel execution. */
		void multiplyDivide(const MatrixDescription<float>& target, const MatrixDescription<float>& numerator, const MatrixDescription<float>& denominator, float epsilon, cudaStream_t stream /* = nullptr */);

		/** Performs an element-wise multiplication of the double-precision target matrix with the numerator matrix, followed by an element-wise divide with the denominator matrix.
		The denominator will be increased by @p epsilon to prevent division by zero.
		@param target Target matrix which should be multiplied and divided.
		@param numerator Matrix for element-wise multiplication.
		@param denominator Matrix for element-wise division.
		@param stream CUDA stream for asynchronous kernel execution. */
		void multiplyDivide(const MatrixDescription<double>& target, const MatrixDescription<double>& numerator, const MatrixDescription<double>& denominator, double epsilon, cudaStream_t stream /* = nullptr */);
	}
}
