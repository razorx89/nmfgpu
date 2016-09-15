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

#pragma once

#include <cuda_runtime_api.h>

namespace nmfgpu {
	// Forward-Declaration
	template<typename NumericType>
	struct MatrixDescription;

	namespace kernel {
		/** Normalizes the columns of a single-precision matrix.
		@param target Target matrix in single-precision format.
		@param stream CUDA stream for asynchronous kernel execution. */
		void normalizeColumns(const MatrixDescription<float>& target, cudaStream_t stream = nullptr);
		
		/** Normalizes the columns of a double-precision matrix.
		@param target Target matrix in double-precision format. 
		@param stream CUDA stream for asynchronous kernel execution. */
		void normalizeColumns(const MatrixDescription<double>& target, cudaStream_t stream = nullptr);
	}
}
