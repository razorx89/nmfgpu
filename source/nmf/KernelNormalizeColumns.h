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
