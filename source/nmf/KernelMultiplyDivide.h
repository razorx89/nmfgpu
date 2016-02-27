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
