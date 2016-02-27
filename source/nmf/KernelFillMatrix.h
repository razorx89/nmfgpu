#pragma once

#include <cuda_runtime_api.h>

namespace nmfgpu {
	template<typename NumericType>
	struct MatrixDescription;

	namespace kernel {
		/** 
		@param target Target matrix which should be multiplied and divided. 
		@param diagValue
		@param offDiagValue
		@param stream CUDA stream for asynchronous kernel execution. */
		void fillMatrix(const MatrixDescription<float>& target, float diagValue, float offDiagValue, cudaStream_t stream /* = nullptr */);

		/** 
		@param target Target matrix which should be multiplied and divided.
		@param diagValue
		@param offDiagValue
		@param stream CUDA stream for asynchronous kernel execution. */
		void fillMatrix(const MatrixDescription<double>& target, double diagValue, double offDiagValue, cudaStream_t stream /* = nullptr */);

		void addConstantToMatrix(const MatrixDescription<float>& target, float diagValue, float offDiagValue, cudaStream_t stream /* = nullptr */);
		void addConstantToMatrix(const MatrixDescription<double>& target, double diagValue, double offDiagValue, cudaStream_t stream /* = nullptr */);
	}
}
