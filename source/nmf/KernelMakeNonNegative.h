#pragma once

#include <cuda_runtime_api.h>

namespace nmfgpu {
	template<typename NumericType>
	struct MatrixDescription;

	namespace kernel {
		void makeNonNegative(const MatrixDescription<float>& matrix, cudaStream_t stream);
		void makeNonNegative(const MatrixDescription<double>& matrix, cudaStream_t stream);
	}
}