#pragma once

#include <cuda_runtime_api.h>
#include <common/Memory.h>

namespace nmfgpu {
	// Forward-Declaration
	template<typename NumericType>
	struct MatrixDescription;

	namespace kernel {
		size_t traceMultiplicationGetElementCount(size_t diagonalElements);

		void traceMultiplication(bool transposeLeft, const MatrixDescription<float>& left, const MatrixDescription<float>& right, DeviceMemory<float>& partialSums, cudaStream_t stream = nullptr);

		void traceMultiplication(bool transposeLeft, const MatrixDescription<double>& left, const MatrixDescription<double>& right, DeviceMemory<double>& partialSums, cudaStream_t stream = nullptr);
	}
}
