#include <common/Logging.h>
#include <common/MatrixDescription.h>
#include <nmf/KernelHelper.cuh>
#include <nmf/KernelMakeNonNegative.h>

namespace nmfgpu {
	namespace kernel {
		namespace impl {
			template<typename NumericType, bool transpose>
			__global__ void kernelMakeNonNegative(unsigned m, unsigned n, const NumericType* A, unsigned ldda, NumericType* B, unsigned lddb) {
				unsigned column = blockIdx.x * blockDim.x + threadIdx.x;
				unsigned row = blockIdx.y * blockDim.y + threadIdx.y;

				if (row >= m || column >= n)
					return;

				NumericType value;
				if (transpose) {
					value = CUDA_READ_ONLY_CACHE(A[row * ldda + column]);
				} else {
					value = A[column * ldda + row];
				}

				B[column * lddb + row] = max(value, NumericType(0.f));
			}
		}

		void makeNonNegative(const MatrixDescription<float>& matrix, cudaStream_t stream) {
			if (matrix.format != StorageFormat::Dense) {
				Logging::instance().error()
					.print("[ERROR] " NMFGPU_FILE_LINE_PREFIX ": Input matrix must be stored in dense format!").lineFeed();
				return;
			}

			auto m = static_cast<unsigned>(matrix.rows);
			auto n = static_cast<unsigned>(matrix.columns);
			auto ld = static_cast<unsigned>(matrix.dense.leadingDimension);
			dim3 blockDim(32, 32);
			dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);
			impl::kernelMakeNonNegative<float, false><<<gridDim, blockDim, 0, stream>>>(m, n, matrix.dense.values, ld, matrix.dense.values, ld);
		}

		void makeNonNegative(const MatrixDescription<double>& matrix, cudaStream_t stream) {
			if (matrix.format != StorageFormat::Dense) {
				Logging::instance().error()
					.print("[ERROR] " NMFGPU_FILE_LINE_PREFIX ": Input matrix must be stored in dense format!").lineFeed();
				return;
			}

			auto m = static_cast<unsigned>(matrix.rows);
			auto n = static_cast<unsigned>(matrix.columns);
			auto ld = static_cast<unsigned>(matrix.dense.leadingDimension);
			dim3 blockDim(32, 32);
			dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);
			impl::kernelMakeNonNegative<double, false><<<gridDim, blockDim, 0, stream>>>(m, n, matrix.dense.values, ld, matrix.dense.values, ld);
		}
	}
}
