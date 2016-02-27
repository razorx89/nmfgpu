#include <algorithm>
#include <common/Matrix.h>
#include <init/KernelMeanColumn.h>
#include <init/MeanColumnStrategy.h>
#include <init/RandomValueStrategy.h>
#include <functional>
#include <random>

namespace nmfgpu {
	template<>
	void MeanColumnStrategy<float>::initializeMatrixH(DeviceMatrix<float>& matrixH) {
		details::fillRandom(matrixH, seed());
	}

	template<>
	void MeanColumnStrategy<double>::initializeMatrixH(DeviceMatrix<double>& matrixH) {
		details::fillRandom(matrixH, seed());
	}

	namespace details {
		template<typename NumericType>
		void initializeMeanColumn(DeviceMatrix<NumericType>& target, const DeviceMatrix<NumericType>& matrixData, unsigned meanColumnCount, unsigned seed) {
			// Generate random column indices
			auto hostIndices = HostMemory<unsigned>(meanColumnCount * target.columns());
			auto rand = std::bind(std::uniform_int_distribution<unsigned>(0, matrixData.columns() - 1), std::mt19937(seed));

			std::generate(&hostIndices.at(0), &hostIndices.at(hostIndices.elements() - 1), rand);

			// Copy to device memory
			auto deviceIndices = DeviceMemory<unsigned>(meanColumnCount * target.columns());
			hostIndices.copyTo(deviceIndices);

			// Execute initialization
			kernel::meanColumn(target.description(), matrixData.description(), deviceIndices.get(), meanColumnCount);
		}
	}

	template<>
	void MeanColumnStrategy<float>::initializeMatrixW(DeviceMatrix<float>& matrixW) {
		details::initializeMeanColumn(matrixW, m_matrixV, m_meanColumnCount, seed());
	}

	template<>
	void MeanColumnStrategy<double>::initializeMatrixW(DeviceMatrix<double>& matrixW) {
		details::initializeMeanColumn(matrixW, m_matrixV, m_meanColumnCount, seed());
	}
}