#pragma once

#include <init/InitializationStrategy.h>
#include <stdexcept>

namespace nmfgpu {
	template<typename NumericType>
	class MeanColumnStrategy : public InitializationStrategy<NumericType> {
		const DeviceMatrix<NumericType>& m_matrixV;
		unsigned m_meanColumnCount;
	public:
		MeanColumnStrategy(int seed, const DeviceMatrix<NumericType>& matrixV, unsigned meanColumnCount = 5)
			: InitializationStrategy<NumericType>(seed)
			, m_matrixV(matrixV) 
			, m_meanColumnCount(meanColumnCount) { 
			if (meanColumnCount == 0 || meanColumnCount > matrixV.columns()) {
				throw std::invalid_argument("At least one column but at most <column count> columns can be processed!");
			}
		}

		void initializeMatrixH(DeviceMatrix<NumericType>& matrixH) override;
		void initializeMatrixW(DeviceMatrix<NumericType>& matrixW) override;
	};

	template<>
	void MeanColumnStrategy<float>::initializeMatrixH(DeviceMatrix<float>& matrixH);

	template<>
	void MeanColumnStrategy<double>::initializeMatrixH(DeviceMatrix<double>& matrixH);


	template<>
	void MeanColumnStrategy<float>::initializeMatrixW(DeviceMatrix<float>& matrixW);

	template<>
	void MeanColumnStrategy<double>::initializeMatrixW(DeviceMatrix<double>& matrixW);
}