#pragma once

#include <init/InitializationStrategy.h>

namespace nmfgpu {
	namespace details {
		void fillRandom(DeviceMatrix<float>& matrix, int seed);
		void fillRandom(DeviceMatrix<double>& matrix, int seed);
	}

	template<typename NumericType>
	class RandomValueStrategy : public InitializationStrategy<NumericType> {

	public:
		RandomValueStrategy(int seed)
			: InitializationStrategy<NumericType>(seed) { }

		void initializeMatrixH(DeviceMatrix<NumericType>& matrixH) override;
		void initializeMatrixW(DeviceMatrix<NumericType>& matrixW) override;
	};

	template<>
	void RandomValueStrategy<float>::initializeMatrixH(DeviceMatrix<float>& matrixH);
	template<>
	void RandomValueStrategy<double>::initializeMatrixH(DeviceMatrix<double>& matrixH);

	template<>
	void RandomValueStrategy<float>::initializeMatrixW(DeviceMatrix<float>& matrixW);
	template<>
	void RandomValueStrategy<double>::initializeMatrixW(DeviceMatrix<double>& matrixW);
}