#pragma once

#include <init/InitializationStrategy.h>
#include <stdexcept>

namespace nmfgpu {
	template<typename NumericType>
	class KMeansStrategy : public InitializationStrategy<NumericType> {
		const DeviceMatrix<NumericType>& m_matrixV;
		DeviceMemory<unsigned> m_dataMembership;
		DeviceMatrix<NumericType> m_savedMatrixW;
		NmfInitializationMethod m_method;

	public:
		KMeansStrategy(int seed, NmfInitializationMethod method, const DeviceMatrix<NumericType>& matrixV)
			: InitializationStrategy<NumericType>(seed)
			, m_matrixV(matrixV)
			, m_method(method) {
		}

		void initializeMatrixH(DeviceMatrix<NumericType>& matrixH) override;
		void initializeMatrixW(DeviceMatrix<NumericType>& matrixW) override;
	};

	template<>
	void KMeansStrategy<float>::initializeMatrixH(DeviceMatrix<float>& matrixH);

	template<>
	void KMeansStrategy<double>::initializeMatrixH(DeviceMatrix<double>& matrixH);


	template<>
	void KMeansStrategy<float>::initializeMatrixW(DeviceMatrix<float>& matrixW);

	template<>
	void KMeansStrategy<double>::initializeMatrixW(DeviceMatrix<double>& matrixW);
}