#pragma once

#include <memory>

namespace nmfgpu {
	template<typename NumericType>
	class Context;

	template<typename NumericType>
	class DeviceMatrix;

	template<typename NumericType>
	struct NMFCoreMatrices;

	/** Interface for different initialization strategies of the matrix pair W and H. The initialization strategy must support 
	initialization of both matrices, but the execution of matrix H initialization can be omitted if it is not required. */
	template<typename NumericType>
	class InitializationStrategy {		
		/** Stores the current seed and gets incremented after each execution. */
		unsigned m_seed;

	protected:
		/** Returns the current seed for the random generator. 
		@returns Current seed for the random generator. */
		unsigned seed() const { return m_seed; }

	public:
		static std::unique_ptr<InitializationStrategy<NumericType>> create(const NmfDescription<NumericType>& data, const NMFCoreMatrices<NumericType>& coreMatrices);

		/** Initializes the strategy base.
		@param initializeH Flag if matrix H needs to be initialized. */
		InitializationStrategy(unsigned seed)
			: m_seed{ seed } { }

		/** Invokes the initialization strategy of the derived class to initialize matrix W and if neccessary matrix H. */
		virtual void initializeMatrixH(DeviceMatrix<NumericType>& matrixH) = 0;
		virtual void initializeMatrixW(DeviceMatrix<NumericType>& matrixW) = 0;
	};

	template<>
	/* static */ std::unique_ptr<InitializationStrategy<float>> InitializationStrategy<float>::create(const NmfDescription<float>& description, const NMFCoreMatrices<float>& coreMatrices);

	template<>
	/* static */ std::unique_ptr<InitializationStrategy<double>> InitializationStrategy<double>::create(const NmfDescription<double>& description, const NMFCoreMatrices<double>& coreMatrices);
}