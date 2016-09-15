/*
nmfgpu - CUDA accelerated computation of Non-negative Matrix Factorizations (NMF)

Copyright (C) 2015-2016  Sven Koitka (sven.koitka@fh-dortmund.de)

This file is part of nmfgpu.

nmfgpu is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

nmfgpu is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with nmfgpu.  If not, see <http://www.gnu.org/licenses/>.
*/

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