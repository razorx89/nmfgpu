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