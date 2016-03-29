/*
nmfgpu - CUDA accelerated computation of Non-negative Matrix Factorizations (NMF)

Copyright (C) 2015-2016  Sven Koitka (svenkoitka@fh-dortmund.de)

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