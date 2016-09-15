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