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

#include <common/Matrix.h>
#include <init/InitializationStrategy.h>
#include <nmfgpu.h>

namespace nmfgpu {

	template<typename NumericType>
	class CopyStrategy : public InitializationStrategy<NumericType> {
		MatrixDescription<NumericType> m_hostMatrixW;
		MatrixDescription<NumericType> m_hostMatrixH;

	public:
		CopyStrategy(const MatrixDescription<NumericType>& hostMatrixW, const MatrixDescription<NumericType>& hostMatrixH)
			: InitializationStrategy<NumericType>(0)
			, m_hostMatrixW(hostMatrixW)
			, m_hostMatrixH(hostMatrixH) { }

		virtual void initializeMatrixH(DeviceMatrix<NumericType>& matrixH) override {
			matrixH.copyFrom(m_hostMatrixH);
		}
		virtual void initializeMatrixW(DeviceMatrix<NumericType>& matrixW) override {
			matrixW.copyFrom(m_hostMatrixW);
		}
	};
}