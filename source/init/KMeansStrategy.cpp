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

#include <common/Matrix.h>
#include <init/EInNMF.h>
#include <init/KMeansStrategy.h>
#include <init/RandomValueStrategy.h>
#include <kmeans/kMeans.h>

namespace nmfgpu {
	namespace details {
		template<typename NumericType>
		void initializeMatrixHGeneric(NmfInitializationMethod method, const DeviceMatrix<NumericType>& matrixV, const DeviceMatrix<NumericType>& matrixW, DeviceMatrix<NumericType>& matrixH, unsigned seed) {
			if (method == NmfInitializationMethod::KMeansAndRandomValues) {
				details::fillRandom(matrixH, seed);
			} else if (method == NmfInitializationMethod::KMeansAndNonNegativeWTV) {
				matrixH = matrixW.transposed() * matrixV;
				matrixH.setNegativeToZero();
			} else if (method == NmfInitializationMethod::EInNMF) {
				kernels::computeDistanceMatrix(matrixW, matrixV, matrixH);
			}
		}
	}

	template<>
	void KMeansStrategy<float>::initializeMatrixH(DeviceMatrix<float>& matrixH) {
		details::initializeMatrixHGeneric(m_method, m_matrixV, m_savedMatrixW, matrixH, seed() + 1);
	}

	template<>
	void KMeansStrategy<double>::initializeMatrixH(DeviceMatrix<double>& matrixH) {
		details::initializeMatrixHGeneric(m_method, m_matrixV, m_savedMatrixW, matrixH, seed() + 1);
	}

	template<>
	void KMeansStrategy<float>::initializeMatrixW(DeviceMatrix<float>& matrixW) {
		m_dataMembership.allocate(m_matrixV.columns());
		computeKMeans(m_matrixV.description(), matrixW.description(), m_dataMembership, seed(), 100, 0.005);
		m_savedMatrixW = matrixW;
	}

	template<>
	void KMeansStrategy<double>::initializeMatrixW(DeviceMatrix<double>& matrixW) {
		m_dataMembership.allocate(m_matrixV.columns());
		computeKMeans(m_matrixV.description(), matrixW.description(), m_dataMembership, seed(), 100, 0.005);
		m_savedMatrixW = matrixW;
	}
}