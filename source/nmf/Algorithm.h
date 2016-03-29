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

#include <nmfgpu.h>
#include <functional>

namespace nmfgpu {
	template<typename NumericType>
	class DeviceMatrix;

	template<typename NumericType>
	struct NMFCoreMatrices {
		DeviceMatrix<NumericType> deviceV;
		DeviceMatrix<NumericType> deviceW;
		DeviceMatrix<NumericType> deviceH;
	};

	class IAlgorithm {
		double m_frobeniusNorm{ 0.0 };
		double m_rmsd{ 0.0 };
		std::function<unsigned()> m_randomGenerator;

	protected:
		IAlgorithm(unsigned seed);

		/** Sets the new remaining error as both frobenius norm and root-mean-squared deviation.
		@param frobenius The frobenius norm of the current approximation.
		@param rmsd The root-mean-squared deviation of the current approximation. */
		void setRemainingError(double frobenius, double rmsd) {
			m_frobeniusNorm = frobenius;
			m_rmsd = rmsd;
		}

		unsigned generateRandomNumber();

	public:
		/** Computes one iteration of the algorithm and optionally the remaining error.
		@param computeError If true the remaining error will be computed and the members get updated.*/
		virtual void computeIteration(bool computeError) = 0;
		
		/** Gets the frobenius norm of the current approximation. 
		@returns Frobenius norm of the current approximation. */
		double frobeniusNorm() const { return m_frobeniusNorm; }

		/** Gets the root-mean-squared deviation of the current approximation.
		@returns Root-mean-squared deviation of the current approximation. */
		double rmsd() const { return m_rmsd; }

		virtual ResultType allocateMemory() = 0;

		virtual void deallocateMemory() = 0;

		virtual void initialize() = 0;

		virtual void storeFactorization() = 0;
		virtual const char* name() const = 0;
	};
}