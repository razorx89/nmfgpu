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

namespace NMF {
	namespace Kernel {
		/**

		@param[in] m

		@param[in] n

		@param[in] r

		@param[in] A

		@param[in] ldda

		@param[in] B

		@param[in] lddb

		@param[out] partialSums

		@param[in,out] psLength

		*/
		void multiplyTrace(bool transposeA, unsigned m, unsigned n, unsigned r, const float* A, unsigned ldda, const float* B, unsigned lddb, float* partialSums, unsigned& psLength);

		/** @copydoc nmf_kernel_mul_trace(unsigned, unsigned, unsigned, const float*, unsigned, const float*, unsigned, float*, unsigned&) */
		void multiplyTrace(bool transposeA, unsigned m, unsigned n, unsigned r, const double* A, unsigned ldda, const double* B, unsigned lddb, double* partialSums, unsigned& psLength);

		float resolveMultiplyTrace(const float* partialSums, unsigned psLength);
		double resolveMultiplyTrace(const double* partialSums, unsigned psLength);
	}
}
