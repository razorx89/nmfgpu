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
