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