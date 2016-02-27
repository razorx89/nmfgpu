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