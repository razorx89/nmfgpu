#pragma once

namespace nmfgpu {
	template<typename NumericType>
	class DeviceMatrix;

	namespace kernels {
		/** Computes the distances of each column of the left matrix to each column of the right matrix. */
		void computeDistanceMatrix(const DeviceMatrix<float>& matrixLeft, const DeviceMatrix<float>& matrixRight, const DeviceMatrix<float>& matrixDistances);
		void computeDistanceMatrix(const DeviceMatrix<double>& matrixLeft, const DeviceMatrix<double>& matrixRight, const DeviceMatrix<double>& matrixDistances);
	}
}