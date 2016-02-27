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