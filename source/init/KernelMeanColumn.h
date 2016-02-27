#pragma once

namespace nmfgpu {
	template<typename NumericType>
	struct MatrixDescription;

	namespace kernel {
		void meanColumn(const MatrixDescription<float>& target, const MatrixDescription<float>& matrixData, const unsigned* indices, unsigned meanCount = 5);
		void meanColumn(const MatrixDescription<double>& target, const MatrixDescription<double>& matrixData, const unsigned* indices, unsigned meanCount = 5);
	}
}