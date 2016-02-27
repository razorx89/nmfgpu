#include <common/Matrix.h>

namespace nmfgpu {
	bool operator == (const MatrixDimension& left, const MatrixDimension& right) {
		return left.rows == right.rows && left.columns == right.columns;
	}

	bool operator != (const MatrixDimension& left, const MatrixDimension& right) {
		return left.rows != right.rows || left.columns != right.columns;
	}
}