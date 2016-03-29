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

namespace nmfgpu {
	template<typename NumericType>
	struct MatrixDescription;

	namespace kernel {
		void meanColumn(const MatrixDescription<float>& target, const MatrixDescription<float>& matrixData, const unsigned* indices, unsigned meanCount = 5);
		void meanColumn(const MatrixDescription<double>& target, const MatrixDescription<double>& matrixData, const unsigned* indices, unsigned meanCount = 5);
	}
}