/*
nmfgpu - CUDA accelerated computation of Non-negative Matrix Factorizations (NMF)

Copyright (C) 2015-2016  Sven Koitka (sven.koitka@fh-dortmund.de)

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

#include <curand.h>
#include <common/Logging.h>
#include <common/Matrix.h>
#include <init/RandomValueStrategy.h>

namespace nmfgpu {
	namespace details {
		void fillRandom(DeviceMatrix<float>& matrix, int seed) {
			// Create and initialize generator
			curandGenerator_t generator;
			CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
			CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, seed));

			// Initialize matrix random generated values in (0, 1]
			CURAND_CALL(curandGenerateUniform(generator, matrix.get(), matrix.leadingDimension() * matrix.columns()));
			CURAND_CALL(curandDestroyGenerator(generator));
		}

		void fillRandom(DeviceMatrix<double>& matrix, int seed) {
			// Create and initialize generator
			curandGenerator_t generator;
			CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
			CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, seed));

			// Initialize matrix random generated values in (0, 1]
			CURAND_CALL(curandGenerateUniformDouble(generator, matrix.get(), matrix.leadingDimension() * matrix.columns()));
			CURAND_CALL(curandDestroyGenerator(generator));
		}
	}

	template<>
	void RandomValueStrategy<float>::initializeMatrixH(DeviceMatrix<float>& matrixH) {
		details::fillRandom(matrixH, seed());
	}

	template<>
	void RandomValueStrategy<double>::initializeMatrixH(DeviceMatrix<double>& matrixH) {
		details::fillRandom(matrixH, seed());
	}
	
	template<>
	void RandomValueStrategy<float>::initializeMatrixW(DeviceMatrix<float>& matrixW) {
		details::fillRandom(matrixW, seed());
	}

	template<>
	void RandomValueStrategy<double>::initializeMatrixW(DeviceMatrix<double>& matrixW) {
		details::fillRandom(matrixW, seed());
	}
}