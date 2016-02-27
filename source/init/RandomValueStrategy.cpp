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