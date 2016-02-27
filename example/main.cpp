#include <algorithm>
#include <cstring>
#include <iostream>
#define NMFGPU_STATIC_LINKING
#include <nmfgpu.h>
#include <random>
#include <time.h>
#include <vector>

#define M 4096
#define N 165
#define R 158

int main(int argc, char** argv) {
	// Initialize the library
	if (nmfgpu::ResultType::Success == nmfgpu::initialize()) {
		// Check parameters
		if (argc > 1) {
			bool error = false;
			for (auto i = 1; i < argc; ++i) {
				if (strcmp(argv[i], "--gpu-index") == 0 || strcmp(argv[i], "-g") == 0) {
					if (i + 1 >= argc) {
						std::cerr << "--gpu-index: No index provided!" << std::endl << std::endl;
						error = true;
						break;
					} else {
						char* endptr;
						auto index = std::strtoul(argv[++i], &endptr, 10);
						if (endptr == argv[i]) {
							std::cerr << "--gpu-index: Invalid number format!" << std::endl << std::endl;
							error = true;
							break;
						}

						auto result = nmfgpu::chooseGpu(index);
						if (result != nmfgpu::ResultType::Success) {
							std::cerr << "--gpu-index: Device cannot be selected!" << std::endl << std::endl;
							return -1;
						} else {
							std::cout << "CUDA device #" << index << " selected for computation" << std::endl;
						}
					}
				} else {
					std::cerr << "Unknown command line argument!" << std::endl << std::endl;
					error = true;
					break;
				}
			}

			if (error) {
				std::cerr << "\t nmfgpu_example [--gpu-index <index>]" << std::endl << std::endl;
				return -1;
			}
		}

		// Allocate memory
		auto matV = std::vector<double>(M * N);
		auto matW = std::vector<double>(M * R);
		auto matH = std::vector<double>(R * N);

		// Initialize data
		srand(time(0));
		for (auto& v : matV) v = (rand() % 255) / 255.0;// std::max(0.0, v);
		for (auto& v : matW) v = (rand() % 255) / 255.0;// std::max(0.0, v);
		for (auto& v : matH) v = (rand() % 255) / 255.0;// std::max(0.0, v);

		// Construct algorithm parameter
		auto context = nmfgpu::NmfDescription<double>();		
		context.inputMatrix.rows = M;
		context.inputMatrix.columns = N;
		context.inputMatrix.format = nmfgpu::StorageFormat::Dense;
		context.inputMatrix.dense.values = matV.data();
		context.inputMatrix.dense.leadingDimension = M;
		
		context.features = R;
		context.initMethod = nmfgpu::NmfInitializationMethod::AllRandomValues;
		context.numIterations = 2000;

		context.outputMatrixW.rows = M;
		context.outputMatrixW.columns = R;
		context.outputMatrixW.format = nmfgpu::StorageFormat::Dense;
		context.outputMatrixW.dense.values = matW.data();
		context.outputMatrixW.dense.leadingDimension = M;

		context.outputMatrixH.rows = R;
		context.outputMatrixH.columns = N;
		context.outputMatrixH.format = nmfgpu::StorageFormat::Dense;
		context.outputMatrixH.dense.values = matH.data();
		context.outputMatrixH.dense.leadingDimension = R;
		
		context.numRuns = 1;
		context.seed = time(nullptr);
		context.thresholdType = nmfgpu::NmfThresholdType::Frobenius;
		context.thresholdValue = 0.01;
		context.callbackUserInterrupt = nullptr;
		context.useConstantBasisVectors = false;

		nmfgpu::Parameter parameters[]{ 
			{"lambdaH", 0.01},
			{"lambdaW", 0.01},
			{"alphaH", 0.01},
			{"alphaW", 0.01},
			{"lambda", 0.01},
			{"theta", 0.5},
		};
		context.parameters = parameters;
		context.numParameters = 6;

		context.algorithm = nmfgpu::NmfAlgorithm::nsNMF;
		nmfgpu::compute(context, nullptr);
		
		// Check frobenius norm on cpu side
		std::cout << "Host verification:" << std::endl;
		double frobenius = 0.0;
		for (int column = 0; column < N; ++column) {
			for (int row = 0; row < M; ++row) {
				double sum = 0.0;
				for (int feature = 0; feature < R; ++feature) {
					sum += matW[feature * M + row] * matH[column * R + feature];
				}
				frobenius += pow(matV[column * M + row] - sum, 2.0);
			}
		}
		frobenius = std::sqrt(frobenius);
		std::cout << "Frobenius norm: " << frobenius << std::endl;

		// Finalize the library
		nmfgpu::finalize();
	} else {
		std::cerr << "[ERROR] Failed to initialize the nmfgpu library!" << std::endl;
	}

	std::cout << "Please press any key to continue...";
	std::cin.get();

	return 0;
}