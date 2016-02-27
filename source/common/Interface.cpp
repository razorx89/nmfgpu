#include <common/Interface.h>
#include <common/Memory.h>
#include <common/Misc.h>
#include <kmeans/kMeans.h>
#include <memory>
#include <nmf/Algorithm.h>
#include <nmf/Context.h>
#include <nmf/AlgorithmAlternatingLeastSquares.h>
#include <nmf/AlgorithmAlternatingHoyerConstrainedLeastSquares.h>
#include <nmf/AlgorithmMultiplicativeFrobenius.h>
#include <nmf/AlgorithmGradientDescentConstrainedLeastSquares.h>
#include <nmf/AlgorithmNonSmoothNMF.h>
#include <nmf/SingleGpuDispatcher.h>
#include <nmf/Summary.h>
#include <init/InitializationStrategy.h>
#include <iostream>
#include <string.h>

namespace nmfgpu {
	int getParameterIndexFromName(Parameter* parameters, unsigned numParameters, const char* name) {
		for (auto it = parameters; it < parameters + numParameters; ++it) {
			if (strcmp(it->name, name) == 0) {
				return std::distance(parameters, it);
			}
		}

		return -1;
	}
	
	THREAD_LOCAL_STORAGE ExternalLibrariesContext* g_context = nullptr; 
	
	/* NMFGPU_EXPORT */ ResultType initialize() {
		if (g_context != nullptr) {
			return ResultType::ErrorAlreadyInitialized;
		}

		// Create context object
		g_context = new ExternalLibrariesContext();

		// FIXME: More accurate check to choose best device
		//cudaSetDevice(0);
		initializeLibraryContexts();

		return ResultType::Success;
	}

	/* NMFGPU_EXPORT */ ResultType finalize() {
		if (g_context == nullptr) {
			return ResultType::ErrorNotInitialized;
		}

		delete g_context;
		g_context = nullptr;

		return ResultType::Success;
	}

	/* NMFGPU_EXPORT */ int version() {
		return NMFGPU_VERSION;
	}

	ResultType initializeLibraryContexts() {
		if (g_context == nullptr) {
			return ResultType::ErrorNotInitialized;
		}

		// Initialize cuBLAS
		auto errorCUBLAS = cublasCreate(&g_context->cublasHandle);
		if (errorCUBLAS != CUBLAS_STATUS_SUCCESS) {
			Logging::instance().error()
				.print("Failed to initialize the cuBLAS library context (", errorCUBLAS, ")!")
				.lineFeed();

			return ResultType::ErrorExternalLibrary;
		}

		// Initialize cuSPARSE
		auto errorCUSPARSE = cusparseCreate(&g_context->cusparseHandle);
		if (errorCUSPARSE != CUSPARSE_STATUS_SUCCESS) {
			Logging::instance().error()
				.print("Failed to initialize the cuSPARSE library context (", errorCUSPARSE, ")!")
				.lineFeed();

			cublasDestroy(g_context->cublasHandle);

			return ResultType::ErrorExternalLibrary;
		}

		// Initialize cuSOLVER
		auto errorCUSOLVER = cusolverDnCreate(&g_context->cusolverDnHandle);
		if (errorCUSOLVER != CUSOLVER_STATUS_SUCCESS) {
			Logging::instance().error()
				.print("Failed to initialize the cuSolverDN library context (", errorCUSOLVER, ")!")
				.lineFeed();

			cublasDestroy(g_context->cublasHandle);
			cusparseDestroy(g_context->cusparseHandle);

			return ResultType::ErrorExternalLibrary;
		}

		return ResultType::Success;
	}

	ResultType finalizeLibraryContexts() {
		if (g_context == nullptr) {
			return ResultType::ErrorNotInitialized;
		}

		// Deinitialize cuSOLVER
		if (CUSOLVER_STATUS_SUCCESS != cusolverDnDestroy(g_context->cusolverDnHandle)) {
			return ResultType::ErrorExternalLibrary;
		}
		g_context->cusolverDnHandle = nullptr;

		// Deinitialize cuSPARSE
		if (CUSPARSE_STATUS_SUCCESS != cusparseDestroy(g_context->cusparseHandle)) {
			return ResultType::ErrorExternalLibrary;
		}
		g_context->cusparseHandle = nullptr;

		// Deinitialize cuBLAS
		if (CUBLAS_STATUS_SUCCESS != cublasDestroy(g_context->cublasHandle)) {
			return ResultType::ErrorExternalLibrary;
		}
		g_context->cublasHandle = nullptr;

		return ResultType::Success;
	}

	/* NMFGPU_EXPORT */ ResultType chooseGpu(unsigned index) {
		auto result = cudaSetDevice(int(index));
		if (result != cudaSuccess) {
			return ResultType::ErrorDeviceSelection;
		} else {
			g_context->deviceID = int(index); 

			finalizeLibraryContexts();
			initializeLibraryContexts();
			return ResultType::Success;
		}
	}

	/* NMFGPU_EXPORT */ unsigned getNumberOfGpu() {
		int num = 0;
		if (cudaSuccess != cudaGetDeviceCount(&num)) {
			return 0u;
		} else {
			return static_cast<unsigned>(num);
		}
	}

	/* NMFGPU_EXPORT */ ResultType getInformationForGpuIndex(unsigned index, GpuInformation& info) {
		int oldDevice = 0;
		auto result = cudaGetDevice(&oldDevice);
		if (result != cudaSuccess) {
			return ResultType::ErrorDeviceSelection;
		}
		result = cudaSetDevice(int(index));
		if (result != cudaSuccess) {
			return ResultType::ErrorDeviceSelection;
		}

		cudaDeviceProp props;
		result = cudaGetDeviceProperties(&props, index);
		if (result == cudaSuccess) {
			strcpy(info.name, props.name);
		} else {
			strcpy(info.name, "N/A");
		}

		result = cudaMemGetInfo(&info.freeMemory, &info.totalMemory);
		cudaSetDevice(oldDevice);
		if (result != cudaSuccess) {
			info.freeMemory = 0ull;
			info.totalMemory = 0ull;
			return ResultType::ErrorExternalLibrary;
		} else {
			return ResultType::Success;
		}
	}

	/* NMFGPU_EXPORT */ void setVerbosity(Verbosity verbosity) {
		Logging::instance().setVerbosity(verbosity);
	}

	/* static */ ISummary* ISummary::create() {
		return new Summary();
	}


	namespace details {
		template<typename NumericType>
		ResultType compute(NmfDescription<NumericType>& description, ISummary* summary) {
			if (g_context == nullptr) {
				return ResultType::ErrorNotInitialized;
			}

			// If NmfInitializationMethod == CopyExisting then no more than one run should be performed
			if (description.initMethod == NmfInitializationMethod::CopyExisting && description.numRuns > 1) {
				Logging::instance().summary()
					.print("[WARNING] When using the CopyExisting initialization method, then no more than one run should be performed because of missing randomization!").lineFeed();
				description.numRuns = 1;
			}

			// Abort if feature count is smaller than one of the data matrix dimensions, but allow it for SSNMF
			if (!description.useConstantBasisVectors && description.features > description.inputMatrix.columns) {
				Logging::instance().error()
					.print("[ERROR] Feature count has to be less than the matrix dimensions!").lineFeed();
				return ResultType::ErrorInvalidArgument;
			}

			// Create algorithm instance
			std::unique_ptr<IAlgorithm> algorithm;
			try {
				switch (description.algorithm) {
				case NmfAlgorithm::Multiplicative:
					algorithm = make_unique<AlgorithmMultiplicativeFrobenius<NumericType>>(description); 
					break;

				case NmfAlgorithm::ALS:
					algorithm = make_unique<AlgorithmAlternatingLeastSquares<NumericType>>(description); 
					break;

				case NmfAlgorithm::ACLS: {
					auto indexLambdaW = getParameterIndexFromName(description.parameters, description.numParameters, "lambdaW");
					if (indexLambdaW == -1) {
						Logging::instance().error()
							.print("[ERROR] ACLS algorithm requires parameter 'lambdaW' to be set!").lineFeed();
						return ResultType::ErrorInvalidArgument;
					}

					auto indexLambdaH = getParameterIndexFromName(description.parameters, description.numParameters, "lambdaH");
					if (indexLambdaH == -1) {
						Logging::instance().error()
							.print("[ERROR] ACLS algorithm requires parameter 'lambdaH' to be set!").lineFeed();
						return ResultType::ErrorInvalidArgument;
					}

					algorithm = make_unique<AlgorithmHoyerConstrainedAlternatingLeastSquares<NumericType>>(description, 
																											description.parameters[indexLambdaW].value, 
																											description.parameters[indexLambdaH].value); 
					break;
				}

				case NmfAlgorithm::AHCLS: {
					auto indexLambdaW = getParameterIndexFromName(description.parameters, description.numParameters, "lambdaW");
					if (indexLambdaW == -1) {
						Logging::instance().error()
							.print("[ERROR] AHCLS algorithm requires parameter 'lambdaW' to be set!").lineFeed();
						return ResultType::ErrorInvalidArgument;
					}

					auto indexLambdaH = getParameterIndexFromName(description.parameters, description.numParameters, "lambdaH");
					if (indexLambdaH == -1) {
						Logging::instance().error()
							.print("[ERROR] AHCLS algorithm requires parameter 'lambdaH' to be set!").lineFeed();
						return ResultType::ErrorInvalidArgument;
					}

					auto indexAlphaW = getParameterIndexFromName(description.parameters, description.numParameters, "alphaW");
					if (indexAlphaW == -1) {
						Logging::instance().error()
							.print("[ERROR] AHCLS algorithm requires parameter 'alphaW' to be set!").lineFeed();
						return ResultType::ErrorInvalidArgument;
					}

					auto indexAlphaH = getParameterIndexFromName(description.parameters, description.numParameters, "alphaH");
					if (indexAlphaH == -1) {
						Logging::instance().error()
							.print("[ERROR] AHCLS algorithm requires parameter 'alphaH' to be set!").lineFeed();
						return ResultType::ErrorInvalidArgument;
					}

					algorithm = make_unique<AlgorithmHoyerConstrainedAlternatingLeastSquares<NumericType>>(description,
																										   description.parameters[indexLambdaW].value,
																										   description.parameters[indexLambdaH].value,
																										   description.parameters[indexAlphaW].value,
																										   description.parameters[indexAlphaH].value); 
					break;
				}

				case NmfAlgorithm::GDCLS: {
					auto indexLambda = getParameterIndexFromName(description.parameters, description.numParameters, "lambda");
					if (indexLambda == -1) {
						Logging::instance().error()
							.print("[ERROR] GDCLS algorithm requires parameter 'lambda' to be set!").lineFeed();
						return ResultType::ErrorInvalidArgument;
					}

					algorithm = make_unique<AlgorithmGradientDescentConstrainedLeastSquares<NumericType>>(description,
																										  description.parameters[indexLambda].value); 
					break;
				}
				
				case NmfAlgorithm::nsNMF: {
					auto indexTheta = getParameterIndexFromName(description.parameters, description.numParameters, "theta");
					if (indexTheta == -1) {
						Logging::instance().error()
							.print("[ERROR] nsNMF algorithm requires parameter 'theta' to be set!").lineFeed();
						return ResultType::ErrorInvalidArgument;
					}

					algorithm = make_unique<AlgorithmNonSmoothNMF<NumericType>>(description, description.parameters[indexTheta].value);

					break;
				}

				default: 
					Logging::instance().error()
						.print("[ERROR] Chosen algorithm is not implemented!")
						.lineFeed();
					finalizeLibraryContexts();
					return ResultType::ErrorInvalidArgument;
				}
			} catch (const std::invalid_argument& e) {
				Logging::instance().error().print("[ERROR] ", e.what()).lineFeed();
				finalizeLibraryContexts();
				return ResultType::ErrorInvalidArgument;
			}

			auto dispatcher = nmfgpu::make_unique<SingleGpuDispatcher>(DispatcherConfig(description), algorithm, static_cast<Summary*>(summary));

			auto result = dispatcher->dispatch() ? ResultType::Success : ResultType::ErrorUserInterrupt;
			return result;
		}
	}

	/* NMFGPU_EXPORT */ ResultType compute(NmfDescription<float>& description, ISummary* summary) {
		return details::compute(description, summary);
	}

	/* NMFGPU_EXPORT */ ResultType compute(NmfDescription<double>& description, ISummary* summary) {
		return details::compute(description, summary);
	}

	namespace Details {
		template<typename NumericType>
		ResultType computeKMeans(nmfgpu::KMeansDescription<NumericType>& desc, KMeansSummary* summary) {
			if (g_context == nullptr) {
				return ResultType::ErrorNotInitialized;
			}

			// Check parameters
			if (desc.numClusters >= desc.inputMatrix.columns) {
				Logging::instance().error()
					.print(" [ERROR] ", NMFGPU_FILE_LINE_PREFIX, ": Number of clusters must be smaller than number of samples in dataset!").lineFeed();
				return ResultType::ErrorInvalidArgument;
			}

			if (desc.outputMatrixClusters.format != StorageFormat::Dense) {
				Logging::instance().error()
					.print(" [ERROR] ", NMFGPU_FILE_LINE_PREFIX, ": Cluster matrix must have a dense storage format!").lineFeed();
				return ResultType::ErrorInvalidArgument;
			}

			if (desc.inputMatrix.rows != desc.outputMatrixClusters.rows) {
				Logging::instance().error()
					.print(" [ERROR] ", NMFGPU_FILE_LINE_PREFIX, ": Input and output matrices must have the same amount of rows!").lineFeed();
				return ResultType::ErrorInvalidArgument;
			}

			if (desc.numClusters == 0 || desc.numClusters >= desc.inputMatrix.columns) {
				Logging::instance().error()
					.print(" [ERROR] ", NMFGPU_FILE_LINE_PREFIX, ": Number of clusters must be smaller than column count of the input matrix!").lineFeed();
				return ResultType::ErrorInvalidArgument;
			}

			// Allocate device memory
			DeviceMatrix<NumericType> deviceData;
			DeviceMatrix<NumericType> deviceClusters;
			DeviceMemory<unsigned> deviceMembership;

			if (!deviceData.allocate(desc.inputMatrix.rows, desc.inputMatrix.columns)
				|| !deviceClusters.allocate(desc.inputMatrix.rows, desc.numClusters)
				|| !deviceMembership.allocate(desc.inputMatrix.columns)) {
				return ResultType::ErrorNotEnoughDeviceMemory;
			}

			// Copy data to GPU-side
			deviceData.copyFrom(desc.inputMatrix);

			// Compute
			computeKMeans(deviceData.description(), deviceClusters.description(), deviceMembership, desc.seed, desc.numIterations, desc.thresholdValue);
			
			// Copy result back to CPU-side
			deviceClusters.copyTo(desc.outputMatrixClusters);

			// Copy memberships to CPU-side
			if (desc.outputMemberships != nullptr) {
				cudaMemcpy(desc.outputMemberships, deviceMembership.get(), deviceMembership.bytes(), cudaMemcpyDeviceToHost);
			}

			// Compute statistics if nessary
			if (summary != nullptr) {

			}

			return ResultType::Success;
		}
	}

	/* NMFGPU_EXPORT */ ResultType computeKMeans(nmfgpu::KMeansDescription<float>& desc, KMeansSummary* summary) {
		return Details::computeKMeans<float>(desc, summary);
	}

	/* NMFGPU_EXPORT */ ResultType computeKMeans(nmfgpu::KMeansDescription<double>& desc, KMeansSummary* summary) {
		return Details::computeKMeans<double>(desc, summary);
	}
}


/* NMFGPU_EXPORT */ nmfgpu::ResultType nmfgpu_initialize() {
	return nmfgpu::initialize();
}

/* NMFGPU_EXPORT */ nmfgpu::ResultType nmfgpu_finalize() {
	return nmfgpu::finalize();
}

/* NMFGPU_EXPORT */ int nmfgpu_version() {
	return nmfgpu::version();
}

/* NMFGPU_EXPORT */ void nmfgpu_set_verbosity(nmfgpu::Verbosity verbosity) {
	nmfgpu::setVerbosity(verbosity);
}

/* NMFGPU_EXPORT */ nmfgpu::ResultType nmfgpu_create_summary(nmfgpu::ISummary** summary) {
	if (summary == nullptr) {
		return nmfgpu::ResultType::ErrorInvalidArgument;
	}

	*summary = nmfgpu::ISummary::create();

	return nmfgpu::ResultType::Success;
}

/* NMFGPU_EXPORT */ nmfgpu::ResultType nmfgpu_compute_single(nmfgpu::NmfDescription<float>* description, nmfgpu::ISummary* summary) {
	return nmfgpu::compute(*description, summary);
}

/* NMFGPU_EXPORT */ nmfgpu::ResultType nmfgpu_compute_double(nmfgpu::NmfDescription<double>* description, nmfgpu::ISummary* summary) {
	return nmfgpu::compute(*description, summary);
}

/* NMFGPU_EXPORT */ nmfgpu::ResultType nmfgpu_compute_kmeans_single(nmfgpu::KMeansDescription<float>* desc) {
	return nmfgpu::computeKMeans(*desc, nullptr);
}

/* NMFGPU_EXPORT */ nmfgpu::ResultType nmfgpu_compute_kmeans_double(nmfgpu::KMeansDescription<double>* desc) {
	return nmfgpu::computeKMeans(*desc, nullptr);
}

/* NMFGPU_EXPORT */ nmfgpu::ResultType nmfgpu_choose_gpu(unsigned index) {
	return nmfgpu::chooseGpu(index);
}
/* NMFGPU_EXPORT */ unsigned nmfgpu_get_number_of_gpu() {
	return nmfgpu::getNumberOfGpu();
}

/* NMFGPU_EXPORT */ nmfgpu::ResultType nmfgpu_get_information_for_gpu_index(unsigned index, nmfgpu::GpuInformation* info) {
	if (info == nullptr) {
		return nmfgpu::ResultType::ErrorInvalidArgument;
	}
	return nmfgpu::getInformationForGpuIndex(index, *info);
}