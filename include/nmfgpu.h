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

/** @file **/

#pragma once

#include <cstddef>

#define NMFGPU_MAJOR 0
#define NMFGPU_MINOR 2
#define NMFGPU_PATCH 3
#define NMFGPU_VERSION ((NMFGPU_MAJOR << 24) | (NMFGPU_MINOR << 16) | NMFGPU_PATCH)

#ifdef WIN32
	#ifdef NMFGPU_STATIC_LINKING
		#define NMFGPU_EXPORT
	#else
		#ifdef NMFGPU_EXPORTING
			#define NMFGPU_EXPORT __declspec(dllexport)
		#else
			#define NMFGPU_EXPORT __declspec(dllimport)
		#endif
	#endif
#else
	#define NMFGPU_EXPORT
#endif

#pragma pack(push, 4)

/** A */
namespace nmfgpu {
	/** Specifies several return values for the API which should be checked after calling a function. */
	enum class ResultType {
		/** Function call was successful and no error occured. */
		Success = 0,

		ErrorAlreadyInitialized,

		/** Library has not been initialized before calling a computational function. */
		ErrorNotInitialized,

		/** Function call had an invalid argument. */
		ErrorInvalidArgument,

		/** Algorithm can not allocate enough host memory for the specified problem size. */
		ErrorNotEnoughHostMemory,

		/** Algorithm can not allocate enough device memory for the specified problem size. */
		ErrorNotEnoughDeviceMemory,

		/** An error has occured in using an external library. */
		ErrorExternalLibrary,

		/** Algorithm was aborted through the CheckUserInterruptCallback invokation. */
		ErrorUserInterrupt,

		ErrorDeviceSelection,
	};

	/** Available initialization methods for the matrices W and H. */
	enum class NmfInitializationMethod {
		/** Initializes the matrices W and H according to the values passed in NMFParameter::W and NMFParameter::H. */
		CopyExisting,

		/** Initializes the matrices W and H using a random value generator. */
		AllRandomValues,

		/** Initializes the matrix W by averaging random columns of V and matrix H with random numbers. */
		MeanColumns,

		KMeansAndRandomValues,

		/** Initializes the matrix W by computing the k-Means and matrix H by computing @f$abs(W^T * V)@f$. */
		KMeansAndAbsoluteWTV,

		/** Initializes the matrix W by computing the k-Means and matrix H by computing @f$max(0, W^T * V)@f$. */
		KMeansAndNonNegativeWTV,

		/** Initializes the matrix W by computing the k-Means and matrix H by computing @f$h_{kq} = 1/\left[\sum^{k}_{k'=1}\frac{d\left(x_q,c_{k'}\right)}{d\left(x_q,c_k\right)}^{2/\left(1-m\right)}\right]@f$ */
		EInNMF,
	};

	enum class NmfThresholdType {
		Frobenius,
		RMSD
	};

	enum class NmfAlgorithm {
		Multiplicative,
		GDCLS,
		ALS,
		ACLS,
		AHCLS,
		nsNMF,
	};

	/** */
	enum class Verbosity {
		/** Permits only error messages to be displayed. */
		None,
		/** Shows a brief summary of the algorithm execution in form of a table. */
		Summary,
		/** Shows detailed information about convergence checks of each algorithm execution. */
		Informative,
		/** Shows a very large amount of information for debugging algorithms. */
		Debugging
	};

	/** Defines if indices are 0-based or 1-based. */
	enum class IndexBase {
		/** Indices are 0-based. */
		Zero,
		/** Indices are 1-based. */
		One
	};

	typedef bool(*UserInterruptCallback)();

	struct ExecutionStatistic {
		double frobenius;
		double rmsd;
		double elapsedTime;
		double sparsityW;
		double sparsityH;
		unsigned numIterations;
	};

	typedef ExecutionStatistic ExecutionRecord;

	class ISummary {
	public:
		NMFGPU_EXPORT static ISummary* create();

		virtual void destroy() = 0;

		/** Retrieves the index of the best run, which can be used to optain the record of the best run using nmfgpu::ISummary::record(unsigned).
		@returns Index of the best run. */
		virtual unsigned bestRun() const = 0;

		/**
		@param[in] index
		@param[out] record
		@returns */
		virtual void record(unsigned index, ExecutionRecord& record) const = 0;

		/** Retrieves the count of saved records, which will be at most the configured number of runs. If the user has interrupted the
		execution then the count of stored records can be less than the maximum number of runs.

		@returns Count of stored records.

		@see nmfgpu::IContext::setRunCount()*/
		virtual unsigned recordCount() const = 0;

	protected:
		virtual ~ISummary() { }
	};

	enum class StorageFormat {
		/** Dense Matrix */
		Dense,
		/** Compressed Sparse Row */
		CSR,
		/** Compressed Sparse Column */
		CSC,
		/** Coordinate */
		COO
	};

	template<typename NumericType>
	struct MatrixDescription {
		/** Number of rows of the matrix */
		unsigned rows;

		/** Number of columns of the matrix. */
		unsigned columns;

		/** Storage format of the matrix. */
		StorageFormat format;
		union {
			/** Description of a column-major dense matrix if the storage format is equal to StorageFormat::Dense. */
			struct {
				NumericType* values;
				unsigned leadingDimension;
			} dense;

			/** Description of a compressed sparse row matrix if the storage format is equal to StorageFormat::CSR. */
			struct {
				NumericType* values;
				int* rowPtr;
				int* columnIndices;
				unsigned nnz;
				IndexBase base;
			} csr;

			/** Description of a compressed sparse column matrix if the storage format is equal to StorageFormat::CSC. */
			struct {
				NumericType* values;
				int* columnPtr;
				int* rowIndices;
				unsigned nnz;
				IndexBase base;
			} csc;

			/** Description of a sparse matrix stored in coordinate format, equal to an uncompressed StorageFormat::CSR format. */
			struct {
				NumericType* values;
				int* rowIndices;
				int* columnIndices;
				unsigned nnz;
				IndexBase base;
			} coo;
		};
	};

	struct Parameter {
		const char* name;
		double value;
	};

	template<typename NumericType>
	struct NmfDescription {
		/** Chooses the algorithm for the non-negative matrix factorization. @see NmfAlgorithm */
		NmfAlgorithm algorithm;

		/** If this flag is set to true, then the matrix W in `outputMatrixW` will be used and remain constant during the factorization process. */
		bool useConstantBasisVectors;

		/** Input data matrix with attributes in rows and examples in columns. Can be either a dense matrix or sparse matrix. */
		MatrixDescription<NumericType> inputMatrix;

		/** Optional array with labels/classes for the dataset examples. Must have a length of at least `inputMatrix.columns` labels. If it is used during
		the factorization process depends on the chosen algorithm.*/
		int* inputLabels;

		/** Output matrix W of the factorization, which must have a dense matrix storage format. */
		MatrixDescription<NumericType> outputMatrixW;

		/** Output matrix H of the factorization, which must have a dense matrix storage format. */
		MatrixDescription<NumericType> outputMatrixH;

		/** Number of features to extract from the input data matrix. */
		unsigned features;

		NmfInitializationMethod initMethod;
		unsigned numIterations;
		unsigned numRuns;
		unsigned seed;
		NmfThresholdType thresholdType;
		double thresholdValue;
		UserInterruptCallback callbackUserInterrupt;

		Parameter* parameters;
		unsigned numParameters;
	};

	/** A*/
	NMFGPU_EXPORT ResultType initialize();

	/** A*/
	NMFGPU_EXPORT ResultType finalize();

	NMFGPU_EXPORT int version();

	NMFGPU_EXPORT ResultType chooseGpu(unsigned index);

	NMFGPU_EXPORT unsigned getNumberOfGpu();

	struct GpuInformation {
		char name[256];
		size_t totalMemory;
		size_t freeMemory;
	};

	NMFGPU_EXPORT ResultType getInformationForGpuIndex(unsigned index, GpuInformation& info);

	/** A*/
	NMFGPU_EXPORT void setVerbosity(Verbosity verbosity);

	NMFGPU_EXPORT ResultType compute(NmfDescription<float>& description, ISummary* summary);
	NMFGPU_EXPORT ResultType compute(NmfDescription<double>& description, ISummary* summary);

	template<typename NumericType>
	struct KMeansDescription {
		MatrixDescription<NumericType> inputMatrix;
		MatrixDescription<NumericType> outputMatrixClusters;
		unsigned* outputMemberships;
		unsigned numClusters;
		unsigned numIterations;
		unsigned seed;
		double thresholdValue;
	};

	struct KMeansSummary {
		/** Number of iterations which have been computed. */
		unsigned iterations;

		/** Sum of squares between clusters. */
		double betweenSS;

		/** Array of size `KMeansDescription::numClusters` with the sum of squares within each cluster.*/
		double* withinSS;

		/** Total sum of squares within each cluster. */
		double totalWithinSS;

		/** Sum of `betweenSS` and `totalWithinSS`. */
		double totalSS;
	};

	NMFGPU_EXPORT ResultType computeKMeans(KMeansDescription<float>& desc, KMeansSummary* summary);
	NMFGPU_EXPORT ResultType computeKMeans(KMeansDescription<double>& desc, KMeansSummary* summary);
}

extern "C" {
	NMFGPU_EXPORT nmfgpu::ResultType nmfgpu_initialize();
	NMFGPU_EXPORT nmfgpu::ResultType nmfgpu_finalize();
	NMFGPU_EXPORT int nmfgpu_version();
	NMFGPU_EXPORT void nmfgpu_set_verbosity(nmfgpu::Verbosity verbosity);

	NMFGPU_EXPORT nmfgpu::ResultType nmfgpu_create_summary(nmfgpu::ISummary** summary);

	NMFGPU_EXPORT nmfgpu::ResultType nmfgpu_compute_single(nmfgpu::NmfDescription<float>* description, nmfgpu::ISummary* summary);
	NMFGPU_EXPORT nmfgpu::ResultType nmfgpu_compute_double(nmfgpu::NmfDescription<double>* description, nmfgpu::ISummary* summary);

	NMFGPU_EXPORT nmfgpu::ResultType nmfgpu_compute_kmeans_single(nmfgpu::KMeansDescription<float>* desc);
	NMFGPU_EXPORT nmfgpu::ResultType nmfgpu_compute_kmeans_double(nmfgpu::KMeansDescription<double>* desc);
	NMFGPU_EXPORT nmfgpu::ResultType nmfgpu_choose_gpu(unsigned);
	NMFGPU_EXPORT unsigned nmfgpu_get_number_of_gpu();
	NMFGPU_EXPORT nmfgpu::ResultType nmfgpu_get_information_for_gpu_index(unsigned, nmfgpu::GpuInformation*);
}

#pragma pack(pop)
