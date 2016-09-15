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

#include <common/Logging.h>

namespace nmfgpu {
	namespace details {
#define CASE_RETURN_ENUM_STRING(x) case x: return #x " (" NMFGPU_STRINGIFY(x) ")" 

		const char* cublasGetErrorEnum(cublasStatus_t error) {
			switch (error) {
			CASE_RETURN_ENUM_STRING(CUBLAS_STATUS_SUCCESS);
			CASE_RETURN_ENUM_STRING(CUBLAS_STATUS_NOT_INITIALIZED);
			CASE_RETURN_ENUM_STRING(CUBLAS_STATUS_ALLOC_FAILED);
			CASE_RETURN_ENUM_STRING(CUBLAS_STATUS_INVALID_VALUE);
			CASE_RETURN_ENUM_STRING(CUBLAS_STATUS_ARCH_MISMATCH);
			CASE_RETURN_ENUM_STRING(CUBLAS_STATUS_MAPPING_ERROR);
			CASE_RETURN_ENUM_STRING(CUBLAS_STATUS_EXECUTION_FAILED);
			CASE_RETURN_ENUM_STRING(CUBLAS_STATUS_INTERNAL_ERROR);
			CASE_RETURN_ENUM_STRING(CUBLAS_STATUS_NOT_SUPPORTED);
			CASE_RETURN_ENUM_STRING(CUBLAS_STATUS_LICENSE_ERROR);
			default: return "<UNKNOWN>";
			}
		}

		const char* cusparseGetErrorEnum(cusparseStatus_t error) {
			switch (error) {
			CASE_RETURN_ENUM_STRING(CUSPARSE_STATUS_NOT_INITIALIZED);
			CASE_RETURN_ENUM_STRING(CUSPARSE_STATUS_ALLOC_FAILED);
			CASE_RETURN_ENUM_STRING(CUSPARSE_STATUS_INVALID_VALUE);
			CASE_RETURN_ENUM_STRING(CUSPARSE_STATUS_ARCH_MISMATCH);
			CASE_RETURN_ENUM_STRING(CUSPARSE_STATUS_MAPPING_ERROR);
			CASE_RETURN_ENUM_STRING(CUSPARSE_STATUS_EXECUTION_FAILED);
			CASE_RETURN_ENUM_STRING(CUSPARSE_STATUS_INTERNAL_ERROR);
			CASE_RETURN_ENUM_STRING(CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
			default: return "<UNKNOWN>";
			}
		}

		const char* cusolverGetErrorEnum(cusolverStatus_t error) {
			switch (error) {
			CASE_RETURN_ENUM_STRING(CUSOLVER_STATUS_NOT_INITIALIZED);
			CASE_RETURN_ENUM_STRING(CUSOLVER_STATUS_ALLOC_FAILED);
			CASE_RETURN_ENUM_STRING(CUSOLVER_STATUS_INVALID_VALUE);
			CASE_RETURN_ENUM_STRING(CUSOLVER_STATUS_ARCH_MISMATCH);
			CASE_RETURN_ENUM_STRING(CUSOLVER_STATUS_EXECUTION_FAILED);
			CASE_RETURN_ENUM_STRING(CUSOLVER_STATUS_INTERNAL_ERROR);
			CASE_RETURN_ENUM_STRING(CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
			default: return "<UNKNOWN>";
			}
		}

		const char* curandGetErrorEnum(curandStatus_t error) {
			switch (error) {
			CASE_RETURN_ENUM_STRING(CURAND_STATUS_VERSION_MISMATCH);
			CASE_RETURN_ENUM_STRING(CURAND_STATUS_NOT_INITIALIZED);
			CASE_RETURN_ENUM_STRING(CURAND_STATUS_ALLOCATION_FAILED);
			CASE_RETURN_ENUM_STRING(CURAND_STATUS_TYPE_ERROR);
			CASE_RETURN_ENUM_STRING(CURAND_STATUS_OUT_OF_RANGE);
			CASE_RETURN_ENUM_STRING(CURAND_STATUS_LENGTH_NOT_MULTIPLE);
			CASE_RETURN_ENUM_STRING(CURAND_STATUS_DOUBLE_PRECISION_REQUIRED);
			CASE_RETURN_ENUM_STRING(CURAND_STATUS_LAUNCH_FAILURE);
			CASE_RETURN_ENUM_STRING(CURAND_STATUS_PREEXISTING_FAILURE);
			CASE_RETURN_ENUM_STRING(CURAND_STATUS_INITIALIZATION_FAILED);
			CASE_RETURN_ENUM_STRING(CURAND_STATUS_ARCH_MISMATCH);
			CASE_RETURN_ENUM_STRING(CURAND_STATUS_INTERNAL_ERROR);
			default: return "<UNKNOWN>";
			}
		}
	}
	
	std::ostream& operator << (std::ostream& stream, cublasStatus_t err) {
		stream << details::cublasGetErrorEnum(err);
		return stream;
	}

	std::ostream& operator << (std::ostream& stream, curandStatus_t err) {
		stream << details::curandGetErrorEnum(err);
		return stream;
	}

	std::ostream& operator << (std::ostream& stream, cusparseStatus_t err) {
		stream << details::cusparseGetErrorEnum(err);
		return stream;
	}

	std::ostream& operator << (std::ostream& stream, cusolverStatus_t err) {
		stream << details::cusolverGetErrorEnum(err);
		return stream;
	}

	LoggingBlock::~LoggingBlock() {
		m_stream << std::flush;
	}

	bool LoggingBlock::allowed() const {
		return static_cast<std::underlying_type<Verbosity>::type>(Logging::instance().verbosity()) >=
			static_cast<std::underlying_type<Verbosity>::type>(m_requiredVerbosity);
	}

	LoggingBlock& LoggingBlock::lineFeed() {
		return print("\n");
	}

	LoggingBlock& LoggingBlock::carriageReturn() {
		return print('\r');
	}

	/* static */ Logging& Logging::instance() {
		static Logging singleton;
		return singleton;
	}
}
