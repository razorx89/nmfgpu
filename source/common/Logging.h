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

#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cusparse.h>
#include <cusolverDn.h>
#include <iostream>
#include <nmfgpu.h>

namespace nmfgpu {
	std::ostream& operator << (std::ostream& stream, cublasStatus_t err);
	std::ostream& operator << (std::ostream& stream, curandStatus_t err);
	std::ostream& operator << (std::ostream& stream, cusparseStatus_t err);
	std::ostream& operator << (std::ostream& stream, cusolverStatus_t err);

	class LoggingBlock { 
		/** Required verbosity to allow output to be written. */
		Verbosity m_requiredVerbosity;

		std::ostream& m_stream;

		/** Extracts one element from the variadic template argument stack and prints it to the standard output.
		@tparam Arg
		@tparam Remaining
		@param arg
		@param remaining
		@return */
		template<typename Arg>
		LoggingBlock& printUnpack(Arg&& arg) {
			if (allowed()) {
				m_stream << std::forward<Arg>(arg);
			}
			return *this;
		}

		/** Extracts one element from the variadic template argument stack and prints it to the standard output.
		@tparam Arg
		@tparam Remaining
		@param arg
		@param remaining 
		@return */
		template<typename Arg, typename... Remaining>
		LoggingBlock& printUnpack(Arg&& arg, Remaining&&... remaining) {
			printUnpack(std::forward<Arg>(arg));
			return printUnpack(std::forward<Remaining>(remaining)...);
		}

	public:
		LoggingBlock(Verbosity verbosity, std::ostream& stream)
			: m_requiredVerbosity(verbosity)
			, m_stream(stream) { };

		~LoggingBlock();

		/** Checks if the required verbosity is set. */
		bool allowed() const;

		template<typename... Args>
		LoggingBlock& print(Args&&... args) {
			return printUnpack(std::forward<Args>(args)...);
		}

		LoggingBlock& lineFeed();
		LoggingBlock& carriageReturn();
	};

	class Logging {
		Verbosity m_verbosity{ Verbosity::Summary };

		Logging() = default;
		
		Logging(const Logging&) = delete;
		Logging(Logging&&) = delete;
		Logging& operator= (const Logging&) = delete;
		Logging& operator= (Logging&&) = delete;

	public:
		static Logging& instance();

		void setVerbosity(Verbosity verbosity) { m_verbosity = verbosity; }
		Verbosity verbosity() const { return m_verbosity; }

		LoggingBlock error() { return LoggingBlock{Verbosity::None, std::cerr}; }
		LoggingBlock summary() { return LoggingBlock{ Verbosity::Summary, std::cout }; }
		LoggingBlock info() { return LoggingBlock{ Verbosity::Informative, std::cout }; }
		LoggingBlock debug() { return LoggingBlock{ Verbosity::Debugging, std::cout }; }
	};
}

#define NMFGPU_STRINGIFY(x) NMFGPU_STRINGIFY_HELPER(x)
#define NMFGPU_STRINGIFY_HELPER(x) #x
#define NMFGPU_FILE_LINE_PREFIX __FILE__ "(" NMFGPU_STRINGIFY(__LINE__) ")"

#ifdef _DEBUG
#	define NMFGPU_PRINT_ERROR_AND_CALL(err, call) .lineFeed().print("  Error: ", err).lineFeed().print("  Call: ", #call).lineFeed()
#else
#	define NMFGPU_PRINT_ERROR_AND_CALL(err, call) .print(" - ", err).lineFeed()
#endif

#define CUDA_CALL(call) { \
	cudaError_t err = (call); \
	if(err != cudaSuccess) { \
		Logging::instance().error() \
			.print("[ERROR] " NMFGPU_FILE_LINE_PREFIX ": Call to CUDA API failed") \
			NMFGPU_PRINT_ERROR_AND_CALL(err, call); \
	} \
}

#define CUBLAS_CALL(call) { \
	cublasStatus_t err = (call); \
	if(err != CUBLAS_STATUS_SUCCESS) { \
		Logging::instance().error() \
			.print("[ERROR] " NMFGPU_FILE_LINE_PREFIX ": Call to cuBLAS API failed") \
			NMFGPU_PRINT_ERROR_AND_CALL(err, call); \
	} \
}

#define CUSPARSE_CALL(call) { \
	cusparseStatus_t err = (call); \
	if(err != CUSPARSE_STATUS_SUCCESS) { \
		Logging::instance().error() \
			.print("[ERROR] " NMFGPU_FILE_LINE_PREFIX ": Call to cuSPARSE API failed") \
			NMFGPU_PRINT_ERROR_AND_CALL(err, call); \
	} \
}

#define CURAND_CALL(call) { \
	curandStatus_t err = (call); \
	if(err != CURAND_STATUS_SUCCESS) { \
		Logging::instance().error() \
			.print("[ERROR] " NMFGPU_FILE_LINE_PREFIX ": Call to cuRAND API failed") \
			NMFGPU_PRINT_ERROR_AND_CALL(err, call); \
	} \
}

#define CUSOLVER_CALL(call) { \
	cusolverStatus_t err = (call); \
	if(err != CUSOLVER_STATUS_SUCCESS) { \
		Logging::instance().error() \
			.print("[ERROR] " NMFGPU_FILE_LINE_PREFIX ": Call to cuSOLVER API failed") \
			NMFGPU_PRINT_ERROR_AND_CALL(err, call); \
	} \
}
