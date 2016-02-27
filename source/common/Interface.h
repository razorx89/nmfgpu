#pragma once

#include <nmfgpu.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cusolverDn.h>
#include <memory>

#ifdef WIN32
	#define THREAD_LOCAL_STORAGE __declspec(thread)
#else
	#define THREAD_LOCAL_STORAGE __thread
#endif

namespace nmfgpu {
	struct ExternalLibrariesContext {
		int deviceID{ 0 };
		cublasHandle_t cublasHandle;
		cusparseHandle_t cusparseHandle;
		cusolverDnHandle_t cusolverDnHandle;
	};

	extern THREAD_LOCAL_STORAGE ExternalLibrariesContext* g_context;

	ResultType initializeLibraryContexts();
	ResultType finalizeLibraryContexts();

	int getParameterIndexFromName(Parameter* parameters, unsigned numParameters, const char* name);
}