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